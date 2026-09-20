"""Exercise transport lifecycle and event/action multiplexing without QQ."""

import asyncio
from contextlib import suppress
from uuid import UUID

import pytest
import pytest_asyncio

import app.onebot.client as client_module
from app.onebot.actions import OneBotActionError, OneBotTimeoutError
from app.onebot.client import OneBotClient
from tests.fakes import FakeWebSocket

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def connected(monkeypatch):
    socket = FakeWebSocket()

    async def connect(*args, **kwargs):
        return socket

    monkeypatch.setattr(client_module, "ws_connect", connect)
    client = OneBotClient("ws://127.0.0.1:3001", "test-token")
    await client.connect()
    try:
        yield client, socket
    finally:
        await client.close()


async def test_action_request_has_uuid_echo_and_returns_full_response(connected):
    client, socket = connected
    pending = asyncio.create_task(client.call_action("get_status", {}))
    request = await socket.next_request()

    assert request["action"] == "get_status"
    assert request["params"] == {}
    assert str(UUID(request["echo"])) == request["echo"]
    assert request["echo"] in client.pending_requests

    response = socket.respond(request, data={"online": True})
    assert await asyncio.wait_for(pending, 1) == response
    assert not client.pending_requests


async def test_concurrent_responses_are_matched_by_echo_not_arrival_order(connected):
    client, socket = connected
    first = asyncio.create_task(client.call_action("first", {"index": 1}))
    second = asyncio.create_task(client.call_action("second", {"index": 2}))
    requests = {
        request["action"]: request
        for request in [await socket.next_request(), await socket.next_request()]
    }
    assert requests["first"]["echo"] != requests["second"]["echo"]

    second_response = socket.respond(requests["second"], data={"value": 2})
    assert await asyncio.wait_for(second, 1) == second_response
    assert not first.done()
    first_response = socket.respond(requests["first"], data={"value": 1})
    assert await asyncio.wait_for(first, 1) == first_response
    assert not client.pending_requests


async def test_event_before_action_response_reaches_handler(monkeypatch):
    socket = FakeWebSocket()
    events = asyncio.Queue()

    async def connect(*args, **kwargs):
        return socket

    async def handle(event):
        await events.put(event)

    monkeypatch.setattr(client_module, "ws_connect", connect)
    client = OneBotClient("ws://127.0.0.1:3001", "token", event_handler=handle)
    await client.connect()
    try:
        action = asyncio.create_task(client.call_action("get_status", {}))
        request = await socket.next_request()
        event = {"post_type": "message", "message_type": "group", "group_id": 42}
        socket.feed(event)
        assert await asyncio.wait_for(events.get(), 1) == event
        assert not action.done()
        response = socket.respond(request, data={"online": True})
        assert await asyncio.wait_for(action, 1) == response
        assert socket.max_receivers == 1
    finally:
        await client.close()


async def test_event_handler_can_await_action_without_blocking_receiver(monkeypatch):
    socket = FakeWebSocket()
    handler_result = asyncio.get_running_loop().create_future()

    async def connect(*args, **kwargs):
        return socket

    async def handle(event):
        try:
            result = await client.call_action("get_status", {})
            handler_result.set_result(result)
        except Exception as exc:
            handler_result.set_exception(exc)

    monkeypatch.setattr(client_module, "ws_connect", connect)
    client = OneBotClient("ws://127.0.0.1:3001", "token", event_handler=handle)
    await client.connect()
    try:
        socket.feed({"post_type": "meta_event", "meta_event_type": "heartbeat"})
        request = await socket.next_request()
        response = socket.respond(request, data={"online": True})
        assert await asyncio.wait_for(handler_result, 1) == response
        assert socket.max_receivers == 1
    finally:
        await client.close()


async def test_missing_response_times_out_and_removes_pending_request(connected):
    client, socket = connected
    action = asyncio.create_task(client.call_action("get_status", {}, timeout=0.02))
    await socket.next_request()
    with pytest.raises(OneBotTimeoutError):
        await asyncio.wait_for(action, 1)
    assert not client.pending_requests


async def test_action_timeout_also_bounds_a_blocked_send(connected):
    client, socket = connected
    socket.send_gate = asyncio.Event()
    action = asyncio.create_task(client.call_action("get_status", {}, timeout=0.02))
    await asyncio.wait_for(socket.send_started.wait(), 1)
    with pytest.raises(OneBotTimeoutError):
        await asyncio.wait_for(action, 1)
    assert not client.pending_requests


async def test_cancelled_action_is_removed_and_late_response_is_harmless(connected):
    client, socket = connected
    action = asyncio.create_task(client.call_action("cancel_me", {}))
    request = await socket.next_request()
    action.cancel()
    with pytest.raises(asyncio.CancelledError):
        await action
    assert not client.pending_requests

    socket.respond(request)
    next_action = asyncio.create_task(client.call_action("still_working", {}))
    next_request = await socket.next_request()
    response = socket.respond(next_request)
    assert await asyncio.wait_for(next_action, 1) == response


@pytest.mark.parametrize("disconnect_error", [None, ConnectionError("lost connection")])
async def test_disconnect_fails_every_pending_action(connected, disconnect_error):
    client, socket = connected
    actions = [asyncio.create_task(client.call_action(f"action_{i}", {})) for i in range(3)]
    for _ in actions:
        await socket.next_request()
    socket.disconnect(disconnect_error)
    results = await asyncio.wait_for(asyncio.gather(*actions, return_exceptions=True), 1)
    assert all(isinstance(result, ConnectionError) for result in results)
    assert not client.pending_requests
    assert not client.is_connected


async def test_close_cancels_pending_actions_and_closes_socket(connected):
    client, socket = connected
    action = asyncio.create_task(client.call_action("get_status", {}))
    await socket.next_request()
    await client.close()
    result = (await asyncio.gather(action, return_exceptions=True))[0]
    assert isinstance(result, asyncio.CancelledError)
    assert socket.closed
    assert not client.pending_requests
    assert not client.is_connected
    assert socket.receivers == 0
    await client.close()  # Closing twice is safe.


async def test_close_cancels_running_event_handlers(monkeypatch):
    socket = FakeWebSocket()
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def connect(*args, **kwargs):
        return socket

    async def handle(event):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    monkeypatch.setattr(client_module, "ws_connect", connect)
    client = OneBotClient("ws://127.0.0.1:3001", "token", event_handler=handle)
    await client.connect()
    try:
        socket.feed({"post_type": "notice"})
        await asyncio.wait_for(started.wait(), 1)
        await asyncio.wait_for(client.close(), 1)
        assert cancelled.is_set()
        assert socket.receivers == 0
    finally:
        await client.close()


async def test_malformed_json_nonobjects_and_unknown_echo_do_not_break_receiver(connected):
    client, socket = connected
    socket.feed_raw("not valid JSON")
    socket.feed_raw(b"\xff")
    for payload in [None, [], 1, "text", {"echo": "unknown"}, {"echo": []}]:
        socket.feed(payload)
    action = asyncio.create_task(client.call_action("get_status", {}))
    request = await socket.next_request()
    response = socket.respond(request)
    assert await asyncio.wait_for(action, 1) == response


@pytest.mark.parametrize("status,retcode", [("failed", 1400), ("failed", 0), ("ok", 1400)])
async def test_failed_action_response_raises_action_error(connected, status, retcode):
    client, socket = connected
    action = asyncio.create_task(client.call_action("send_group_msg", {"group_id": 42}))
    request = await socket.next_request()
    socket.respond(request, status=status, retcode=retcode, wording="rejected")
    with pytest.raises(OneBotActionError):
        await asyncio.wait_for(action, 1)
    assert not client.pending_requests


async def test_send_group_message_uses_literal_text_segment_and_returns_int(connected):
    client, socket = connected
    message = "Literal [CQ:at,qq=all] must stay text"
    action = asyncio.create_task(client.send_group_message(42, message))
    request = await socket.next_request()
    assert request["action"] == "send_group_msg"
    assert request["params"] == {
        "group_id": 42,
        "message": [{"type": "text", "data": {"text": message}}],
    }
    socket.respond(request, data={"message_id": 123})
    message_id = await asyncio.wait_for(action, 1)
    assert message_id == 123
    assert type(message_id) is int


async def test_send_group_message_preserves_segment_array(connected):
    client, socket = connected
    segments = [
        {"type": "reply", "data": {"id": "123"}},
        {"type": "text", "data": {"text": "reply text"}},
    ]
    action = asyncio.create_task(client.send_group_message(42, segments))
    request = await socket.next_request()
    assert request["params"]["message"] == segments
    socket.respond(request, data={"message_id": 456})
    assert await asyncio.wait_for(action, 1) == 456


async def test_action_when_disconnected_fails_immediately():
    client = OneBotClient("ws://127.0.0.1:3001", "token")
    try:
        with pytest.raises(ConnectionError):
            await asyncio.wait_for(client.call_action("get_status", {}), 1)
        assert not client.pending_requests
    finally:
        await client.close()


async def test_connect_uses_bearer_auth_and_disables_proxy(monkeypatch):
    socket = FakeWebSocket()
    arguments = []

    async def connect(*args, **kwargs):
        arguments.append((args, kwargs))
        return socket

    monkeypatch.setattr(client_module, "ws_connect", connect)
    client = OneBotClient("ws://127.0.0.1:3001", "test-token", connect_timeout=0.25)
    try:
        await client.connect()
        assert arguments[0][0] == ("ws://127.0.0.1:3001",)
        assert arguments[0][1]["additional_headers"] == {"Authorization": "Bearer test-token"}
        assert arguments[0][1]["proxy"] is None
        assert arguments[0][1]["open_timeout"] == 0.25
    finally:
        await client.close()


async def test_close_cancels_an_in_progress_connection_attempt(monkeypatch):
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def connect(*args, **kwargs):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    monkeypatch.setattr(client_module, "ws_connect", connect)
    client = OneBotClient("ws://127.0.0.1:3001", "token")
    runner = asyncio.create_task(client.run())
    try:
        await asyncio.wait_for(started.wait(), 1)
        await asyncio.wait_for(client.close(), 1)
        await asyncio.wait_for(runner, 1)
        assert cancelled.is_set()
        assert not client.is_connected
    finally:
        await client.close()
        runner.cancel()
        with suppress(asyncio.CancelledError):
            await runner


async def test_reconnect_backoff_is_capped_and_close_stops_retries(monkeypatch):
    attempts = 0
    delays = []

    async def connect(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        raise OSError("NapCat unavailable")

    monkeypatch.setattr(client_module, "ws_connect", connect)
    client = OneBotClient("ws://127.0.0.1:3001", "token")

    async def wait_backoff(delay):
        delays.append(delay)
        if len(delays) == 7:
            await client.close()

    monkeypatch.setattr(client, "_wait_backoff", wait_backoff)
    try:
        await asyncio.wait_for(client.run(), 1)
        assert delays == [1, 2, 4, 8, 15, 15, 15]
        assert attempts == 7
        assert not client.is_connected
    finally:
        await client.close()


async def test_successful_connection_resets_backoff(monkeypatch):
    attempts = 0
    delays = []
    socket = FakeWebSocket()
    socket.disconnect()

    async def connect(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts <= 2:
            raise OSError("NapCat unavailable")
        return socket

    monkeypatch.setattr(client_module, "ws_connect", connect)
    client = OneBotClient("ws://127.0.0.1:3001", "token")

    async def wait_backoff(delay):
        delays.append(delay)
        if len(delays) == 3:
            await client.close()

    monkeypatch.setattr(client, "_wait_backoff", wait_backoff)
    try:
        await asyncio.wait_for(client.run(), 1)
        assert attempts == 3
        assert delays == [1, 2, 1]
    finally:
        await client.close()


async def test_close_interrupts_backoff_without_another_connect_attempt(monkeypatch):
    attempts = 0
    waiting = asyncio.Event()

    async def connect(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        raise OSError("NapCat unavailable")

    monkeypatch.setattr(client_module, "ws_connect", connect)
    client = OneBotClient("ws://127.0.0.1:3001", "token")
    original_wait = client._wait_backoff

    async def wait_backoff(delay):
        waiting.set()
        await original_wait(30)

    monkeypatch.setattr(client, "_wait_backoff", wait_backoff)
    runner = asyncio.create_task(client.run())
    try:
        await asyncio.wait_for(waiting.wait(), 1)
        await client.close()
        await asyncio.wait_for(runner, 1)
        assert attempts == 1
    finally:
        await client.close()
        runner.cancel()
        with suppress(asyncio.CancelledError):
            await runner


async def test_run_after_connect_reuses_socket_and_single_receiver(monkeypatch):
    socket = FakeWebSocket()
    attempts = 0
    events = asyncio.Queue()

    async def connect(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        return socket

    async def handle(event):
        await events.put(event)

    monkeypatch.setattr(client_module, "ws_connect", connect)
    client = OneBotClient("ws://127.0.0.1:3001", "token")
    await client.connect()
    runner = asyncio.create_task(client.run(handle))
    try:
        await asyncio.sleep(0)
        await client.wait_until_connected(timeout=1)
        event = {"post_type": "meta_event"}
        socket.feed(event)
        assert await asyncio.wait_for(events.get(), 1) == event
        assert attempts == 1
        assert socket.max_receivers == 1
        await client.close()
        await asyncio.wait_for(runner, 1)
    finally:
        await client.close()
        runner.cancel()
        with suppress(asyncio.CancelledError):
            await runner
