"""Real loopback WebSocket and command-line checks without NapCat or QQ."""

import asyncio
import json
import os
import signal
import sys
from pathlib import Path

import pytest
from websockets.asyncio.server import serve

from app.onebot.client import OneBotClient

pytestmark = pytest.mark.asyncio

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_TOKEN = "runtime-test-token"
TEST_GROUP_ID = 123456
TEST_EVENT = {
    "post_type": "message",
    "message_type": "group",
    "message_id": 789,
    "group_id": TEST_GROUP_ID,
    "user_id": 222,
    "self_id": 111,
    "message": [{"type": "text", "data": {"text": "private chat content"}}],
    "sender": {"user_id": 222, "nickname": "Test user"},
}


def server_url(server) -> str:
    return f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}"


async def start_cli(tmp_path: Path, url: str, *args: str, overrides: dict | None = None):
    # Do not inherit developer credentials, app settings, or a project .env.
    env = {
        "PATH": os.environ.get("PATH", os.defpath),
        "PYTHONPATH": str(PROJECT_ROOT),
        "PYTHONUNBUFFERED": "1",
        "PYTHONASYNCIODEBUG": "1",
        "NAPCAT_WS_URL": url,
        "NAPCAT_ACCESS_TOKEN": TEST_TOKEN,
        "ALLOWED_GROUP_IDS": str(TEST_GROUP_ID),
        "ADMIN_QQ_IDS": "",
        "LLM_API_KEY": "",
        "LLM_BASE_URL": "",
        "LLM_MODEL": "",
        "CONNECT_TIMEOUT_SECONDS": "2",
        "ACTION_TIMEOUT_SECONDS": "2",
        "LOG_LEVEL": "INFO",
    }
    env.update(overrides or {})
    return await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "app.main",
        *args,
        cwd=tmp_path,
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )


async def cleanup_process(process) -> None:
    if process.returncode is None:
        process.kill()
    await process.communicate()


def assert_clean_output(output: str) -> None:
    for unexpected in (
        "Traceback",
        "Task was destroyed",
        "Task exception was never retrieved",
        "RuntimeWarning",
        TEST_TOKEN,
        "private chat content",
    ):
        assert unexpected not in output


async def test_real_websocket_auth_and_interleaved_event_before_action_response():
    received = asyncio.Queue()
    events = asyncio.Queue()
    handled = asyncio.Event()

    async def server_handler(connection):
        request = json.loads(await connection.recv())
        await received.put((connection.request.headers["Authorization"], request))
        await connection.send(json.dumps(TEST_EVENT))
        await handled.wait()
        await connection.send(
            json.dumps(
                {"status": "ok", "retcode": 0, "data": {"online": True}, "echo": request["echo"]}
            )
        )
        await connection.wait_closed()

    async def handle_event(event):
        await events.put(event)
        handled.set()

    async with asyncio.timeout(5):
        async with serve(server_handler, "127.0.0.1", 0) as server:
            client = OneBotClient(server_url(server), TEST_TOKEN, event_handler=handle_event)
            try:
                await client.connect()
                response = await client.call_action("get_status", {})
                auth, request = await received.get()
                assert auth == f"Bearer {TEST_TOKEN}"
                assert request["action"] == "get_status"
                assert response["echo"] == request["echo"]
                assert response["data"] == {"online": True}
                assert await events.get() == TEST_EVENT
                assert not client.pending_requests
            finally:
                await client.close()


async def test_real_websocket_reconnects_and_resolves_action_on_second_connection():
    connections = 0
    authorizations = []
    second_connection = asyncio.Event()

    async def server_handler(connection):
        nonlocal connections
        connections += 1
        authorizations.append(connection.request.headers["Authorization"])
        if connections == 1:
            await connection.close(code=1012, reason="Simulated NapCat restart")
            return
        second_connection.set()
        request = json.loads(await connection.recv())
        assert request["action"] == "get_status"
        await connection.send(
            json.dumps(
                {"status": "ok", "retcode": 0, "data": {"online": True}, "echo": request["echo"]}
            )
        )
        await connection.wait_closed()

    async with asyncio.timeout(6):
        async with serve(server_handler, "127.0.0.1", 0) as server:
            client = OneBotClient(server_url(server), TEST_TOKEN)
            runner = asyncio.create_task(client.run())
            try:
                await second_connection.wait()
                await client.wait_until_connected(timeout=1)
                response = await client.call_action("get_status", {})
                assert response["data"] == {"online": True}
                assert connections == 2
                assert authorizations == [f"Bearer {TEST_TOKEN}"] * 2
                assert not client.pending_requests
                await client.close()
                await runner
                assert not client.is_connected
                assert connections == 2
            finally:
                await client.close()
                runner.cancel()
                await asyncio.gather(runner, return_exceptions=True)


async def test_cli_send_transmits_reply_and_literal_text_and_exits(tmp_path):
    received = asyncio.Queue()
    message = "local smoke [CQ:at,qq=all]"

    async def server_handler(connection):
        request = json.loads(await connection.recv())
        await received.put((connection.request.headers["Authorization"], request))
        await connection.send(json.dumps(TEST_EVENT))
        await connection.send(
            json.dumps(
                {
                    "status": "ok",
                    "retcode": 0,
                    "data": {"message_id": 900},
                    "echo": request["echo"],
                }
            )
        )
        await connection.wait_closed()

    async with asyncio.timeout(8):
        async with serve(server_handler, "127.0.0.1", 0) as server:
            process = await start_cli(
                tmp_path,
                server_url(server),
                "send",
                "--group-id",
                str(TEST_GROUP_ID),
                "--message",
                message,
                "--reply-to",
                "789",
            )
            try:
                raw_output, _ = await process.communicate()
                output = raw_output.decode()
                assert process.returncode == 0, output
                auth, request = await received.get()
                assert auth == f"Bearer {TEST_TOKEN}"
                assert request["action"] == "send_group_msg"
                assert request["params"] == {
                    "group_id": TEST_GROUP_ID,
                    "message": [
                        {"type": "reply", "data": {"id": "789"}},
                        {"type": "text", "data": {"text": message}},
                    ],
                }
                assert f"Sent group message group_id={TEST_GROUP_ID} message_id=900" in output
                assert "Bot stopped" in output
                assert_clean_output(output)
            finally:
                await cleanup_process(process)


@pytest.mark.parametrize("stop_signal", [signal.SIGTERM, signal.SIGINT])
async def test_cli_listener_logs_metadata_and_stops_cleanly_on_signal(tmp_path, stop_signal):
    actions = asyncio.Queue()

    async def server_handler(connection):
        await connection.send(json.dumps(TEST_EVENT))
        async for packet in connection:
            await actions.put(packet)

    async with asyncio.timeout(8):
        async with serve(server_handler, "127.0.0.1", 0) as server:
            process = await start_cli(tmp_path, server_url(server), "listen")
            try:
                lines = []
                while True:
                    line = await process.stdout.readline()
                    assert line, b"".join(lines).decode()
                    lines.append(line)
                    if b"Group message event" in line:
                        break
                process.send_signal(stop_signal)
                remainder, _ = await process.communicate()
                output = (b"".join(lines) + remainder).decode()
                assert process.returncode == 0, output
                assert f"group_id={TEST_GROUP_ID} user_id=222 message_id=789" in output
                assert "Bot stopped" in output
                assert actions.empty(), "The Phase 1 listener must not reply automatically"
                assert_clean_output(output)
            finally:
                await cleanup_process(process)


async def test_cli_refuses_nonallowlisted_group_before_connecting(tmp_path):
    connections = 0

    async def server_handler(connection):
        nonlocal connections
        connections += 1
        await connection.wait_closed()

    async with asyncio.timeout(8):
        async with serve(server_handler, "127.0.0.1", 0) as server:
            process = await start_cli(
                tmp_path,
                server_url(server),
                "send",
                "--group-id",
                "654321",
                "--message",
                "must not be sent",
            )
            try:
                raw_output, _ = await process.communicate()
                output = raw_output.decode()
                assert process.returncode == 2, output
                assert "not in ALLOWED_GROUP_IDS" in output
                assert connections == 0
                assert_clean_output(output)
            finally:
                await cleanup_process(process)


async def test_cli_echo_routes_mention_and_sends_quoted_literal_text(tmp_path):
    received = asyncio.Queue()

    async def server_handler(connection):
        event = {
            **TEST_EVENT,
            "message": [
                {"type": "at", "data": {"qq": "111"}},
                {"type": "text", "data": {"text": " echo hello [CQ:at,qq=all]"}},
            ],
        }
        await connection.send(json.dumps(event))
        request = json.loads(await connection.recv())
        await connection.send(
            json.dumps(
                {
                    "status": "ok",
                    "retcode": 0,
                    "echo": request["echo"],
                    "data": {"message_id": 901},
                }
            )
        )
        await received.put(request)
        await connection.wait_closed()

    async with asyncio.timeout(8):
        async with serve(server_handler, "127.0.0.1", 0) as server:
            process = await start_cli(tmp_path, server_url(server), "echo")
            try:
                request = await received.get()
                assert request["action"] == "send_group_msg"
                assert request["params"] == {
                    "group_id": TEST_GROUP_ID,
                    "message": [
                        {"type": "reply", "data": {"id": "789"}},
                        {"type": "text", "data": {"text": "hello [CQ:at,qq=all]"}},
                    ],
                }
                process.send_signal(signal.SIGTERM)
                raw_output, _ = await process.communicate()
                assert process.returncode == 0, raw_output.decode()
                assert_clean_output(raw_output.decode())
            finally:
                await cleanup_process(process)
