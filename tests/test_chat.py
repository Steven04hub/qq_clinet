"""Model-enabled chat behavior, without network calls or application settings."""

import asyncio
from copy import deepcopy

import pytest

from app.handlers.chat import ChatHandler
from app.onebot.models import GroupMessageEvent
from app.services.llm import LLMError
from app.services.session import MemorySessionStore


def group_event(*, text="什么是 Transformer？", group_id=10, message_id=1):
    return GroupMessageEvent.model_validate(
        {
            "post_type": "message",
            "message_type": "group",
            "message_id": message_id,
            "group_id": group_id,
            "user_id": 200,
            "self_id": 100,
            "message": [
                {"type": "at", "data": {"qq": "100"}},
                {"type": "text", "data": {"text": text}},
            ],
            "sender": {"user_id": 200, "nickname": "张三"},
        }
    )


class RecordingLLM:
    def __init__(self, answer="这是一条模型回答。", error=None):
        self.answer = answer
        self.error = error
        self.calls = []

    async def chat(self, messages):
        self.calls.append(deepcopy(messages))
        if self.error is not None:
            raise self.error
        return self.answer


class RecordingClient:
    def __init__(self):
        self.messages = []

    async def send_group_message(self, group_id, message):
        self.messages.append((group_id, message))
        return len(self.messages)


def sent_text(client, index=0, *, message_id=1, group_id=10):
    sent_group, segments = client.messages[index]
    assert sent_group == group_id
    assert segments[0] == {"type": "reply", "data": {"id": str(message_id)}}
    assert segments[1]["type"] == "text"
    return segments[1]["data"]["text"]


async def stop_tasks(*tasks):
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


async def test_model_receives_configured_system_prompt_and_identified_user():
    client = RecordingClient()
    llm = RecordingLLM(answer="回答 [CQ:at,qq=all] 作为文字。")
    handler = ChatHandler(client, llm=llm, system_prompt="使用简洁中文回答。")
    await handler.handle(group_event())
    assert llm.calls == [
        [
            {"role": "system", "content": "使用简洁中文回答。"},
            {"role": "user", "content": "[张三/200] 什么是 Transformer？"},
        ]
    ]
    assert len(client.messages) == 1
    assert sent_text(client) == "回答 [CQ:at,qq=all] 作为文字。"


async def test_model_chat_reuses_shared_group_history_and_clear_removes_it():
    client = RecordingClient()
    llm = RecordingLLM()
    store = MemorySessionStore(max_messages=20)
    handler = ChatHandler(client, llm=llm, session_store=store)
    await handler.handle(group_event(text="第一问", message_id=1))
    await handler.handle(group_event(text="第二问", message_id=2))
    assert len(llm.calls) == 2
    assert len(llm.calls[1]) == 4
    assert llm.calls[1][1]["content"].endswith("第一问")
    assert llm.calls[1][2]["role"] == "assistant"
    await store.clear("group:10")
    assert await store.get("group:10") == []


async def test_echo_diagnostic_does_not_call_configured_model():
    client = RecordingClient()
    llm = RecordingLLM()
    await ChatHandler(client, llm=llm).handle(group_event(text="echo hello"))
    assert llm.calls == []
    assert sent_text(client) == "hello"


async def test_no_model_retains_echo_mode():
    client = RecordingClient()
    await ChatHandler(client, llm=None).handle(group_event(text="hello"))
    assert "echo" in sent_text(client)


async def test_oversized_input_gets_short_reply_without_model_call():
    client = RecordingClient()
    llm = RecordingLLM()
    await ChatHandler(client, llm=llm, max_input_chars=8).handle(group_event(text="x" * 9))
    assert llm.calls == []
    assert len(client.messages) == 1
    assert 0 < len(sent_text(client)) < 200


async def test_input_at_limit_is_accepted():
    client = RecordingClient()
    llm = RecordingLLM()
    await ChatHandler(client, llm=llm, max_input_chars=8).handle(group_event(text="x" * 8))
    assert len(llm.calls) == 1


async def test_empty_text_has_no_model_call_or_reply():
    client = RecordingClient()
    llm = RecordingLLM()
    await ChatHandler(client, llm=llm).handle(group_event(text="  \n "))
    assert llm.calls == []
    assert client.messages == []


async def test_long_model_output_is_capped_at_three_quoted_segments():
    client = RecordingClient()
    llm = RecordingLLM(answer="内容。" * 100)
    await ChatHandler(client, llm=llm, max_reply_chars=20).handle(group_event())
    assert len(client.messages) == 3
    for index in range(3):
        assert 0 < len(sent_text(client, index)) <= 20
    assert "截断" in sent_text(client, 2)


@pytest.mark.parametrize(
    "kind",
    [
        "timeout",
        "auth",
        "balance",
        "rate_limit",
        "api",
        "connection",
        "empty",
        "malformed",
        "unknown",
    ],
)
async def test_model_errors_send_only_a_short_safe_reply(kind):
    client = RecordingClient()
    error = LLMError(kind)
    # Exception text can originate outside this handler; never echo it into QQ.
    error.args = ("provider detail Authorization: Bearer sk-private-test-token",)
    llm = RecordingLLM(error=error)
    await ChatHandler(client, llm=llm).handle(group_event())
    assert len(client.messages) == 1
    answer = sent_text(client)
    assert 0 < len(answer) < 200
    assert "sk-private-test-token" not in answer
    assert "Authorization" not in answer
    assert "provider detail" not in answer


async def test_unknown_model_exception_gets_safe_fallback():
    client = RecordingClient()
    llm = RecordingLLM(error=RuntimeError("secret=sk-private-test-token"))
    await ChatHandler(client, llm=llm).handle(group_event())
    assert len(client.messages) == 1
    answer = sent_text(client)
    assert 0 < len(answer) < 200
    assert "sk-private-test-token" not in answer


@pytest.mark.parametrize("llm_error", [None, LLMError("timeout")])
async def test_send_failure_propagates_without_a_second_reply(llm_error):
    class FailingClient(RecordingClient):
        async def send_group_message(self, group_id, message):
            await super().send_group_message(group_id, message)
            raise ConnectionError("WebSocket disconnected")

    client = FailingClient()
    llm = RecordingLLM(error=llm_error)
    with pytest.raises(ConnectionError):
        await ChatHandler(client, llm=llm).handle(group_event())
    assert len(client.messages) == 1
    assert len(llm.calls) == 1


async def test_same_group_serializes_model_calls():
    class BlockingLLM(RecordingLLM):
        def __init__(self):
            super().__init__()
            self.entered = asyncio.Event()
            self.release = asyncio.Event()

        async def chat(self, messages):
            self.calls.append(deepcopy(messages))
            self.entered.set()
            await self.release.wait()
            return "answer"

    client = RecordingClient()
    llm = BlockingLLM()
    handler = ChatHandler(client, llm=llm)
    first = asyncio.create_task(handler.handle(group_event(message_id=1)))
    second = None
    try:
        async with asyncio.timeout(2):
            await llm.entered.wait()
            second_started = asyncio.Event()

            async def second_call():
                second_started.set()
                await handler.handle(group_event(message_id=2))

            second = asyncio.create_task(second_call())
            await second_started.wait()
            done, _ = await asyncio.wait({second}, timeout=0.03)
            assert not done
            assert len(llm.calls) == 1
            llm.release.set()
            await asyncio.gather(first, second)
            assert len(llm.calls) == 2
            assert len(client.messages) == 2
    finally:
        await stop_tasks(*[task for task in (first, second) if task is not None])


async def test_same_group_remains_locked_until_sending_finishes():
    class BlockingClient(RecordingClient):
        def __init__(self):
            super().__init__()
            self.entered = asyncio.Event()
            self.release = asyncio.Event()

        async def send_group_message(self, group_id, message):
            self.entered.set()
            await self.release.wait()
            return await super().send_group_message(group_id, message)

    client = BlockingClient()
    llm = RecordingLLM()
    handler = ChatHandler(client, llm=llm)
    first = asyncio.create_task(handler.handle(group_event(message_id=1)))
    second = None
    try:
        async with asyncio.timeout(2):
            await client.entered.wait()
            second_started = asyncio.Event()

            async def second_call():
                second_started.set()
                await handler.handle(group_event(message_id=2))

            second = asyncio.create_task(second_call())
            await second_started.wait()
            done, _ = await asyncio.wait({second}, timeout=0.03)
            assert not done
            assert len(llm.calls) == 1
            client.release.set()
            await asyncio.gather(first, second)
            assert len(llm.calls) == 2
    finally:
        await stop_tasks(*[task for task in (first, second) if task is not None])


async def test_different_groups_can_call_model_concurrently():
    class BlockingLLM(RecordingLLM):
        def __init__(self):
            super().__init__()
            self.entered = asyncio.Queue()
            self.release = asyncio.Event()

        async def chat(self, messages):
            self.calls.append(deepcopy(messages))
            await self.entered.put(messages)
            await self.release.wait()
            return "answer"

    client = RecordingClient()
    llm = BlockingLLM()
    handler = ChatHandler(client, llm=llm)
    tasks = [
        asyncio.create_task(handler.handle(group_event(group_id=group_id))) for group_id in (10, 20)
    ]
    try:
        async with asyncio.timeout(2):
            await llm.entered.get()
            await llm.entered.get()
            assert len(llm.calls) == 2
            llm.release.set()
            await asyncio.gather(*tasks)
            assert {group_id for group_id, _ in client.messages} == {10, 20}
    finally:
        await stop_tasks(*tasks)


async def test_group_queue_timeout_replies_without_model_call():
    lock = asyncio.Lock()
    await lock.acquire()
    client = RecordingClient()
    llm = RecordingLLM()
    handler = ChatHandler(client, llm=llm, group_locks={10: lock}, queue_timeout_seconds=0.01)
    try:
        async with asyncio.timeout(1):
            await handler.handle(group_event())
        assert llm.calls == []
        assert len(client.messages) == 1
        assert 0 < len(sent_text(client)) < 200
        assert lock.locked(), "A waiter must not release another request's lock"
    finally:
        lock.release()


async def test_cancelled_model_call_propagates_and_releases_group_lock():
    entered = asyncio.Event()

    class CancelledLLM(RecordingLLM):
        async def chat(self, messages):
            entered.set()
            await asyncio.Event().wait()

    lock = asyncio.Lock()
    client = RecordingClient()
    handler = ChatHandler(client, llm=CancelledLLM(), group_locks={10: lock})
    task = asyncio.create_task(handler.handle(group_event()))
    try:
        async with asyncio.timeout(2):
            await entered.wait()
            assert lock.locked()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert not lock.locked()
        assert client.messages == []
    finally:
        await stop_tasks(task)


async def test_cancelled_send_propagates_and_releases_group_lock():
    entered = asyncio.Event()

    class BlockingClient(RecordingClient):
        async def send_group_message(self, group_id, message):
            entered.set()
            await asyncio.Event().wait()

    lock = asyncio.Lock()
    client = BlockingClient()
    handler = ChatHandler(client, llm=RecordingLLM(), group_locks={10: lock})
    task = asyncio.create_task(handler.handle(group_event()))
    try:
        async with asyncio.timeout(2):
            await entered.wait()
            assert lock.locked()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert not lock.locked()
        assert client.messages == []
    finally:
        await stop_tasks(task)


async def test_cancelling_waiting_request_does_not_release_another_requests_lock():
    lock = asyncio.Lock()
    await lock.acquire()
    client = RecordingClient()
    llm = RecordingLLM()
    handler = ChatHandler(client, llm=llm, group_locks={10: lock})
    started = asyncio.Event()

    async def call():
        started.set()
        await handler.handle(group_event())

    task = asyncio.create_task(call())
    try:
        async with asyncio.timeout(2):
            await started.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert lock.locked()
        assert llm.calls == []
        assert client.messages == []
    finally:
        lock.release()
        await stop_tasks(task)
