"""Quiet-window scheduling with a deterministic clock and no live QQ/model calls."""

import asyncio
from copy import deepcopy

import pytest

from app.config import Settings
from app.services.control import BotController
from app.web.settings import ControlSettings, ControlSettingsStore
from tests.test_control import FakeClient, participation, payload


async def settle():
    """Let timer/model continuations run without advancing the simulated clock."""
    for _ in range(24):
        await asyncio.sleep(0)


class ManualClock:
    def __init__(self):
        self.now = 100.0
        self.waiters = []

    def __call__(self):
        return self.now

    async def sleep(self, delay):
        if delay <= 0:
            await asyncio.sleep(0)
            return
        future = asyncio.get_running_loop().create_future()
        waiter = (self.now + delay, future)
        self.waiters.append(waiter)
        try:
            await future
        finally:
            self.waiters.remove(waiter)

    async def advance(self, seconds):
        assert seconds >= 0
        self.now += seconds
        for deadline, future in list(self.waiters):
            if deadline <= self.now + 1e-9 and not future.done():
                future.set_result(None)
        await settle()


class WindowLLM:
    def __init__(self, clock, *, blocked=False, swallow_cancel=False):
        self.clock = clock
        self.blocked = blocked
        self.swallow_cancel = swallow_cancel
        self.calls = []
        self.call_times = []
        self.releases = []
        self.cancelled_calls = []
        self.answer = '{"should_reply":false,"reply":""}'

    async def chat_json(self, messages):
        index = len(self.calls)
        self.calls.append(deepcopy(messages))
        self.call_times.append(self.clock.now)
        release = asyncio.get_running_loop().create_future()
        self.releases.append(release)
        if not self.blocked:
            release.set_result(self.answer)
        while True:
            try:
                return await asyncio.shield(release)
            except asyncio.CancelledError:
                self.cancelled_calls.append(index)
                if not self.swallow_cancel:
                    raise

    def release(self, index, answer=None):
        future = self.releases[index]
        if not future.done():
            future.set_result(answer or self.answer)

    def release_all(self):
        for index in range(len(self.releases)):
            self.release(index)

    async def close(self):
        self.release_all()


@pytest.fixture
async def window_controller(tmp_path):
    controllers = []

    def make(*, blocked=False, swallow_cancel=False):
        clock = ManualClock()
        llm = WindowLLM(clock, blocked=blocked, swallow_cancel=swallow_cancel)
        config = Settings(
            _env_file=None,
            napcat_access_token="test-token",
            allowed_group_ids=[10],
            llm_api_key="test-key",
            llm_base_url="http://127.0.0.1:9999/v1",
            llm_model="test-model",
            max_context_messages=60,
            user_rate_limit_seconds=0.001,
        )
        store = ControlSettingsStore(
            tmp_path / f"window-{len(controllers)}.json",
            ControlSettings(mode="chat", model="test-model", target_group_id="10"),
        )
        controller = BotController(
            config,
            store,
            client=FakeClient(),
            llm_factory=lambda _: llm,
            chat_clock=clock,
            chat_sleep=clock.sleep,
        )
        controllers.append((controller, llm))
        return controller, llm, clock

    yield make
    for controller, llm in controllers:
        llm.release_all()
        await settle()
        await controller.close()


async def receive(controller, event):
    await controller.handle_payload(event)
    await settle()


async def test_one_message_waits_exactly_five_seconds(window_controller):
    controller, llm, clock = window_controller()
    await receive(controller, payload())
    await clock.advance(4.99)
    assert llm.calls == []
    await clock.advance(0.01)
    assert llm.call_times == [105.0]
    assert participation(llm.calls[0])["new_message_count"] == 1


async def test_new_message_resets_quiet_deadline(window_controller):
    controller, llm, clock = window_controller()
    await receive(controller, payload(message_id=1))
    await clock.advance(4)
    await receive(controller, payload("还有一句", message_id=2))
    await clock.advance(1)
    assert llm.calls == []
    await clock.advance(3.99)
    assert llm.calls == []
    await clock.advance(0.01)
    assert llm.call_times == [109.0]
    assert participation(llm.calls[0])["new_message_count"] == 2


async def test_continuous_messages_force_decision_at_thirty_then_start_fresh_window(
    window_controller,
):
    controller, llm, clock = window_controller()
    await receive(controller, payload(message_id=1))
    for message_id in range(2, 9):
        await clock.advance(4)
        await receive(controller, payload(f"连续聊天 {message_id}", message_id=message_id))
        assert llm.calls == []
    await clock.advance(1.99)
    assert llm.calls == []
    await clock.advance(0.01)
    assert llm.call_times == [130.0]
    assert participation(llm.calls[0])["new_message_count"] == 8

    await receive(controller, payload("新一轮消息", message_id=9))
    await clock.advance(4.99)
    assert len(llm.calls) == 1
    await clock.advance(0.01)
    assert llm.call_times == [130.0, 135.0]
    assert participation(llm.calls[1])["new_message_count"] == 1


async def test_silent_decision_does_not_reschedule_without_new_messages(window_controller):
    controller, llm, clock = window_controller()
    await receive(controller, payload())
    await clock.advance(5)
    await clock.advance(300)
    assert len(llm.calls) == 1
    assert controller.client.messages == []
    assert not clock.waiters


async def test_message_during_model_request_has_its_own_five_second_window(window_controller):
    controller, llm, clock = window_controller(blocked=True)
    await receive(controller, payload(message_id=1))
    await clock.advance(5)
    assert llm.call_times == [105.0]
    await clock.advance(1)
    await receive(controller, payload("模型思考期间的新消息", message_id=2))
    await clock.advance(1)
    llm.release(0)
    await settle()
    await clock.advance(3.99)
    assert len(llm.calls) == 1
    await clock.advance(0.01)
    assert llm.call_times == [105.0, 111.0]
    assert participation(llm.calls[1])["new_message_count"] == 1


async def test_slow_old_request_cannot_delay_next_deadline_or_send_stale_reply(window_controller):
    controller, llm, clock = window_controller(blocked=True, swallow_cancel=True)
    await receive(controller, payload("旧话题", message_id=1))
    await clock.advance(5)
    await clock.advance(1)
    await receive(controller, payload("新的连续话题", message_id=2))
    for message_id in range(3, 10):
        await clock.advance(4)
        await receive(controller, payload(f"继续新话题 {message_id}", message_id=message_id))
    assert llm.call_times == [105.0]
    await clock.advance(2)
    assert llm.call_times == [105.0, 136.0]
    assert 0 in llm.cancelled_calls
    assert participation(llm.calls[1])["new_message_count"] == 8

    llm.release(1, '{"should_reply":true,"reply":"新梗接住了"}')
    await settle()
    llm.release(0, '{"should_reply":true,"reply":"这是过期的回复"}')
    await settle()
    assert controller.client.messages == [(10, "新梗接住了")]
    await clock.advance(60)
    assert len(llm.calls) == 2


async def test_next_quiet_deadline_supersedes_unfinished_old_request(window_controller):
    controller, llm, clock = window_controller(blocked=True, swallow_cancel=True)
    await receive(controller, payload(message_id=1))
    await clock.advance(5)
    await clock.advance(1)
    await receive(controller, payload("换了话题", message_id=2))
    await clock.advance(5)
    assert llm.call_times == [105.0, 111.0]
    llm.release(0, '{"should_reply":true,"reply":"旧回答"}')
    await settle()
    assert controller.client.messages == []
    llm.release(1, '{"should_reply":true,"reply":"现在的回答"}')
    await settle()
    assert controller.client.messages == [(10, "现在的回答")]


async def test_window_is_snapshotted_before_slow_memory_lookup(window_controller):
    controller, llm, clock = window_controller()
    memory_started = asyncio.Event()
    memory_release = asyncio.Event()
    lookups = 0

    async def slow_context(user_ids, text, generation):
        nonlocal lookups
        lookups += 1
        if lookups == 1:
            memory_started.set()
            await memory_release.wait()
        return ""

    controller._profile_context = slow_context
    await receive(controller, payload("第一轮独有文字", message_id=1))
    await clock.advance(5)
    assert memory_started.is_set()
    await clock.advance(1)
    await receive(controller, payload("第二轮独有文字", message_id=2))
    await clock.advance(1)
    memory_release.set()
    await settle()
    assert llm.call_times == [107.0]
    assert participation(llm.calls[0])["new_message_count"] == 1
    assert "第一轮独有文字" in str(llm.calls[0])
    assert "第二轮独有文字" not in str(llm.calls[0])
    await clock.advance(4)
    assert llm.call_times == [107.0, 111.0]
    assert participation(llm.calls[1])["new_message_count"] == 1


@pytest.mark.parametrize("kind", ["image", "face", "mface"])
async def test_non_text_chat_messages_extend_window_and_reach_model_as_placeholders(
    window_controller, kind
):
    controller, llm, clock = window_controller()
    await receive(controller, payload(message_id=1))
    await clock.advance(4)
    event = payload(message_id=2)
    event["message"] = [{"type": kind, "data": {"id": "13", "file": "test-image"}}]
    await receive(controller, event)
    await clock.advance(1)
    assert llm.calls == []
    await clock.advance(4)
    assert llm.call_times == [109.0]
    assert participation(llm.calls[0])["new_message_count"] == 2
    user_messages = [message["content"] for message in llm.calls[0] if message["role"] == "user"]
    assert len(user_messages) == 2
    assert user_messages[-1].split("]", 1)[-1].strip()


@pytest.mark.parametrize("ignored", ["other_group", "own_message", "duplicate", "empty"])
async def test_ignored_message_does_not_extend_window(window_controller, ignored):
    controller, llm, clock = window_controller()
    first = payload(message_id=1)
    await receive(controller, first)
    await clock.advance(4)
    event = {
        "other_group": payload(group=20, message_id=2),
        "own_message": payload(user=100, message_id=2),
        "duplicate": deepcopy(first),
        "empty": payload("   ", message_id=2),
    }[ignored]
    await receive(controller, event)
    await clock.advance(1)
    assert llm.call_times == [105.0]
    assert participation(llm.calls[0])["new_message_count"] == 1


@pytest.mark.parametrize("mutation", ["pause", "switch", "clear"])
async def test_pause_switch_and_clear_cancel_window_then_new_messages_start_fresh(
    window_controller, mutation
):
    controller, llm, clock = window_controller()
    await receive(controller, payload("失效的旧消息", message_id=1))
    await clock.advance(4)
    if mutation == "pause":
        await controller.update(controller.settings.model_copy(update={"enabled": False}))
    elif mutation == "switch":
        await controller.update(controller.settings.model_copy(update={"target_group_id": "20"}))
    else:
        await controller.clear_context()
    await clock.advance(30)
    assert llm.calls == []
    assert not clock.waiters
    if mutation == "pause":
        await controller.update(controller.settings.model_copy(update={"enabled": True}))
    group = 20 if mutation == "switch" else 10
    await receive(controller, payload("新的消息", group=group, message_id=2))
    await clock.advance(4.99)
    assert llm.calls == []
    await clock.advance(0.01)
    assert llm.call_times == [139.0]
    assert participation(llm.calls[0])["new_message_count"] == 1


async def test_close_cancels_waiting_window(window_controller):
    controller, llm, clock = window_controller()
    await receive(controller, payload())
    await clock.advance(4)
    await controller.close()
    await clock.advance(40)
    assert llm.calls == []
    assert not clock.waiters
