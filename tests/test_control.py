"""Controller behavior without a model provider, a browser, or a QQ connection."""

import asyncio
import json
import logging
from copy import deepcopy
from unittest.mock import AsyncMock

import pytest

from app.config import Settings
from app.services.control import BotController
from app.services.conversation import RollingReplyBudget
from app.services.llm import LLMError
from app.web.settings import ControlSettings, ControlSettingsStore


class FakeLLM:
    def __init__(self, answer="模型回答", *, blocked=False, swallow_cancel=False):
        self.answer = answer
        self.calls = []
        self.call_times = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        if not blocked:
            self.release.set()
        self.swallow_cancel = swallow_cancel
        self.cancelled = False
        self.closed = False

    async def chat(self, messages):
        self.calls.append(deepcopy(messages))
        self.call_times.append(asyncio.get_running_loop().time())
        self.started.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            if not self.swallow_cancel:
                raise
        return self.answer

    async def close(self):
        self.closed = True

    async def chat_json(self, messages):
        return await self.chat(messages)

    async def list_models(self):
        return ["model-b", "model-a", "model-b"]


class FakeClient:
    is_connected = True

    def __init__(self, *, send_error=None):
        self.messages = []
        self.attempts = []
        self.send_error = send_error
        self.closed = False

    async def send_group_message(self, group_id, message):
        self.attempts.append((group_id, deepcopy(message)))
        if self.send_error:
            raise self.send_error
        self.messages.append((group_id, deepcopy(message)))
        return len(self.messages)

    async def close(self):
        self.closed = True


def payload(text="今天终于周五了", *, group=10, user=200, message_id=1, at=False, nickname="小明"):
    segments = [{"type": "text", "data": {"text": text}}]
    if at:
        segments.insert(0, {"type": "at", "data": {"qq": "100"}})
    return {
        "post_type": "message",
        "message_type": "group",
        "group_id": group,
        "self_id": 100,
        "user_id": user,
        "message_id": message_id,
        "message": segments,
        "sender": {"user_id": user, "nickname": nickname},
    }


def reply_text(client, index=0):
    message = client.messages[index][1]
    if isinstance(message, str):
        return message
    return "".join(segment["data"]["text"] for segment in message if segment["type"] == "text")


def participation(messages):
    prompt = messages[0]["content"]
    return json.loads(prompt.split("本轮参与状态（由程序提供）：", 1)[1].splitlines()[0])


async def drain(controller):
    async with asyncio.timeout(2):
        while controller._jobs:
            await asyncio.gather(*controller._jobs, return_exceptions=True)
            await asyncio.sleep(0)


@pytest.fixture
async def make_controller(tmp_path):
    controllers = []

    def make(
        *,
        mode="assistant",
        llm=None,
        client=None,
        factory=None,
        cap=10,
        max_reply_chars=1500,
        max_context_messages=20,
        debounce=0,
    ):
        selected_llm = llm or FakeLLM()
        config = Settings(
            _env_file=None,
            napcat_access_token="fake-napcat-token",
            allowed_group_ids=[10],
            llm_api_key="fake-model-key",
            llm_base_url="http://127.0.0.1:9999/v1",
            llm_model="model-a",
            user_rate_limit_seconds=0.001,
            max_reply_chars=max_reply_chars,
            max_context_messages=max_context_messages,
        )
        store = ControlSettingsStore(
            tmp_path / f"settings-{len(controllers)}.json",
            ControlSettings(
                mode=mode,
                model="model-a",
                target_group_id="10",
                reply_limit_per_minute=cap,
            ),
        )
        controller = BotController(
            config,
            store,
            client=client or FakeClient(),
            llm_factory=factory or (lambda model: selected_llm),
            debounce_seconds=debounce,
        )
        controllers.append(controller)
        return controller

    yield make
    for controller in controllers:
        await controller.close()


async def test_chat_can_reply_without_at_but_assistant_requires_at(make_controller):
    llm = FakeLLM(json.dumps({"should_reply": True, "reply": "我的周末已经提前下班了。"}))
    chat = make_controller(mode="chat", llm=llm)
    await chat.handle_payload(payload())
    await drain(chat)
    assert len(llm.calls) == 1
    assert reply_text(chat.client) == "我的周末已经提前下班了。"
    assert isinstance(chat.client.messages[0][1], str)

    assistant_llm = FakeLLM()
    assistant = make_controller(llm=assistant_llm)
    await assistant.handle_payload(payload())
    await drain(assistant)
    assert assistant_llm.calls == []
    assert assistant.client.messages == []
    await assistant.handle_payload(payload(message_id=2, at=True))
    await drain(assistant)
    assert len(assistant_llm.calls) == 1
    assert reply_text(assistant.client) == "模型回答"
    assert assistant.client.messages[0][1][0] == {"type": "reply", "data": {"id": "2"}}


@pytest.mark.parametrize(
    "answer, activity_kind",
    [('{"should_reply": false, "reply": ""}', "silent"), ("模型坏掉了", "error")],
)
async def test_chat_can_stay_silent_and_does_not_retry_without_new_messages(
    make_controller, answer, activity_kind
):
    llm = FakeLLM(answer)
    controller = make_controller(mode="chat", llm=llm)
    await controller.handle_payload(payload())
    await drain(controller)
    await asyncio.sleep(0.02)
    assert len(llm.calls) == 1
    assert controller.client.messages == []
    assert controller.status()["context_messages"] == 1
    outcomes = [
        item["kind"] for item in controller.status()["activity"] if item["kind"] != "received"
    ]
    assert outcomes == [activity_kind]
    assert bool(controller.status()["last_error"]) == (activity_kind == "error")


async def test_only_valid_silence_updates_observation_count_and_sent_reply_resets_it(
    make_controller,
):
    llm = FakeLLM()
    controller = make_controller(mode="chat", llm=llm)
    answers = [
        '{"should_reply": false, "reply": ""}',
        "格式不正确的模型回答",
        '{"should_reply": false, "reply": ""}',
        '{"should_reply": true, "reply": "这个梗我接住了。"}',
        '{"should_reply": false, "reply": ""}',
    ]
    for index, answer in enumerate(answers):
        llm.answer = answer
        await controller.handle_payload(payload(f"新话题 {index}", message_id=index))
        await drain(controller)
    states = [participation(messages) for messages in llm.calls]
    assert [state["consecutive_silent_turns"] for state in states] == [0, 1, 1, 2, 0]
    assert [state["new_message_count"] for state in states] == [1] * 5
    assert "连续旁听多轮" not in llm.calls[3][0]["content"]
    assert len(controller.client.messages) == 1
    assert reply_text(controller.client) == "这个梗我接住了。"
    assert controller.status()["last_error"] is None


async def test_other_groups_rejected_before_dispatch_validation_logging_or_memory(
    make_controller, caplog
):
    controller = make_controller(mode="chat")
    controller.dispatcher.handle_event = AsyncMock()
    with caplog.at_level(logging.DEBUG):
        for value in (20, "20", True, 10.0, None, "010"):
            await controller.handle_payload(
                {"group_id": value, "message": "private-other-group-message", "secret": "private"}
            )
    controller.dispatcher.handle_event.assert_not_called()
    assert controller.status()["context_messages"] == 0
    assert controller.status()["activity"] == []
    assert "private" not in caplog.text
    assert controller.client.messages == []


async def test_private_own_and_non_message_events_do_not_reach_chat(make_controller):
    controller = make_controller(mode="chat")
    await controller.handle_payload(payload(user=100))
    private = payload()
    private["message_type"] = "private"
    await controller.handle_payload(private)
    notification = payload()
    notification["post_type"] = "notice"
    await controller.handle_payload(notification)
    await drain(controller)
    assert controller.status()["context_messages"] == 0
    assert controller.status()["activity"] == []
    assert controller._llm.calls == []


async def test_duplicate_target_message_is_not_recorded_or_processed_twice(make_controller):
    controller = make_controller(mode="chat")
    event = payload()
    await controller.handle_payload(event)
    await controller.handle_payload(deepcopy(event))
    await drain(controller)
    assert len(controller._llm.calls) == 1
    assert controller.status()["context_messages"] == 1
    assert sum(item["kind"] == "received" for item in controller.status()["activity"]) == 1


async def test_only_eligible_text_and_media_update_new_message_or_mention_state(make_controller):
    controller = make_controller(mode="chat")
    await controller.handle_payload(payload())
    await controller.handle_payload(payload(at=True))  # Same message ID: a duplicate.
    await controller.handle_payload(payload(at=True, user=100, message_id=2))
    await controller.handle_payload(payload(at=True, group=20, message_id=3))
    image_only = payload(at=True, message_id=4)
    image_only["message"] = [
        {"type": "at", "data": {"qq": "100"}},
        {"type": "image", "data": {"file": "ignored-image"}},
    ]
    await controller.handle_payload(image_only)
    await drain(controller)
    assert len(controller._llm.calls) == 1
    assert participation(controller._llm.calls[0]) == {
        "new_message_count": 2,
        "consecutive_silent_turns": 0,
        "addressed_to_bot": True,
    }
    assert "未识别画面" in controller._llm.calls[0][-1]["content"]


async def test_chat_bursts_coalesce_into_one_decision_and_preserve_speakers(make_controller):
    controller = make_controller(mode="chat", debounce=0.01)
    for index in range(8):
        await controller.handle_payload(
            payload(f"话题 {index}", message_id=index, user=200 + index, nickname=f"群友 {index}")
        )
    await drain(controller)
    assert len(controller._llm.calls) == 1
    history = controller._llm.calls[0][1:]
    assert len(history) == 8
    assert history[0]["content"] == "[群友 0/200] 话题 0"
    assert history[-1]["content"] == "[群友 7/207] 话题 7"
    assert participation(controller._llm.calls[0])["new_message_count"] == 8


async def test_chat_burst_count_is_limited_to_retained_context(make_controller):
    controller = make_controller(mode="chat", max_context_messages=4)
    for index in range(30):
        await controller.handle_payload(payload(f"话题 {index}", message_id=index))
    await drain(controller)
    assert len(controller._llm.calls) == 1
    messages = controller._llm.calls[0]
    assert participation(messages)["new_message_count"] == 4
    assert len(messages[1:]) == 4
    assert messages[1]["content"].endswith("话题 26")


async def test_new_batch_waits_for_its_own_quiet_window_even_when_previous_was_silent(
    make_controller,
):
    llm = FakeLLM('{"should_reply": false, "reply": ""}')
    controller = make_controller(mode="chat", llm=llm, debounce=0.05)
    await controller.handle_payload(payload())
    await drain(controller)
    await controller.handle_payload(payload(message_id=2))
    await drain(controller)
    assert len(llm.calls) == 2
    assert llm.call_times[1] - llm.call_times[0] >= 0.045
    assert controller.client.messages == []


async def test_messages_arriving_during_model_call_get_one_followup_decision(make_controller):
    llm = FakeLLM('{"should_reply": false, "reply": ""}', blocked=True)
    controller = make_controller(mode="chat", llm=llm)
    await controller.handle_payload(payload("首条"))
    await asyncio.wait_for(llm.started.wait(), 1)
    for index in range(2, 7):
        await controller.handle_payload(payload(f"后来 {index}", message_id=index, at=index == 4))
    llm.release.set()
    await drain(controller)
    assert len(llm.calls) == 2
    assert len(llm.calls[0]) == 2
    assert len(llm.calls[1]) == 7
    assert participation(llm.calls[0]) == {
        "new_message_count": 1,
        "consecutive_silent_turns": 0,
        "addressed_to_bot": False,
    }
    assert participation(llm.calls[1]) == {
        "new_message_count": 5,
        "consecutive_silent_turns": 1,
        "addressed_to_bot": True,
    }
    await controller.handle_payload(payload("再来一条", message_id=7))
    await drain(controller)
    assert len(llm.calls) == 3
    assert participation(llm.calls[2]) == {
        "new_message_count": 1,
        "consecutive_silent_turns": 2,
        "addressed_to_bot": False,
    }


@pytest.mark.parametrize("new_messages", [0, 3])
async def test_model_failure_processes_pending_new_messages_without_retrying_old_ones(
    make_controller, new_messages
):
    class FailsFirstLLM(FakeLLM):
        async def chat(self, messages):
            first_call = not self.calls
            answer = await super().chat(messages)
            if first_call:
                raise LLMError("timeout")
            return answer

    llm = FailsFirstLLM('{"should_reply": true, "reply": "新话题来了，我继续接。"}', blocked=True)
    controller = make_controller(mode="chat", llm=llm)
    await controller.handle_payload(payload("旧话题"))
    await asyncio.wait_for(llm.started.wait(), 1)
    for index in range(new_messages):
        await controller.handle_payload(payload(f"新话题 {index}", message_id=index + 2))
    llm.release.set()
    await drain(controller)
    await asyncio.sleep(0.02)
    assert len(llm.calls) == (2 if new_messages else 1)
    assert len(controller.client.messages) == (1 if new_messages else 0)
    if new_messages:
        assert participation(llm.calls[1])["new_message_count"] == new_messages
        assert participation(llm.calls[1])["consecutive_silent_turns"] == 0
        assert controller.status()["last_error"] is None
    else:
        assert controller.status()["last_error"]


@pytest.mark.parametrize("action", ["clear", "pause", "target", "mode", "style", "model"])
async def test_chat_setting_changes_clear_old_silence_and_pending_participation(
    make_controller, action
):
    llm = FakeLLM('{"should_reply": false, "reply": ""}')
    controller = make_controller(mode="chat", llm=llm)
    await controller.handle_payload(payload("第一轮旁听"))
    await drain(controller)
    llm.started.clear()
    llm.release.clear()
    await controller.handle_payload(payload("第二轮进行中", message_id=2))
    await asyncio.wait_for(llm.started.wait(), 1)
    await controller.handle_payload(payload("等着回应的新消息", at=True, message_id=3))
    if action == "clear":
        await controller.clear_context()
    else:
        changes = {
            "pause": {"enabled": False},
            "target": {"target_group_id": "20"},
            "mode": {"mode": "assistant"},
            "style": {"chat_style": "简短接梗"},
            "model": {"model": "model-b"},
        }
        await controller.update(controller.settings.model_copy(update=changes[action]))
    await drain(controller)
    # The zero-delay fixture can start the next independent batch while settings
    # are being persisted; every outstanding decision must still be cancelled.
    assert llm.cancelled
    assert controller.client.messages == []
    if action in {"pause", "mode"}:
        await controller.update(
            controller.settings.model_copy(update={"enabled": True, "mode": "chat"})
        )
    llm.release.set()
    await controller.handle_payload(
        payload("新设置下的新话题", group=int(controller.settings.target_group_id), message_id=4)
    )
    await drain(controller)
    assert participation(llm.calls[-1]) == {
        "new_message_count": 1,
        "consecutive_silent_turns": 0,
        "addressed_to_bot": False,
    }


async def test_unsent_chat_reply_does_not_reset_silence_count(make_controller):
    llm = FakeLLM('{"should_reply": false, "reply": ""}')
    controller = make_controller(mode="chat", llm=llm)
    await controller.handle_payload(payload())
    await drain(controller)
    controller.client.is_connected = False
    llm.answer = '{"should_reply": true, "reply": "这条因断线无法发出。"}'
    await controller.handle_payload(payload(message_id=2))
    await drain(controller)
    assert controller.client.attempts == []
    llm.answer = '{"should_reply": false, "reply": ""}'
    await controller.handle_payload(payload(message_id=3))
    await drain(controller)
    assert participation(llm.calls[-1])["consecutive_silent_turns"] == 1


async def test_budget_skipped_messages_do_not_leak_mention_or_count_into_next_decision(
    make_controller,
):
    class Clock:
        now = 0

        def __call__(self):
            return self.now

    clock = Clock()
    llm = FakeLLM('{"should_reply": false, "reply": ""}')
    controller = make_controller(mode="chat", llm=llm, cap=1)
    controller.budget = RollingReplyBudget(1, clock=clock)
    await controller.handle_payload(payload())
    await drain(controller)
    assert controller.budget.allow()
    await controller.handle_payload(payload("限额期间的消息", at=True, message_id=2))
    await drain(controller)
    assert len(llm.calls) == 1
    clock.now = 60
    await controller.handle_payload(payload("限额恢复后的消息", message_id=3))
    await drain(controller)
    assert participation(llm.calls[-1]) == {
        "new_message_count": 1,
        "consecutive_silent_turns": 1,
        "addressed_to_bot": False,
    }


async def test_commands_and_chunks_share_budget_and_exhaustion_prevents_model_call(make_controller):
    controller = make_controller(cap=2, max_reply_chars=20)
    await controller.handle_payload(payload("/help"))
    await drain(controller)
    assert len(controller.client.messages) == 2
    assert controller.status()["replies_last_minute"] == 2
    await controller.handle_payload(payload("回答一下", at=True, user=201, message_id=2))
    await drain(controller)
    assert controller._llm.calls == []
    assert len(controller.client.messages) == 2


async def test_failed_send_consumes_budget_and_does_not_store_unsent_reply(make_controller):
    controller = make_controller(cap=1, client=FakeClient(send_error=RuntimeError("offline")))
    await controller.handle_payload(payload(at=True))
    await drain(controller)
    assert len(controller.client.attempts) == 1
    assert controller.client.messages == []
    assert controller.status()["replies_last_minute"] == 1
    assert controller.status()["context_messages"] == 0
    assert controller.status()["last_error"]


async def test_budget_limit_changes_keep_existing_reservations_and_recover_after_sixty_seconds(
    make_controller,
):
    class Clock:
        now = 0

        def __call__(self):
            return self.now

    clock = Clock()
    controller = make_controller(cap=2)
    controller.budget = RollingReplyBudget(2, clock=clock)
    await controller.handle_payload(payload("/status"))
    await drain(controller)
    await controller.update(controller.settings.model_copy(update={"reply_limit_per_minute": 1}))
    await controller.handle_payload(payload(at=True, user=201, message_id=2))
    await drain(controller)
    assert controller._llm.calls == []
    clock.now = 60
    await controller.handle_payload(payload(at=True, user=202, message_id=3))
    await drain(controller)
    assert len(controller._llm.calls) == 1
    assert len(controller.client.messages) == 2


@pytest.mark.parametrize("action", ["pause", "clear", "target", "mode", "style"])
@pytest.mark.parametrize("mode", ["assistant", "chat"])
async def test_mutations_cancel_model_work_even_if_provider_swallows_cancellation(
    make_controller, action, mode
):
    answer = '{"should_reply": true, "reply": "旧任务的回答"}' if mode == "chat" else "旧任务的回答"
    llm = FakeLLM(answer, blocked=True, swallow_cancel=True)
    controller = make_controller(mode=mode, llm=llm)
    await controller.handle_payload(payload(at=mode == "assistant"))
    await asyncio.wait_for(llm.started.wait(), 1)
    if action == "clear":
        await controller.clear_context()
    else:
        updates = {
            "pause": {"enabled": False},
            "target": {"target_group_id": "20"},
            "mode": {"mode": "assistant" if mode == "chat" else "chat"},
            "style": {"chat_style": "安静简短"},
        }[action]
        await controller.update(controller.settings.model_copy(update=updates))
    await drain(controller)
    assert llm.cancelled
    assert controller.client.messages == []
    if action != "pause":
        assert controller.status()["context_messages"] == 0


async def test_clear_cancels_unscheduled_coroutines_without_leaking_them(make_controller):
    controller = make_controller()
    await controller.handle_payload(payload(at=True))
    await controller.clear_context()
    await drain(controller)
    assert controller._llm.calls == []
    assert controller.client.messages == []
    assert controller.status()["context_messages"] == 0


async def test_clear_command_cancels_inflight_answer_and_replies_without_deadlocking(
    make_controller,
):
    llm = FakeLLM("旧回答", blocked=True, swallow_cancel=True)
    controller = make_controller(llm=llm)
    await controller.handle_payload(payload(at=True))
    await asyncio.wait_for(llm.started.wait(), 1)
    await controller.handle_payload(payload("/clear", user=201, message_id=2))
    await drain(controller)
    assert llm.cancelled
    assert len(controller.client.messages) == 1
    assert "已清空" in reply_text(controller.client)
    assert controller.status()["context_messages"] == 0


async def test_target_change_erases_old_memory_and_activity_and_filters_previous_group(
    make_controller,
):
    controller = make_controller(mode="chat")
    await controller.handle_payload(payload("旧群的私密话题", nickname="旧群成员"))
    await drain(controller)
    assert controller.status()["context_messages"] == 1
    await controller.update(controller.settings.model_copy(update={"target_group_id": "20"}))
    state = controller.status()
    assert state["context_messages"] == 0
    assert "旧群" not in json.dumps(state, ensure_ascii=False)
    await controller.handle_payload(payload("旧群又发言了", message_id=2))
    await controller.handle_payload(payload("新群的话题", group=20, message_id=1))
    await drain(controller)
    assert len(controller._llm.calls) == 2
    assert len(controller._llm.calls[1]) == 2
    assert controller._llm.calls[1][-1]["content"].endswith("新群的话题")
    assert "旧群" not in json.dumps(controller.status(), ensure_ascii=False)


async def test_model_change_uses_selected_factory_model_closes_old_client_and_cancels_old_work(
    make_controller,
):
    original = FakeLLM("旧模型回答", blocked=True, swallow_cancel=True)
    replacement = FakeLLM("新模型回答")
    chosen = []

    def factory(model):
        chosen.append(model)
        return {"model-a": original, "model-b": replacement}[model]

    controller = make_controller(factory=factory)
    await controller.handle_payload(payload(at=True))
    await asyncio.wait_for(original.started.wait(), 1)
    await controller.update(controller.settings.model_copy(update={"model": "model-b"}))
    assert chosen == ["model-a", "model-b"]
    assert original.cancelled and original.closed
    assert not replacement.closed
    assert controller.client.messages == []
    await controller.handle_payload(payload(at=True, user=201, message_id=2))
    await drain(controller)
    assert len(replacement.calls) == 1
    assert reply_text(controller.client) == "新模型回答"


async def test_paused_mode_ignores_messages_and_resuming_preserves_configuration(make_controller):
    controller = make_controller()
    await controller.update(controller.settings.model_copy(update={"enabled": False}))
    before = controller.status()
    await controller.handle_payload(payload(at=True))
    await drain(controller)
    assert controller.status() == before
    await controller.update(controller.settings.model_copy(update={"enabled": True}))
    await controller.handle_payload(payload(at=True, message_id=2))
    await drain(controller)
    assert len(controller.client.messages) == 1
    assert controller.store.load().enabled


async def test_activity_hides_configured_credentials_even_if_they_appear_in_group_text(
    make_controller,
):
    controller = make_controller()
    await controller.handle_payload(payload("fake-model-key fake-napcat-token"))
    public = json.dumps(controller.status(), ensure_ascii=False)
    assert "fake-model-key" not in public
    assert "fake-napcat-token" not in public
    assert "[已隐藏]" in public


async def test_model_list_deduplicates_and_always_includes_selected_model(make_controller):
    controller = make_controller()
    assert await controller.list_models() == {"models": ["model-a", "model-b"], "error": None}
