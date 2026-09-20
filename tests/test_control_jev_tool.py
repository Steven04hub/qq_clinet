"""Public user decisions and the private participation gate remain independent."""

import asyncio
import json
from copy import deepcopy

import pytest

from app.config import Settings
from app.services.control import BotController
from app.services.jev import QUESTION_IDS, JevError, JevQuestionResult
from app.services.jev_tool import JevToolPlan, JevToolPlanError
from app.services.llm import LLMError
from app.services.profile_store import GroupMemoryStore
from app.web.settings import ControlSettings, ControlSettingsStore
from tests.test_chat_window import ManualClock, settle
from tests.test_control import FakeClient, FakeLLM, drain, payload, reply_text
from tests.test_control_jev import FakeJev as InternalFakeJev

REQUEST = "用 Jev 帮我决策一下，今晚要不要再来一局？"
DECISION = JevToolPlan(action="decide", question="今晚要不要再来一局？", context="已经玩了两局")
CLARIFY = JevToolPlan(action="clarify", clarification="你想让 Jev 决策什么？")


class FakePlanner:
    def __init__(self, plans=None, *, error=None, blocked=False, swallow_cancel=False):
        self.plans = list(plans or [DECISION])
        self.error = error
        self.calls = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        if not blocked:
            self.release.set()
        self.swallow_cancel = swallow_cancel
        self.cancelled = False

    async def __call__(self, llm, **kwargs):
        self.calls.append(deepcopy(kwargs))
        self.started.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            if not self.swallow_cancel:
                raise
        if self.error:
            raise self.error
        return self.plans[min(len(self.calls) - 1, len(self.plans) - 1)]


class FakeJev(InternalFakeJev):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.question_calls = []

    async def evaluate_question(self, **kwargs):
        self.question_calls.append(deepcopy(kwargs))
        self.started.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            if not self.swallow_cancel:
                raise
        if self.error:
            raise self.error
        options = kwargs["options"]
        return JevQuestionResult(
            question=kwargs["question"],
            kind="choice" if options else "noul",
            answer=options[0] if options else "否",
            probabilities=(
                {label: float(index == 0) for index, label in enumerate(options)}
                if options
                else {"是": 0.25, "否": 0.75}
            ),
            confidence=0.9 if options else None,
            elapsed_ms=12,
            input_tokens=100,
            model="jev-1.13.0",
        )


class RunningFakeClient(FakeClient):
    async def run(self, on_payload):
        await asyncio.Future()


@pytest.fixture
async def tool_controller(tmp_path):
    created = []

    def make(
        *,
        bot_mode="chat",
        jev_mode="shadow",
        jev=None,
        llm=None,
        planner=None,
        limit=10,
        debounce=0,
        configured=True,
        memory=None,
        cooldown=0.001,
    ):
        clock = ManualClock()
        selected_llm = llm or FakeLLM('{"should_reply":false,"reply":""}')
        gate = (jev or FakeJev()) if configured else None
        planner = planner or FakePlanner()
        config = Settings(
            _env_file=None,
            napcat_access_token="napcat-test-secret",
            llm_api_key="llm-test-secret",
            typesafe_api_key="typesafe-test-secret" if configured else "",
            jev_mode=jev_mode,
            profile_auto_import=False,
            user_rate_limit_seconds=cooldown,
        )
        store = ControlSettingsStore(
            tmp_path / f"control-tool-{len(created)}.json",
            ControlSettings(
                target_group_id="10",
                model="test-model",
                mode=bot_mode,
                reply_limit_per_minute=limit,
            ),
        )
        controller = BotController(
            config,
            store,
            client=RunningFakeClient(),
            llm_factory=lambda _: selected_llm,
            jev_client=gate,
            jev_planner=planner,
            debounce_seconds=debounce,
            memory=memory,
            chat_clock=clock,
            chat_sleep=clock.sleep,
        )
        controller._user_rate._clock = clock
        created.append((controller, gate, planner))
        return controller, selected_llm, gate, planner, clock

    yield make
    for controller, gate, planner in created:
        planner.release.set()
        if gate:
            gate.release.set()
        await controller.close()
        if gate:
            assert gate.closed


@pytest.mark.parametrize("bot_mode", ["chat", "assistant"])
@pytest.mark.parametrize("jev_mode", ["off", "shadow", "gate"])
async def test_user_question_runs_independently_of_private_gate_mode(
    tool_controller, bot_mode, jev_mode
):
    controller, llm, jev, planner, _ = tool_controller(bot_mode=bot_mode, jev_mode=jev_mode)
    await controller.handle_payload(payload(REQUEST))
    await drain(controller)
    assert len(planner.calls) == len(jev.question_calls) == len(controller.client.messages) == 1
    assert jev.calls == llm.calls == []
    assert jev.question_calls[0] == {
        "question": DECISION.question,
        "context": DECISION.context,
        "options": [],
    }
    assert controller.budget.used == 1
    assert controller.client.messages[0][0] == 10
    text = reply_text(controller.client)
    assert "Jev 决策结果" in text and "75%" in text and "12 ms" in text
    assert "再来一局" in text
    assert not any(name in text for name in QUESTION_IDS)
    assert not any(label in text for label in ("接话机会", "期待回应", "会打断对话", "建议静默"))
    assert controller._jev_tool_tasks == {}


@pytest.mark.parametrize(
    "text, forced", [("/jev", True), (REQUEST, False), ("那个决策模型帮我选选今晚吃啥", False)]
)
async def test_candidate_only_selects_for_semantic_planning(tool_controller, text, forced):
    controller, _, jev, planner, _ = tool_controller()
    await controller.handle_payload(payload(text))
    await drain(controller)
    assert planner.calls[0]["forced"] is forced
    assert len(jev.question_calls) == 1


async def test_choice_question_options_and_context_are_passed_without_internal_rubrics(
    tool_controller,
):
    plan = JevToolPlan(
        action="decide",
        question="今晚吃哪个？",
        context="四个人，预算 200",
        options=["火锅", "烧烤"],
    )
    controller, _, jev, _, _ = tool_controller(planner=FakePlanner([plan]))
    await controller.handle_payload(payload("让 Jev 选一下：火锅还是烧烤？我们四个人，预算200"))
    await drain(controller)
    assert jev.calls == []
    assert jev.question_calls == [
        {"question": plan.question, "context": plan.context, "options": plan.options}
    ]
    assert "火锅" in reply_text(controller.client) and "烧烤" in reply_text(controller.client)


@pytest.mark.parametrize("bot_mode", ["chat", "assistant"])
async def test_missing_question_asks_for_own_problem_instead_of_testing_group_chat(
    tool_controller, bot_mode
):
    controller, llm, jev, _, _ = tool_controller(bot_mode=bot_mode, planner=FakePlanner([CLARIFY]))
    await controller.handle_payload(payload("用 Jev 帮我决策一下", at=True))
    await drain(controller)
    assert jev.calls == jev.question_calls == llm.calls == []
    assert reply_text(controller.client) == CLARIFY.clarification
    assert "200" in controller._jev_tool_pending


@pytest.mark.parametrize("bot_mode", ["chat", "assistant"])
async def test_semantic_none_keeps_regular_chat_and_never_calls_public_tool(
    tool_controller, bot_mode
):
    controller, llm, jev, planner, _ = tool_controller(
        bot_mode=bot_mode,
        jev_mode="off",
        planner=FakePlanner([JevToolPlan(action="none")]),
    )
    if bot_mode == "assistant":
        llm.answer = "Jev 是决策模型"
    await controller.handle_payload(payload("Jev 是什么原理？", at=True))
    await drain(controller)
    assert len(planner.calls) == len(llm.calls) == 1
    assert jev.question_calls == jev.calls == []


async def test_followup_within_180_seconds_uses_only_requester_context(tool_controller):
    planner = FakePlanner([CLARIFY, DECISION])
    controller, _, jev, _, clock = tool_controller(
        bot_mode="assistant", planner=planner, cooldown=60
    )
    await controller.handle_payload(payload("用 Jev 帮我做个决定", message_id=1))
    await drain(controller)
    await clock.advance(61)
    await controller.handle_payload(payload("今晚要不要再来一局", message_id=2))
    await drain(controller)
    assert len(planner.calls) == 2
    context = planner.calls[1]["requester_context"]
    assert context == [
        {"role": "user", "content": "用 Jev 帮我做个决定"},
        {"role": "assistant", "content": CLARIFY.clarification},
    ]
    assert len(jev.question_calls) == 1
    assert len(controller.client.messages) == 2


async def test_pending_question_does_not_capture_another_members_followup(tool_controller):
    controller, _, jev, planner, clock = tool_controller(
        bot_mode="assistant", planner=FakePlanner([CLARIFY])
    )
    await controller.handle_payload(payload("用 Jev 帮我做个决定", message_id=1))
    await drain(controller)
    await clock.advance(1)
    await controller.handle_payload(payload("我想选火锅", user=201, message_id=2))
    await drain(controller)
    assert len(planner.calls) == 1
    assert jev.question_calls == []
    assert "201" not in controller._jev_tool_pending


async def test_followup_retains_details_received_while_clarification_was_planning(tool_controller):
    planner = FakePlanner([CLARIFY, DECISION], blocked=True)
    controller, _, jev, _, clock = tool_controller(bot_mode="assistant", planner=planner)
    await controller.handle_payload(payload("Jev 帮我决策下", message_id=1))
    await asyncio.wait_for(planner.started.wait(), 1)
    await controller.handle_payload(payload("火锅还是烧烤？预算 200 元", message_id=2))
    await controller.handle_payload(payload("别人的问题不参与", user=201, message_id=3))
    planner.release.set()
    await drain(controller)
    await clock.advance(1)
    await controller.handle_payload(payload("就是刚才那个问题", message_id=4))
    await drain(controller)
    context = json.dumps(planner.calls[-1]["requester_context"], ensure_ascii=False)
    assert "火锅还是烧烤" in context and "200 元" in context
    assert "别人的问题" not in context
    assert len(jev.question_calls) == 1


async def test_followup_expires_at_180_seconds(tool_controller):
    controller, _, jev, planner, clock = tool_controller(
        bot_mode="assistant", planner=FakePlanner([CLARIFY])
    )
    await controller.handle_payload(payload("用 Jev 帮我做个决定", message_id=1))
    await drain(controller)
    await clock.advance(180)
    await controller.handle_payload(payload("今晚要不要再来一局", message_id=2))
    await drain(controller)
    assert len(planner.calls) == 1
    assert jev.question_calls == []
    assert "200" not in controller._jev_tool_pending


async def test_initial_context_contains_only_same_users_history(tool_controller):
    controller, llm, jev, planner, _ = tool_controller(bot_mode="assistant")
    await controller.handle_payload(payload("我的预算是 200", message_id=1))
    await controller.handle_payload(
        payload("another-person-private-question", user=201, message_id=2)
    )
    await controller.handle_payload(payload("用 Jev 帮我决策一下", message_id=3))
    await drain(controller)
    context = planner.calls[0]["requester_context"]
    assert context == [{"role": "user", "content": "我的预算是 200"}]
    assert "another-person-private-question" not in json.dumps(planner.calls, ensure_ascii=False)
    assert llm.calls == []
    assert len(jev.question_calls) == 1


@pytest.mark.parametrize("case", ["other_group", "own_message", "paused", "private"])
async def test_target_group_and_enabled_checks_precede_tool_planning(tool_controller, case):
    controller, llm, jev, planner, _ = tool_controller()
    event = payload(REQUEST)
    if case == "other_group":
        event["group_id"] = 20
    elif case == "own_message":
        event["user_id"] = 100
    elif case == "private":
        event["message_type"] = "private"
    else:
        await controller.update(controller.settings.model_copy(update={"enabled": False}))
    await controller.handle_payload(event)
    await drain(controller)
    assert (
        planner.calls
        == jev.calls
        == jev.question_calls
        == llm.calls
        == controller.client.messages
        == []
    )


async def test_message_to_another_member_is_not_a_tool_request(tool_controller):
    controller, _, jev, planner, _ = tool_controller(bot_mode="assistant")
    event = payload(REQUEST)
    event["message"].insert(0, {"type": "at", "data": {"qq": "300"}})
    await controller.handle_payload(event)
    await drain(controller)
    assert planner.calls == jev.question_calls == []


async def test_duplicate_event_only_plans_calls_and_sends_once(tool_controller):
    controller, _, jev, planner, _ = tool_controller()
    event = payload(REQUEST)
    await controller.handle_payload(event)
    await controller.handle_payload(deepcopy(event))
    await drain(controller)
    assert len(planner.calls) == len(jev.question_calls) == len(controller.client.messages) == 1


async def test_user_cooldown_precedes_planner(tool_controller):
    controller, _, jev, planner, _ = tool_controller(cooldown=60)
    await controller.handle_payload(payload(REQUEST, message_id=1))
    await drain(controller)
    await controller.handle_payload(payload(REQUEST, message_id=2))
    await drain(controller)
    assert len(planner.calls) == len(jev.question_calls) == len(controller.client.messages) == 1


async def test_exhausted_reply_budget_precedes_planning_and_provider(tool_controller):
    controller, _, jev, planner, _ = tool_controller(limit=1)
    assert controller.budget.allow()
    await controller.handle_payload(payload(REQUEST))
    await drain(controller)
    assert planner.calls == jev.question_calls == controller.client.messages == []
    assert controller.budget.used == 1


async def test_budget_filled_during_planning_prevents_provider_call(tool_controller):
    planner = FakePlanner(blocked=True)
    controller, _, jev, _, _ = tool_controller(planner=planner, limit=1)
    await controller.handle_payload(payload(REQUEST))
    await asyncio.wait_for(planner.started.wait(), 1)
    assert controller.budget.allow()
    planner.release.set()
    await drain(controller)
    assert jev.question_calls == controller.client.messages == []


@pytest.mark.parametrize("stage", ["planner", "provider"])
@pytest.mark.parametrize("change", [{"enabled": False}, {"target_group_id": "20"}])
async def test_lifecycle_invalidates_inflight_tool_even_if_cancellation_ignored(
    tool_controller, stage, change
):
    planner = FakePlanner(blocked=stage == "planner", swallow_cancel=True)
    jev = FakeJev(blocked=stage == "provider", swallow_cancel=True)
    controller, _, _, _, _ = tool_controller(planner=planner, jev=jev)
    await controller.handle_payload(payload(REQUEST))
    blocked = planner if stage == "planner" else jev
    await asyncio.wait_for(blocked.started.wait(), 1)
    await controller.update(controller.settings.model_copy(update=change))
    await drain(controller)
    assert blocked.cancelled
    assert controller.client.messages == []
    assert controller._jev_tool_pending == controller._jev_tool_tasks == {}
    if stage == "planner":
        assert jev.question_calls == []


@pytest.mark.parametrize("kind", ["auth", "timeout", "malformed"])
async def test_provider_failure_never_fabricates_a_decision(tool_controller, kind):
    controller, llm, jev, _, _ = tool_controller(jev=FakeJev(error=JevError(kind)))
    await controller.handle_payload(payload(REQUEST))
    await drain(controller)
    assert len(jev.question_calls) == len(controller.client.messages) == 1
    assert llm.calls == jev.calls == []
    text = reply_text(controller.client)
    assert "没拿到" in text
    assert "Jev 决策结果" not in text and "%" not in text


@pytest.mark.parametrize("error", [JevToolPlanError(), LLMError("auth"), TimeoutError()])
async def test_planning_failure_does_not_call_provider_or_invent_result(tool_controller, error):
    controller, llm, jev, _, _ = tool_controller(planner=FakePlanner(error=error))
    await controller.handle_payload(payload(REQUEST))
    await drain(controller)
    assert jev.calls == jev.question_calls == llm.calls == []
    assert "决策问题" in reply_text(controller.client)
    assert "Jev 决策结果" not in reply_text(controller.client)


async def test_missing_key_reports_actual_configuration_without_fake_result(tool_controller):
    controller, llm, jev, _, _ = tool_controller(configured=False)
    await controller.handle_payload(payload(REQUEST))
    await drain(controller)
    assert jev is None and llm.calls == []
    assert "没配置" in reply_text(controller.client)
    assert "Jev 决策结果" not in reply_text(controller.client)


async def test_public_question_and_report_use_own_memory_source(tool_controller, tmp_path):
    memory = GroupMemoryStore(tmp_path / "tool-memory.sqlite3")
    controller, _, jev, _, _ = tool_controller(memory=memory)
    await controller.handle_payload(payload(REQUEST, message_id=101))
    await drain(controller)
    records = memory.messages("10")
    assert len(jev.question_calls) == 1
    assert len(records) == 2
    assert {row["source"] for row in records} == {"jev_tool"}
    answer = next(row for row in records if row["is_bot"])
    assert "Jev 决策结果" in answer["text"]
    assert answer["reply_to"] == "101"


@pytest.mark.parametrize("bot_mode", ["chat", "assistant"])
async def test_legacy_internal_reports_do_not_enter_requester_context_after_restart(
    tool_controller, tmp_path, bot_mode
):
    memory = GroupMemoryStore(tmp_path / "legacy.sqlite3")
    memory.ingest(
        "10",
        [
            {
                "message_id": "100",
                "user_id": "200",
                "nickname": "甲",
                "text": "我的预算是 200",
                "time": 1,
            }
        ],
    )
    memory.ingest(
        "10",
        [
            {
                "message_id": "101",
                "user_id": "200",
                "nickname": "甲",
                "text": "用 Jev 测试上轮群聊",
                "time": 2,
            },
            {
                "message_id": "102",
                "user_id": "100",
                "nickname": "机器人",
                "text": "Jev 实测：期待回应 25%，接话机会 30%，旧内部报告",
                "time": 3,
                "is_bot": True,
            },
        ],
        source="jev_test",
    )
    controller, _, jev, planner, _ = tool_controller(memory=memory, bot_mode=bot_mode)
    await controller.start()
    await controller.handle_payload(payload(REQUEST, message_id=103))
    await drain(controller)
    context = json.dumps(planner.calls[0]["requester_context"], ensure_ascii=False)
    assert "我的预算" in context
    assert "旧内部报告" not in context and "测试上轮群聊" not in context
    assert jev.calls == []
    assert len(jev.question_calls) == 1


async def test_secrets_redacted_before_planning_and_provider(tool_controller):
    secrets = "typesafe-test-secret llm-test-secret napcat-test-secret"
    plan = JevToolPlan(action="decide", question="要不要 " + secrets, context=secrets, options=[])
    controller, _, jev, planner, _ = tool_controller(planner=FakePlanner([plan]))
    await controller.handle_payload(payload("让 Jev 决策 " + secrets))
    await drain(controller)
    exposed = json.dumps(
        [planner.calls, jev.question_calls, controller.client.messages, controller.status()],
        ensure_ascii=False,
    )
    assert all(secret not in exposed for secret in secrets.split())


async def test_legacy_internal_records_cannot_reappear_through_rag(tool_controller, tmp_path):
    memory = GroupMemoryStore(tmp_path / "legacy-rag.sqlite3")
    memory.ingest(
        "10",
        [
            {
                "message_id": "100",
                "user_id": "200",
                "nickname": "甲",
                "text": "周末爬山记得带水，这是正常的旧聊天",
                "time": 1,
            }
        ],
    )
    memory.ingest(
        "10",
        [
            {
                "message_id": "101",
                "user_id": "200",
                "nickname": "甲",
                "text": "周末爬山的内部旧测试材料，不应进入新对话",
                "time": 2,
            }
        ],
        source="jev_test",
    )
    controller, _, _, _, _ = tool_controller(memory=memory)
    context = await controller._profile_context(["200"], "周末爬山", controller._generation)
    assert "正常的旧聊天" in context
    assert "内部旧测试材料" not in context
    assert len(memory.messages("10")) == 2  # Filtering preserves stored evidence.


async def test_same_user_cannot_queue_second_request_while_provider_running(tool_controller):
    jev = FakeJev(blocked=True)
    controller, _, _, planner, clock = tool_controller(jev=jev)
    await controller.handle_payload(payload(REQUEST, message_id=1))
    await asyncio.wait_for(jev.started.wait(), 1)
    await clock.advance(1)
    await controller.handle_payload(payload(REQUEST, message_id=2))
    await settle()
    assert len(planner.calls) == len(jev.question_calls) == 1
    jev.release.set()
    await drain(controller)
    assert len(controller.client.messages) == 1
    assert controller._jev_tool_tasks == {}


async def test_confirmed_tool_cancels_old_regular_reply_after_planning(tool_controller):
    llm = FakeLLM(
        '{"should_reply":true,"reply":"过时的普通回复"}', blocked=True, swallow_cancel=True
    )
    planner = FakePlanner(blocked=True)
    controller, _, jev, _, _ = tool_controller(llm=llm, planner=planner, jev_mode="off")
    await controller.handle_payload(payload("普通消息", message_id=1))
    await asyncio.wait_for(llm.started.wait(), 1)
    await controller.handle_payload(payload(REQUEST, message_id=2))
    await asyncio.wait_for(planner.started.wait(), 1)
    await settle()
    assert not llm.cancelled
    assert controller.client.messages == []
    planner.release.set()
    await drain(controller)
    assert llm.cancelled
    assert len(jev.question_calls) == len(controller.client.messages) == 1
    assert "过时的普通回复" not in reply_text(controller.client)


async def test_semantic_none_preserves_existing_window_and_counts(tool_controller):
    planner = FakePlanner([JevToolPlan(action="none")], blocked=True)
    controller, llm, jev, _, clock = tool_controller(planner=planner, jev_mode="off", debounce=5)
    await controller.handle_payload(payload("前面的普通群聊", message_id=1))
    await settle()
    first_at = controller._chat_first_message_at
    await clock.advance(2)
    await controller.handle_payload(payload("Jev 是什么原理？", message_id=2))
    await asyncio.wait_for(planner.started.wait(), 1)
    assert controller._chat_first_message_at == first_at
    assert controller._chat_pending_messages == 1
    planner.release.set()
    await settle()
    assert controller._chat_first_message_at == first_at
    assert controller._chat_pending_messages == 2
    await clock.advance(5)
    await drain(controller)
    assert len(llm.calls) == 1
    assert jev.question_calls == []
    text = json.dumps(llm.calls, ensure_ascii=False)
    assert "前面的普通群聊" in text and "Jev 是什么原理" in text


async def test_slow_planner_does_not_cancel_later_regular_window(tool_controller):
    planner = FakePlanner(blocked=True)
    controller, llm, jev, _, clock = tool_controller(planner=planner, jev_mode="off", debounce=5)
    await controller.handle_payload(payload(REQUEST, message_id=1))
    await asyncio.wait_for(planner.started.wait(), 1)
    await controller.handle_payload(payload("后来出现的普通消息", user=201, message_id=2))
    await settle()
    planner.release.set()
    await settle()
    assert len(jev.question_calls) == len(controller.client.messages) == 1
    await clock.advance(5)
    await drain(controller)
    assert len(llm.calls) == 1
    assert "后来出现的普通消息" in json.dumps(llm.calls, ensure_ascii=False)


async def test_pending_report_does_not_replace_new_window_message_batch(tool_controller):
    jev = FakeJev(blocked=True)
    controller, llm, _, _, clock = tool_controller(jev=jev, jev_mode="gate", debounce=5)
    await controller.handle_payload(payload(REQUEST, message_id=1))
    await asyncio.wait_for(jev.started.wait(), 1)
    await controller.handle_payload(payload("后来出现的新话题", user=201, message_id=2))
    await settle()
    jev.release.set()
    await settle()
    await clock.advance(5)
    await drain(controller)
    assert len(jev.question_calls) == len(jev.calls) == 1
    assert [row["text"] for row in jev.calls[0]["new_messages"] if row["speaker"] != "bot"] == [
        "后来出现的新话题"
    ]
    assert len(llm.calls) == 1
    assert len(controller.client.messages) == 1
