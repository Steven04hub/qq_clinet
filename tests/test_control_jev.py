"""Jev routing must not bypass the existing QQ lifecycle or send budget."""

import asyncio
import json
from copy import deepcopy

import pytest

from app.config import Settings
from app.services.control import BotController
from app.services.jev import JevDecision, JevError
from app.web.settings import ControlSettings, ControlSettingsStore
from tests.test_chat_window import settle
from tests.test_control import FakeClient, FakeLLM, drain, payload


class FakeJev:
    def __init__(self, route="reply", *, error=None, blocked=False, swallow_cancel=False):
        self.route = route
        self.error = error
        self.calls = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        if not blocked:
            self.release.set()
        self.swallow_cancel = swallow_cancel
        self.cancelled = False
        self.closed = False

    async def evaluate(self, state):
        self.calls.append(deepcopy(state))
        self.started.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            if not self.swallow_cancel:
                raise
        if self.error:
            raise self.error
        return JevDecision(self.route, "test_reason", {}, 12, 100)

    async def close(self):
        self.closed = True


@pytest.fixture
async def controller_factory(tmp_path):
    created = []

    def make(*, jev=None, mode="gate", bot_mode="chat", limit=10):
        llm = FakeLLM('{"should_reply":true,"reply":"这就叫提前下班"}')
        gate = jev or FakeJev()
        config = Settings(
            _env_file=None,
            napcat_access_token="napcat-test-secret",
            llm_api_key="llm-test-secret",
            typesafe_api_key="typesafe-test-secret",
            jev_mode=mode,
            profile_auto_import=False,
        )
        store = ControlSettingsStore(
            tmp_path / f"control-{len(created)}.json",
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
            client=FakeClient(),
            llm_factory=lambda _: llm,
            jev_client=gate,
            debounce_seconds=0,
        )
        created.append((controller, gate))
        return controller, llm, gate

    yield make
    for controller, gate in created:
        gate.release.set()
        await controller.close()
        assert gate.closed


async def test_gate_silence_skips_generation_and_sending(controller_factory):
    controller, llm, gate = controller_factory(jev=FakeJev("silent"))
    await controller.handle_payload(payload())
    await drain(controller)
    assert len(gate.calls) == 1
    assert llm.calls == []
    assert controller.client.messages == []
    assert controller._chat_silent_turns == 1
    assert controller.budget.used == 0
    assert controller.status()["decision_model"]["last"]["route"] == "silent"


@pytest.mark.parametrize("route", ["reply", "fallback"])
async def test_positive_and_uncertain_routes_call_llm_once(controller_factory, route):
    controller, llm, gate = controller_factory(jev=FakeJev(route))
    await controller.handle_payload(payload(at=True))
    await drain(controller)
    assert len(gate.calls) == len(llm.calls) == len(controller.client.messages) == 1
    assert gate.calls[0]["bot_id"] == "bot"
    assert gate.calls[0]["new_messages"]


@pytest.mark.parametrize("kind", ["auth", "timeout", "malformed", "connection"])
async def test_jev_failure_falls_back_without_disabling_chat(controller_factory, kind):
    controller, llm, gate = controller_factory(jev=FakeJev(error=JevError(kind)))
    await controller.handle_payload(payload())
    await drain(controller)
    assert len(gate.calls) == len(llm.calls) == len(controller.client.messages) == 1
    assert controller.status()["last_error"] is None
    assert controller.status()["decision_model"]["last"]["reason"] == kind


async def test_shadow_cannot_delay_or_suppress_reply(controller_factory):
    controller, llm, gate = controller_factory(jev=FakeJev("silent", blocked=True), mode="shadow")
    await controller.handle_payload(payload())
    await settle()
    assert gate.started.is_set()
    assert len(llm.calls) == len(controller.client.messages) == 1
    gate.release.set()
    await drain(controller)
    assert controller._chat_silent_turns == 0
    assert controller.status()["decision_model"]["last"]["route"] == "silent"


async def test_off_mode_makes_no_decision_call(controller_factory):
    controller, llm, gate = controller_factory(mode="off")
    await controller.handle_payload(payload())
    await drain(controller)
    assert gate.calls == []
    assert len(llm.calls) == 1


async def test_assistant_mode_does_not_use_jev(controller_factory):
    controller, llm, gate = controller_factory(bot_mode="assistant")
    llm.answer = "简短答案"
    await controller.handle_payload(payload(at=True))
    await drain(controller)
    assert gate.calls == []
    assert len(llm.calls) == 1


async def test_other_groups_and_reply_budget_precede_jev(controller_factory):
    controller, llm, gate = controller_factory(limit=1)
    await controller.handle_payload(payload(group=20))
    await drain(controller)
    assert gate.calls == []
    assert controller.budget.allow()
    await controller.handle_payload(payload())
    await drain(controller)
    assert gate.calls == []
    assert llm.calls == []


async def test_pause_invalidates_gate_even_if_it_swallows_cancellation(controller_factory):
    controller, llm, gate = controller_factory(jev=FakeJev(blocked=True, swallow_cancel=True))
    await controller.handle_payload(payload())
    await asyncio.wait_for(gate.started.wait(), 1)
    await controller.update(controller.settings.model_copy(update={"enabled": False}))
    await drain(controller)
    assert gate.cancelled
    assert llm.calls == []
    assert controller.client.messages == []
    assert controller.status()["decision_model"]["last"] is None


async def test_new_window_supersedes_pending_gate(controller_factory):
    controller, llm, gate = controller_factory(jev=FakeJev(blocked=True, swallow_cancel=True))
    await controller.handle_payload(payload("第一轮", message_id=1))
    await asyncio.wait_for(gate.started.wait(), 1)
    await controller.handle_payload(payload("第二轮新内容", message_id=2))
    await settle()
    assert gate.cancelled
    assert len(gate.calls) == 2
    assert llm.calls == []
    gate.release.set()
    await drain(controller)
    assert len(llm.calls) == len(controller.client.messages) == 1
    assert len(gate.calls[1]["new_messages"]) == 1


async def test_secret_redaction_covers_jev_state_activity_and_status(controller_factory):
    controller, _, gate = controller_factory(jev=FakeJev("silent"))
    text = "typesafe-test-secret llm-test-secret napcat-test-secret"
    await controller.handle_payload(payload(text, nickname="typesafe-test-secret"))
    await drain(controller)
    exposed = json.dumps([controller.status(), gate.calls], ensure_ascii=False)
    for secret in text.split():
        assert secret not in exposed


async def test_reply_route_still_allows_generator_silence(controller_factory):
    controller, llm, _ = controller_factory()
    llm.answer = '{"should_reply":false,"reply":""}'
    await controller.handle_payload(payload())
    await drain(controller)
    assert len(llm.calls) == 1
    assert controller.client.messages == []


def test_missing_jev_key_keeps_existing_chat_available(tmp_path):
    config = Settings(_env_file=None, napcat_access_token="test", jev_mode="gate")
    controller = BotController(
        config,
        ControlSettingsStore(
            tmp_path / "control.json", ControlSettings(model="test", target_group_id="10")
        ),
        client=FakeClient(),
        llm_factory=lambda _: FakeLLM(),
    )
    assert controller._jev is None
    assert controller.status()["decision_model"]["configured"] is False
    assert controller.settings.enabled
