"""Profile controller integration uses local SQLite and fake QQ/model services."""

import asyncio
import json
import threading
import time
import zipfile
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.config import Settings
from app.services.control import BotController
from app.services.group_history import normalize_message
from app.services.profile_store import GroupMemoryStore
from app.web.settings import ControlSettings, ControlSettingsStore
from tests.test_control import FakeClient, FakeLLM, drain, payload


def grounded_analysis(user_id="200", message_id="501", *, alias="摸鱼王"):
    return {
        "members": [
            {
                "user_id": user_id,
                "summary": "常用摸鱼梗自嘲",
                "facts": [
                    {
                        "category": "catchphrases",
                        "text": "经常说摸鱼也是生产力",
                        "confidence": "observed",
                        "evidence_ids": [message_id],
                    }
                ],
            }
        ],
        "aliases": [
            {
                "alias": alias,
                "user_ids": [user_id],
                "status": "likely",
                "evidence_ids": [message_id],
                "explanation": "群友用这个绰号接话",
            }
        ],
        "memes": [
            {
                "name": "生产力",
                "meaning": "把摸鱼戏称为生产力",
                "user_ids": [user_id],
                "evidence_ids": [message_id],
            }
        ],
    }


def seed(memory, group_id="10", *, user_id="200", alias="摸鱼王"):
    record = normalize_message(
        payload("摸鱼也是生产力 😂", group=int(group_id), user=int(user_id), message_id=501),
        group_id,
    )
    memory.ingest(group_id, [record])
    memory.save_analysis(group_id, grounded_analysis(user_id, alias=alias))


class ProfileFactory:
    def __init__(self):
        self.models = []
        self.instances = []

    def __call__(self, model):
        self.models.append(model)
        llm = FakeLLM()
        self.instances.append(llm)
        return llm


@pytest.fixture
async def profile_controller(tmp_path):
    controllers = []

    def make(*, mode="assistant", memory=None, llm=None, directory=None):
        directory = directory or tmp_path / f"controller-{len(controllers)}"
        selected_llm = llm or FakeLLM(
            json.dumps({"should_reply": True, "reply": "摸鱼也得算绩效。"})
            if mode == "chat"
            else "摸鱼也得算绩效。"
        )
        config = Settings(
            _env_file=None,
            napcat_access_token="private-napcat-test-token",
            allowed_group_ids=[10],
            llm_api_key="private-model-test-key",
            llm_base_url="http://127.0.0.1:9999/v1",
            llm_model="model-a",
            user_rate_limit_seconds=0.001,
        )
        settings = ControlSettingsStore(
            directory / "settings.json",
            ControlSettings(mode=mode, model="model-a", target_group_id="10"),
        )
        factory = ProfileFactory()
        controller = BotController(
            config,
            settings,
            client=FakeClient(),
            llm_factory=lambda _: selected_llm,
            debounce_seconds=0,
            memory=memory or GroupMemoryStore(directory / "memory" / "groups.sqlite3"),
            profile_llm_factory=factory,
        )
        controllers.append(controller)
        return controller, selected_llm, factory

    yield make
    for controller in controllers:
        await controller.close()


async def test_live_memory_accepts_only_target_group_including_observed_emoji(profile_controller):
    controller, llm, _ = profile_controller()
    ignored = payload("别的群独有记录", group=20, message_id=30)
    await controller.handle_payload(ignored)
    accepted = payload("今天摸鱼 😂", message_id=31)
    accepted["message"].extend(
        [
            {"type": "face", "data": {"id": "14"}},
            {"type": "image", "data": {"file": "unused-private-image"}},
        ]
    )
    await controller.handle_payload(accepted)
    await controller.handle_payload(deepcopy(accepted))
    await controller.handle_payload(payload("机器人上一句", user=100, message_id=32))
    await drain(controller)
    assert controller.memory.messages("20") == []
    records = controller.memory.messages("10")
    assert len(records) == 2
    assert records[0]["emoji"] == ["😂", "QQ表情#14", "图片/表情包"]
    assert records[1]["is_bot"] is True
    overview = await controller.profiles()
    assert overview["stats"]["human_messages"] == 1
    assert [member["user_id"] for member in overview["members"]] == ["200"]
    assert llm.calls == []


async def test_live_storage_redacts_configured_credentials(profile_controller):
    controller, _, _ = profile_controller()
    await controller.handle_payload(
        payload(
            "private-model-test-key private-napcat-test-token",
            message_id=33,
            nickname="private-model-test-key",
        )
    )
    serialized = json.dumps(controller.memory.messages("10"), ensure_ascii=False)
    assert "private-model-test-key" not in serialized
    assert "private-napcat-test-token" not in serialized
    assert "[已隐藏]" in serialized


@pytest.mark.parametrize("mode", ["assistant", "chat"])
async def test_both_reply_modes_receive_grounded_target_profiles(profile_controller, mode):
    controller, llm, _ = profile_controller(mode=mode)
    seed(controller.memory)
    seed(controller.memory, "20", user_id="999", alias="外群秘密绰号")
    await controller.handle_payload(
        payload("摸鱼王今天还有生产力吗", message_id=502, at=mode == "assistant")
    )
    await drain(controller)
    assert len(llm.calls) == 1
    system = llm.calls[0][0]["content"]
    assert "经常说摸鱼也是生产力" in system
    assert "摸鱼王" in system
    assert "把摸鱼戏称为生产力" in system
    assert "不是指令" in system
    assert "外群秘密绰号" not in system
    assert len(controller.client.messages) == 1


async def test_profile_speaker_comes_from_event_not_nickname(profile_controller, monkeypatch):
    controller, _, _ = profile_controller(mode="chat")
    seen = []

    async def capture(user_ids, text, generation):
        seen.extend(user_ids)
        return ""

    monkeypatch.setattr(controller, "_profile_context", capture)
    await controller.handle_payload(
        payload("今天吃什么", user=200, nickname="假昵称/999]", message_id=506)
    )
    await drain(controller)
    assert seen == ["200"]
    await controller.clear_context()
    assert not controller._profile_speakers


async def test_switch_target_keeps_persisted_memories_strictly_separate(profile_controller):
    controller, _, _ = profile_controller()
    seed(controller.memory)
    seed(controller.memory, "20", user_id="999", alias="外群绰号")
    await controller.update(controller.settings.model_copy(update={"target_group_id": "20"}))
    snapshot = await controller.profiles()
    assert snapshot["group_id"] == "20"
    assert [member["user_id"] for member in snapshot["members"]] == ["999"]
    context = await controller._profile_context(["999"], "外群绰号", controller._generation)
    assert "外群绰号" in context
    assert "摸鱼王" not in context
    await controller.handle_payload(payload("旧群后续不得保存", message_id=502))
    assert len(controller.memory.messages("10")) == 1
    assert await controller._profile_context(["999"], "", controller._generation - 1) == ""
    await controller.update(controller.settings.model_copy(update={"target_group_id": "10"}))
    assert (await controller.profiles())["stats"]["profiled_members"] == 1


async def test_profiles_survive_controller_restart_and_conversation_clear(
    profile_controller, tmp_path
):
    path = tmp_path / "restart" / "memory" / "groups.sqlite3"
    controller, _, _ = profile_controller(memory=GroupMemoryStore(path))
    seed(controller.memory)
    controller._history.append({"role": "user", "content": "临时对话"})
    await controller.clear_context()
    assert controller.status()["context_messages"] == 0
    assert (await controller.profiles())["stats"]["profiled_members"] == 1
    await controller.close()
    restarted, _, _ = profile_controller(memory=GroupMemoryStore(path))
    assert (await restarted.profiles())["members"][0]["profile"]["summary"] == "常用摸鱼梗自嘲"
    assert "摸鱼王" in await restarted._profile_context(["200"], "", restarted._generation)


async def test_refresh_imports_mocked_history_analyzes_and_closes_model(
    profile_controller, monkeypatch
):
    controller, _, factory = profile_controller()
    imported = normalize_message(payload("摸鱼也是生产力", message_id=501), "10")
    calls = []

    async def fetch(client, group_id):
        calls.append(("history", group_id))
        return "100", [{"user_id": "200", "nickname": "小明", "card": "明哥"}], [imported]

    async def analyze(llm, messages, members, **kwargs):
        calls.append(("analysis", [message["message_id"] for message in messages]))
        assert members[0]["display_name"] in {"小明", "明哥"}
        return grounded_analysis()

    monkeypatch.setattr("app.services.control.fetch_group_history", fetch)
    monkeypatch.setattr("app.services.control.analyze_profiles", analyze)
    await controller.refresh_profiles("10")
    await asyncio.wait_for(controller._profile_task, 2)
    overview = await controller.profiles()
    assert calls == [("history", "10"), ("analysis", ["501"])]
    assert overview["stats"]["profiled_members"] == 1
    assert overview["job"]["running"] is False
    assert overview["job"]["error"] is None
    assert overview["job"]["updated_at"] is not None
    assert overview["job"]["cleaning"]["original_count"] == 1
    # FakeLLM deliberately returns plain text for cleaning; the module falls back
    # to original records and analysis continues successfully.
    assert overview["job"]["cleaning"]["ai_failed_batches"] == 1
    assert overview["job"]["cleaning"]["ai_used"] is False
    assert factory.models == ["model-a"]
    assert factory.instances[0].closed is True
    assert controller.client.messages == []


async def test_backfill_covers_older_messages_and_continues_after_invalid_batch(
    profile_controller, monkeypatch
):
    controller, _, _ = profile_controller()
    records = [
        normalize_message(payload(f"摸鱼也是生产力，独立消息{i}", message_id=1000 + i), "10")
        for i in range(51)
    ]
    records.append(normalize_message(payload("摸鱼也是生产力", user=201, message_id=1051), "10"))
    inspected = []

    async def fetch(client, group_id):
        return "100", [], records

    async def analyze(llm, messages, members, *, focus_user_ids):
        uid = focus_user_ids[0]
        own = [m for m in messages if m["user_id"] == uid]
        inspected.extend(m["message_id"] for m in own)
        if len(inspected) == 51:
            raise ValueError("invalid model response")
        return grounded_analysis(uid, own[-1]["message_id"])

    monkeypatch.setattr("app.services.control.fetch_group_history", fetch)
    monkeypatch.setattr("app.services.control.analyze_profiles", analyze)
    await controller.refresh_profiles("10")
    await asyncio.wait_for(controller._profile_task, 2)
    result = await controller.profiles()
    assert set(inspected) == {str(1000 + i) for i in range(52)}
    assert result["job"]["running"] is False
    assert result["job"]["error"]
    assert any(member["user_id"] == "201" and member["profile"] for member in result["members"])
    assert controller.client.messages == []


@pytest.mark.parametrize(
    "change",
    [
        {"enabled": False},
        {"mode": "chat"},
        {"target_group_id": "20"},
    ],
)
async def test_settings_change_cancels_analysis_and_discards_even_late_results(
    profile_controller, monkeypatch, change
):
    controller, _, factory = profile_controller()
    controller.memory.ingest("10", [normalize_message(payload(message_id=501), "10")])
    started = asyncio.Event()
    never_released = asyncio.Event()
    cancelled = asyncio.Event()

    async def analyze(llm, messages, members, **kwargs):
        started.set()
        try:
            await never_released.wait()
        except asyncio.CancelledError:
            # A provider finishing just as it is cancelled must not write stale findings.
            cancelled.set()
        return grounded_analysis()

    monkeypatch.setattr("app.services.control.analyze_profiles", analyze)
    await controller.refresh_profiles("10", backfill=False)
    await asyncio.wait_for(started.wait(), 2)
    first_task = controller._profile_task
    await controller.refresh_profiles("10", backfill=False)
    assert controller._profile_task is first_task
    await controller.update(controller.settings.model_copy(update=change))
    assert cancelled.is_set()
    assert controller.memory.overview("10")["stats"]["profiled_members"] == 0
    assert controller.memory.overview("20")["members"] == []
    assert factory.instances[0].closed


async def test_refresh_cleans_before_analysis_and_exposes_real_cleaning_stats(
    profile_controller, monkeypatch
):
    controller, _, _ = profile_controller()
    records = [
        normalize_message(payload("这一条原始聊天内容很完整用于保留上下文", message_id=601), "10"),
        normalize_message(payload("中间完全普通冗余消息不包含个人风格线索", message_id=602), "10"),
        normalize_message(payload("最后一条原始聊天记录作为近期上下文保留", message_id=603), "10"),
    ]
    order = []

    class CleaningLLM(FakeLLM):
        async def chat_json(self, prompt):
            order.append("cleaning")
            self.calls.append(deepcopy(prompt))
            assert "数据清洗" in prompt[0]["content"]
            assert controller._profile_job["stage"] == "AI 清洗重复与无效聊天记录"
            return json.dumps({"keep_message_ids": ["601", "603"]})

    cleaner = CleaningLLM()
    controller._profile_llm_factory = lambda _: cleaner

    async def fetch(client, group_id):
        order.append("history")
        return "100", [], records

    async def analyze(llm, messages, members, **kwargs):
        order.append("analysis")
        assert llm is cleaner
        assert [item["message_id"] for item in messages] == ["601", "603"]
        assert all(item["repeat_count"] == 1 for item in messages)
        return grounded_analysis(message_id="601")

    monkeypatch.setattr("app.services.control.fetch_group_history", fetch)
    monkeypatch.setattr("app.services.control.analyze_profiles", analyze)
    await controller.refresh_profiles("10")
    await asyncio.wait_for(controller._profile_task, 2)
    result = await controller.profiles()
    assert order == ["history", "cleaning", "analysis"]
    assert result["job"]["cleaning"]["original_count"] == 3
    assert result["job"]["cleaning"]["rule_kept_count"] == 3
    assert result["job"]["cleaning"]["ai_kept_count"] == 2
    assert result["job"]["cleaning"]["ai_removed_count"] == 1
    assert result["job"]["cleaning"]["ai_used"] is True
    assert len(controller.memory.messages("10")) == 3  # Cleaning never deletes stored originals.
    assert cleaner.closed and controller.client.messages == []


async def test_refresh_deduplicates_before_batching_and_retains_repeat_evidence(
    profile_controller, monkeypatch
):
    controller, _, _ = profile_controller()
    records = [normalize_message(payload("离谱", message_id=700 + i), "10") for i in range(51)]
    analyzed = []

    async def fetch(client, group_id):
        return "100", [], records

    async def analyze(llm, messages, members, **kwargs):
        analyzed.append(messages)
        return grounded_analysis(message_id=messages[0]["message_id"])

    monkeypatch.setattr("app.services.control.fetch_group_history", fetch)
    monkeypatch.setattr("app.services.control.analyze_profiles", analyze)
    await controller.refresh_profiles("10")
    await asyncio.wait_for(controller._profile_task, 2)
    assert len(analyzed) == 1
    assert [item["message_id"] for item in analyzed[0]] == ["700", "750"]
    assert all(item["repeat_count"] == 51 for item in analyzed[0])
    assert all(len(item["repeat_message_ids"]) == 51 for item in analyzed[0])
    assert (await controller.profiles())["job"]["cleaning"]["duplicate_count"] == 49


@pytest.mark.parametrize("mode", ["assistant", "chat"])
async def test_retrieved_original_history_reaches_both_actual_model_requests(
    profile_controller, mode
):
    controller, llm, _ = profile_controller(mode=mode)
    old_text = "星光联赛结束后我们约好周五去海边烧烤，这是当时的约定。"
    other_text = "星光联赛外群独有秘密约定是星期天去北极。"
    controller.memory.ingest("10", [normalize_message(payload(old_text, message_id=801), "10")])
    controller.memory.ingest(
        "20", [normalize_message(payload(other_text, group=20, message_id=802), "20")]
    )
    await controller.handle_payload(
        payload("星光联赛结束后之前约了啥？", message_id=803, at=mode == "assistant")
    )
    await drain(controller)
    assert len(llm.calls) == 1
    system = llm.calls[0][0]["content"]
    assert "周五去海边烧烤" in system
    assert "历史原话" in system
    assert '"message_id": "801"' in system
    assert other_text not in system
    assert len(controller.client.messages) == 1


@pytest.mark.parametrize("mode", ["assistant", "chat"])
async def test_switch_target_during_retrieval_drops_late_old_group_memory(
    profile_controller, monkeypatch, mode
):
    controller, llm, _ = profile_controller(mode=mode)
    controller.memory.ingest(
        "10", [normalize_message(payload("星光联赛旧群秘密是海边烧烤", message_id=811), "10")]
    )
    controller.memory.ingest(
        "20",
        [normalize_message(payload("星光联赛新群回忆是山顶露营", group=20, message_id=812), "20")],
    )
    began, release = threading.Event(), threading.Event()
    real_search = controller.memory.search

    def blocked_search(group_id, query, **kwargs):
        result = real_search(group_id, query, **kwargs)
        if group_id == "10":
            began.set()
            assert release.wait(2)
        return result

    monkeypatch.setattr(controller.memory, "search", blocked_search)
    stale_context = asyncio.create_task(
        controller._profile_context(["200"], "星光联赛", controller._generation)
    )
    try:
        assert await asyncio.to_thread(began.wait, 1)
        await controller.update(controller.settings.model_copy(update={"target_group_id": "20"}))
    finally:
        release.set()
    assert await asyncio.wait_for(stale_context, 2) == ""
    await controller.handle_payload(
        payload("星光联赛当时说什么？", group=20, message_id=813, at=mode == "assistant")
    )
    await drain(controller)
    assert len(llm.calls) == 1
    system = llm.calls[0][0]["content"]
    assert "山顶露营" in system and "海边烧烤" not in system


@pytest.mark.parametrize("mode", ["assistant", "chat"])
async def test_manual_alias_mapping_remains_authoritative_in_reply_context(
    profile_controller, mode
):
    controller, llm, _ = profile_controller(mode=mode)
    seed(controller.memory)
    controller.memory.ingest_members("10", [{"user_id": "201", "nickname": "小红", "card": ""}])
    await controller.correct_alias("10", "摸鱼王", ["201"])
    controller.memory.save_analysis("10", grounded_analysis())
    await controller.handle_payload(
        payload("摸鱼王刚才说什么？", message_id=821, at=mode == "assistant")
    )
    await drain(controller)
    system = llm.calls[0][0]["content"]
    context_line = next(line for line in system.splitlines() if line.startswith('{"group_id":'))
    assert "source=manual" in system
    assert "不能根据重复玩笑推翻已经确认的对应关系" in system
    context = json.loads(context_line)
    actual = next(item for item in context["aliases"] if item["alias"] == "摸鱼王")
    assert actual["user_ids"] == ["201"] and actual["source"] == "manual"


async def test_start_restores_only_current_groups_recent_history_without_sending(
    profile_controller, monkeypatch
):
    controller, llm, factory = profile_controller(mode="chat")
    records = []
    for i in range(3):
        row = normalize_message(
            payload(f"本群已保存历史{i}", message_id=830 + i, user=100 if i == 1 else 200), "10"
        )
        row["time"] = i + 1
        records.append(row)
    controller.memory.ingest("10", records)
    controller.memory.ingest(
        "20", [normalize_message(payload("绝不能恢复的其他群历史", group=20, message_id=840), "20")]
    )
    waiting = asyncio.Event()

    async def run(handler):
        await waiting.wait()

    monkeypatch.setattr(controller.client, "run", run, raising=False)
    await controller.start()
    restored = list(controller._history)
    assert [item["role"] for item in restored] == ["user", "assistant", "user"]
    assert all("本群已保存历史" in item["content"] for item in restored)
    assert list(controller._profile_speakers) == ["200", "200"]
    assert llm.calls == [] and factory.instances == [] and controller.client.messages == []


@pytest.mark.parametrize("existing_profile", [False, True])
async def test_auto_bootstrap_backfills_once_only_when_local_profiles_are_absent(
    profile_controller, monkeypatch, existing_profile
):
    controller, _, _ = profile_controller()
    if existing_profile:
        seed(controller.memory)
    refresh = AsyncMock(return_value={})
    monkeypatch.setattr(controller, "refresh_profiles", refresh)
    real_sleep = asyncio.sleep
    ticks = 0

    async def fast_tick(delay):
        nonlocal ticks
        assert delay == 5
        ticks += 1
        if ticks > 2:
            raise asyncio.CancelledError
        await real_sleep(0)

    with monkeypatch.context() as scoped:
        scoped.setattr("app.services.control.asyncio.sleep", fast_tick)
        with pytest.raises(asyncio.CancelledError):
            await controller._auto_profiles()
    if existing_profile:
        refresh.assert_not_called()
    else:
        refresh.assert_awaited_once_with("10")
    assert controller.client.messages == []


async def test_export_first_use_persists_effective_controls_and_produces_valid_migration(
    profile_controller, tmp_path
):
    controller, llm, factory = profile_controller()
    assert not controller.store.path.exists()
    seed(controller.memory)
    destination = await controller.export_memory(tmp_path / "first-export.zip")
    assert controller.store.load() == controller.settings
    assert controller.settings.enabled is True
    with zipfile.ZipFile(destination) as archive:
        controls = json.loads(archive.read("control.json"))
        assert controls["target_group_id"] == "10"
        assert controls["enabled"] is False
        assert "memory/groups.sqlite3" in archive.namelist()
        assert "private-model-test-key" not in archive.read("control.json").decode()
    assert llm.calls == [] and factory.instances == [] and controller.client.messages == []


async def test_bootstrap_disconnect_during_lookup_does_not_stop_loop_and_retries_later(
    profile_controller, monkeypatch
):
    controller, _, _ = profile_controller()
    real_overview = controller.memory.overview
    real_refresh = controller.refresh_profiles
    real_sleep = asyncio.sleep
    clock = [0.0]
    first_lookup = True
    attempts = []
    ticks = iter([0, 10, 301, 302, 700])

    def disconnect_during_first_lookup(group_id):
        nonlocal first_lookup
        snapshot = real_overview(group_id)
        if first_lookup:
            first_lookup = False
            controller.client.is_connected = False
        return snapshot

    async def refresh(group_id, **kwargs):
        attempts.append(clock[0])
        return await real_refresh(group_id, **kwargs)

    async def fetch(client, group_id):
        return "100", [], []

    async def tick(delay):
        assert delay == 5
        try:
            clock[0] = next(ticks)
        except StopIteration:
            raise asyncio.CancelledError from None
        if clock[0] >= 10:
            controller.client.is_connected = True
        if controller._profile_task:
            await controller._profile_task
        await real_sleep(0)

    monkeypatch.setattr(controller.memory, "overview", disconnect_during_first_lookup)
    monkeypatch.setattr(controller, "refresh_profiles", refresh)
    monkeypatch.setattr("app.services.control.fetch_group_history", fetch)
    with monkeypatch.context() as scoped:
        scoped.setattr(
            "app.services.control.time", SimpleNamespace(monotonic=lambda: clock[0], time=time.time)
        )
        scoped.setattr("app.services.control.asyncio.sleep", tick)
        with pytest.raises(asyncio.CancelledError):
            await controller._auto_profiles()
    assert attempts == [0, 301]
    assert "10" in controller._profile_bootstrapped
    assert controller.client.messages == []


async def test_failed_bootstrap_task_retries_after_cooldown_then_stops_after_success(
    profile_controller, monkeypatch
):
    controller, _, _ = profile_controller()
    real_sleep = asyncio.sleep
    clock = [0.0]
    attempts = []
    ticks = iter([0, 1, 300, 302, 303, 900])

    async def refresh(group_id, **kwargs):
        attempts.append(clock[0])

        async def completed_job():
            controller._profile_job["error"] = "temporary failure" if len(attempts) == 1 else None

        controller._profile_task = asyncio.create_task(completed_job())
        return {}

    async def tick(delay):
        assert delay == 5
        try:
            clock[0] = next(ticks)
        except StopIteration:
            raise asyncio.CancelledError from None
        if controller._profile_task:
            await controller._profile_task
        await real_sleep(0)

    monkeypatch.setattr(controller, "refresh_profiles", refresh)
    with monkeypatch.context() as scoped:
        scoped.setattr(
            "app.services.control.time", SimpleNamespace(monotonic=lambda: clock[0], time=time.time)
        )
        scoped.setattr("app.services.control.asyncio.sleep", tick)
        with pytest.raises(asyncio.CancelledError):
            await controller._auto_profiles()
    assert attempts == [0, 302]
    assert "10" in controller._profile_bootstrapped
    assert controller.client.messages == []


async def test_alias_correction_is_current_group_only_and_survives_analysis(profile_controller):
    controller, _, _ = profile_controller()
    seed(controller.memory)
    controller.memory.ingest_members("10", [{"user_id": "201", "nickname": "小红", "card": ""}])
    result = await controller.correct_alias("10", "摸鱼王", ["201"])
    corrected = next(alias for alias in result["aliases"] if alias["alias"] == "摸鱼王")
    assert corrected["user_ids"] == ["201"]
    assert corrected["source"] == "manual"
    controller.memory.save_analysis("10", grounded_analysis())
    assert next(
        alias for alias in (await controller.profiles())["aliases"] if alias["alias"] == "摸鱼王"
    )["user_ids"] == ["201"]
    with pytest.raises(ValueError):
        await controller.correct_alias("20", "摸鱼王", ["201"])
    with pytest.raises(ValueError):
        await controller.correct_alias("10", "摸鱼王", ["999"])
    with pytest.raises(ValueError):
        await controller.refresh_profiles("20")


async def test_failed_profile_analysis_is_reported_without_raw_error_or_invented_facts(
    profile_controller, monkeypatch
):
    controller, _, factory = profile_controller()
    controller.memory.ingest("10", [normalize_message(payload(message_id=501), "10")])

    async def analyze(llm, messages, members, **kwargs):
        raise RuntimeError("private-model-test-key confidential provider response")

    monkeypatch.setattr("app.services.control.analyze_profiles", analyze)
    await controller.refresh_profiles("10", backfill=False)
    await asyncio.wait_for(controller._profile_task, 2)
    result = await controller.profiles()
    assert result["job"]["error"]
    assert "private-model-test-key" not in json.dumps(result)
    assert "confidential" not in json.dumps(result)
    assert result["stats"]["profiled_members"] == 0
    assert factory.instances[0].closed
