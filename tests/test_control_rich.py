"""Rich chat behavior stays local: fake OneBot/model services plus temporary assets."""

import asyncio
import base64
import json

import pytest

from app.config import Settings
from app.services.control import BotController
from app.services.conversation import ChatDecision
from app.services.profile_store import GroupMemoryStore
from app.services.stickers import StickerLibrary
from app.web.settings import ControlSettings, ControlSettingsStore
from tests.test_control import FakeClient, FakeLLM, drain, payload, reply_text

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGNgYGAAAAAEAAH2FzhVAAAAAElFTkSuQmCC"
)


@pytest.fixture
async def rich_controller(tmp_path):
    controllers = []

    def make(*, answer=None, cap=10, client=None, with_memory=False):
        directory = tmp_path / f"controller-{len(controllers)}"
        llm = FakeLLM(answer or '{"should_reply":false,"reply":""}')
        config = Settings(
            _env_file=None,
            napcat_access_token="test-napcat-token",
            allowed_group_ids=[10],
            llm_api_key="test-model-key",
            llm_base_url="http://127.0.0.1:9999/v1",
            llm_model="model-a",
            user_rate_limit_seconds=0.001,
        )
        store = ControlSettingsStore(
            directory / "control.json",
            ControlSettings(
                mode="chat", model="model-a", target_group_id="10", reply_limit_per_minute=cap
            ),
        )
        controller = BotController(
            config,
            store,
            client=client or FakeClient(),
            llm_factory=lambda _: llm,
            debounce_seconds=0,
            stickers=StickerLibrary(directory / "stickers"),
            memory=GroupMemoryStore(directory / "memory" / "groups.sqlite3")
            if with_memory
            else None,
        )
        controllers.append(controller)
        return controller, llm

    yield make
    for controller in controllers:
        await controller.close()


async def test_at_text_and_face_are_one_send_one_budget_slot_and_remembered(rich_controller):
    controller, llm = rich_controller(
        answer=json.dumps(
            {
                "should_reply": True,
                "reply": "你可太会了。下一个问题。",
                "mention_user_id": "200",
                "sticker_id": "face_13",
            }
        ),
        with_memory=True,
    )
    await controller.handle_payload(payload("这也能行？", message_id=501, nickname="群友"))
    await drain(controller)
    assert controller.client.messages == [
        (
            10,
            [
                {"type": "at", "data": {"qq": "200"}},
                {"type": "text", "data": {"text": " 你可太会了"}},
                {"type": "face", "data": {"id": "13"}},
            ],
        )
    ]
    assert controller.budget.used == 1
    sent_history = [item for item in controller._history if item["role"] == "assistant"]
    assert len(sent_history) == 1
    assert "你可太会了" in sent_history[0]["content"]
    assert "@群友" in sent_history[0]["content"]
    assert "face_13" in sent_history[0]["content"]
    bot_records = [row for row in controller.memory.messages("10") if row["is_bot"]]
    assert len(bot_records) == 1
    assert bot_records[0]["mentions"] == ["200"]
    assert bot_records[0]["emoji"] == ["face_13"]
    prompt = llm.calls[0][0]["content"]
    mention_catalog = json.loads(prompt.split("可 @ 名单（JSON 数据）：", 1)[1].splitlines()[0])
    sticker_catalog = json.loads(prompt.split("可用表情目录（JSON 数据）：", 1)[1].splitlines()[0])
    assert mention_catalog == [{"user_id": "200", "name": "群友"}]
    assert "face_13" in {item["id"] for item in sticker_catalog}


async def test_sticker_only_reply_is_sent_and_kept_in_history(rich_controller):
    controller, _ = rich_controller(
        answer='{"should_reply":true,"reply":"","sticker_id":"face_14"}'
    )
    await controller.handle_payload(payload(message_id=502))
    await drain(controller)
    assert controller.client.messages == [(10, [{"type": "face", "data": {"id": "14"}}])]
    assert controller.budget.used == 1
    assert controller._history[-1]["role"] == "assistant"
    assert "face_14" in controller._history[-1]["content"]


async def test_registered_image_only_reply_uses_local_verified_base64(rich_controller):
    controller, llm = rich_controller()
    sticker = controller.stickers.register(PNG, label="看热闹", tags=["吃瓜"])
    llm.answer = json.dumps({"should_reply": True, "reply": "", "sticker_id": sticker["id"]})
    await controller.handle_payload(payload(message_id=503))
    await drain(controller)
    assert len(controller.client.messages) == 1
    group_id, segments = controller.client.messages[0]
    assert group_id == 10
    assert len(segments) == 1 and segments[0]["type"] == "image"
    assert base64.b64decode(segments[0]["data"]["file"].removeprefix("base64://")) == PNG
    assert controller.budget.used == 1
    assert str(controller.stickers.path) not in json.dumps(llm.calls)


@pytest.mark.parametrize(
    "extra",
    [
        {"mention_user_id": "999"},
        {"sticker_id": "face_999"},
        {"sticker_id": "img_unknown"},
        {"mention_user_id": "all"},
        {"mention_user_id": "0"},
        {"sticker_id": "file:///etc/passwd"},
        {"sticker_id": "../../private.png"},
        {"sticker_id": "https://example.org/not-a-local-sticker.png"},
    ],
)
async def test_unknown_recipients_and_arbitrary_assets_never_send_or_use_budget(
    rich_controller, extra
):
    controller, _ = rich_controller(
        answer=json.dumps({"should_reply": True, "reply": "这条不能发", **extra})
    )
    await controller.handle_payload(payload(message_id=504))
    await drain(controller)
    assert controller.client.attempts == []
    assert controller.budget.used == 0
    assert all(item["role"] != "assistant" for item in controller._history)


async def test_plain_cleanup_keeps_real_question_and_removes_empty_host_reply(rich_controller):
    controller, llm = rich_controller(answer='{"should_reply":true,"reply":"下一题！"}')
    await controller.handle_payload(payload(message_id=505))
    await drain(controller)
    assert controller.client.messages == []
    assert controller.budget.used == 0
    llm.answer = '{"should_reply":true,"reply":"你今晚几点来？"}'
    await controller.handle_payload(payload("今晚还去", message_id=506))
    await drain(controller)
    assert reply_text(controller.client) == "你今晚几点来？"


async def test_repeated_plain_reply_is_suppressed_after_new_messages(rich_controller):
    controller, llm = rich_controller(
        answer='{"should_reply":true,"reply":"你这个操作属实给我整不会了"}'
    )
    await controller.handle_payload(payload(message_id=507))
    await drain(controller)
    llm.answer = '{"should_reply":true,"reply":"你这个操作属实给我整不会了！"}'
    await controller.handle_payload(payload("那你再看看这个", message_id=508))
    await drain(controller)
    assert len(llm.calls) == 2
    assert len(controller.client.messages) == 1
    assert controller.budget.used == 1


async def test_repeated_rich_reply_is_suppressed_even_with_caption_metadata(rich_controller):
    controller, llm = rich_controller(
        answer=json.dumps(
            {
                "should_reply": True,
                "reply": "你可太会了",
                "mention_user_id": "200",
                "sticker_id": "face_13",
            }
        )
    )
    await controller.handle_payload(payload(message_id=509))
    await drain(controller)
    await controller.handle_payload(payload("你再看看", message_id=510))
    await drain(controller)
    assert len(llm.calls) == 2
    assert len(controller.client.messages) == 1
    assert controller.budget.used == 1


async def test_rich_and_plain_messages_share_the_same_rolling_budget(rich_controller):
    controller, llm = rich_controller(
        cap=1, answer='{"should_reply":true,"reply":"","sticker_id":"face_1"}'
    )
    await controller.handle_payload(payload(message_id=511))
    await drain(controller)
    llm.answer = '{"should_reply":true,"reply":"再发一句"}'
    await controller.handle_payload(payload("还有这个", message_id=512))
    await drain(controller)
    assert len(controller.client.messages) == 1
    assert len(llm.calls) == 1
    assert await controller._send(10, "命令也不能超限", controller._generation) == ""
    assert controller.budget.used == 1


async def test_failed_rich_send_reserves_budget_without_automatic_retry(rich_controller):
    client = FakeClient(send_error=RuntimeError("fake transport failure"))
    controller, llm = rich_controller(
        client=client, answer='{"should_reply":true,"reply":"","sticker_id":"face_1"}'
    )
    await controller.handle_payload(payload(message_id=513))
    await drain(controller)
    await asyncio.sleep(0)
    assert len(client.attempts) == 1
    assert client.messages == []
    assert len(llm.calls) == 1
    assert controller.budget.used == 1
    assert all(item["role"] != "assistant" for item in controller._history)


async def test_target_switch_clears_old_recipients_and_rejects_cross_group_mentions(
    rich_controller,
):
    controller, llm = rich_controller()
    await controller.handle_payload(payload(message_id=514))
    await drain(controller)
    assert controller._seen_members == {"200": "小明"}
    await controller.handle_payload(payload("别的群的用户", group=20, user=300, message_id=515))
    assert "300" not in controller._seen_members
    await controller.update(controller.settings.model_copy(update={"target_group_id": "20"}))
    assert controller._seen_members == {}
    llm.answer = '{"should_reply":true,"reply":"不能叫旧群成员","mention_user_id":"200"}'
    await controller.handle_payload(payload("新群的消息", group=20, user=300, message_id=516))
    await drain(controller)
    assert controller._seen_members == {"300": "小明"}
    assert controller.client.messages == []
    mention_catalog = json.loads(
        llm.calls[-1][0]["content"].split("可 @ 名单（JSON 数据）：", 1)[1].splitlines()[0]
    )
    assert [item["user_id"] for item in mention_catalog] == ["300"]


async def test_stale_generation_and_disconnected_client_do_not_send_rich_reply(rich_controller):
    controller, _ = rich_controller()
    decision = ChatDecision(True, "", sticker_id="face_13")
    assert await controller._send_chat(10, decision, controller._generation - 1) == ""
    controller.client.is_connected = False
    assert await controller._send_chat(10, decision, controller._generation) == ""
    assert controller.client.attempts == []
    assert controller.budget.used == 0
