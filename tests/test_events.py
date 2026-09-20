"""Group event validation, filtering, and default logging privacy."""

import logging
from copy import deepcopy
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from app.onebot.events import EventDispatcher
from app.onebot.models import GroupMessageEvent, MessageSegment


@pytest.fixture
def group_payload() -> dict:
    return {
        "post_type": "message",
        "message_type": "group",
        "message_id": 11,
        "group_id": 100,
        "user_id": 200,
        "self_id": 300,
        "message": [
            {"type": "at", "data": {"qq": "300"}},
            {"type": "text", "data": {"text": "private question"}},
        ],
        "raw_message": "private raw message",
        "sender": {"user_id": 200, "nickname": "private nickname"},
    }


def test_group_event_models_important_fields(group_payload: dict) -> None:
    event = GroupMessageEvent.model_validate(group_payload)

    assert event.group_id == 100
    assert event.user_id == 200
    assert event.message[0] == MessageSegment(type="at", data={"qq": "300"})
    assert event.sender.card == ""
    assert event.sender.role == "member"


def test_group_event_ignores_napcat_extension_fields(group_payload: dict) -> None:
    group_payload["time"] = 12345678
    group_payload["extension"] = {"arbitrary": True}
    group_payload["sender"]["level"] = "99"
    group_payload["message"][0]["extension"] = "arbitrary"

    event = GroupMessageEvent.model_validate(group_payload)

    assert "time" not in event.model_dump()
    assert "extension" not in event.model_dump()
    assert "level" not in event.sender.model_dump()
    assert "extension" not in event.message[0].model_dump()


def test_unknown_segment_types_and_payloads_are_preserved(group_payload: dict) -> None:
    extension = {"type": "future_segment", "data": {"nested": {"value": [1, 2]}}}
    group_payload["message"].extend(
        [
            {"type": "reply", "data": {"id": "10"}},
            {"type": "image", "data": {"file": "image.jpg"}},
            {"type": "face", "data": {"id": "1"}},
            extension,
        ]
    )

    event = GroupMessageEvent.model_validate(group_payload)

    assert event.message[-1].model_dump() == extension
    assert len(event.message) == 6


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("post_type", "notice"),
        ("message_type", "private"),
        ("message_id", "invalid"),
        ("message", "CQ string instead of segment array"),
        ("message", [{"type": "text", "data": "invalid"}]),
        ("sender", {}),
    ],
)
def test_invalid_event_models_are_rejected(group_payload: dict, field: str, value) -> None:
    group_payload[field] = value

    with pytest.raises(ValidationError):
        GroupMessageEvent.model_validate(group_payload)


@pytest.mark.asyncio
async def test_eligible_event_logs_metadata_without_chat_content(
    group_payload: dict, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="app.onebot.events")

    await EventDispatcher([100]).handle_event(group_payload)

    assert len(caplog.records) == 1
    assert "group_id=100" in caplog.text
    assert "user_id=200" in caplog.text
    assert "message_id=11" in caplog.text
    assert "segment_count=2" in caplog.text
    assert "segment_types=at,text" in caplog.text
    assert "private question" not in caplog.text
    assert "private raw message" not in caplog.text
    assert "private nickname" not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "changes",
    [
        {"post_type": "meta_event"},
        {"post_type": "notice"},
        {"post_type": "message_sent"},
        {"message_type": "private"},
        {"user_id": 300},
        {"group_id": 999},
    ],
)
async def test_ineligible_events_are_completely_ignored(
    group_payload: dict, changes: dict, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG, logger="app.onebot.events")
    group_payload.update(changes)

    await EventDispatcher({100}).handle_event(group_payload)

    assert not caplog.records


@pytest.mark.asyncio
async def test_empty_allowlist_accepts_no_groups(
    group_payload: dict, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG, logger="app.onebot.events")

    await EventDispatcher([]).handle_event(group_payload)

    assert not caplog.records


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("allowed_group_ids", "changes"),
    [
        ([100], {"group_id": 999}),
        ([100], {"group_id": "999"}),
        ([100], {"group_id": "999.0"}),
        ([], {}),
        ([], {"group_id": "invalid", "user_id": "invalid", "self_id": "invalid"}),
        ([100], {"user_id": 300}),
        ([100], {"user_id": "300", "self_id": 300}),
        ([100], {"user_id": "300.0", "self_id": "300"}),
    ],
)
@pytest.mark.parametrize("field", ["message", "sender"])
async def test_ineligible_malformed_events_are_silent(
    group_payload: dict,
    allowed_group_ids: list[int],
    changes: dict,
    field: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG, logger="app.onebot.events")
    group_payload.update(changes)
    group_payload[field] = "private malformed content"

    await EventDispatcher(allowed_group_ids).handle_event(group_payload)

    assert not caplog.records


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["group_id", "user_id", "self_id", "message", "sender"])
async def test_eligible_malformed_events_warn_without_payload(
    group_payload: dict, field: str, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG, logger="app.onebot.events")
    group_payload[field] = "private malformed content"

    await EventDispatcher([100]).handle_event(group_payload)

    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.WARNING
    assert "validation_errors=1" in caplog.text
    assert "private" not in caplog.text


@pytest.mark.asyncio
async def test_eligible_numeric_string_ids_are_normalized(
    group_payload: dict, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="app.onebot.events")
    group_payload.update(group_id="100.0", user_id="200", self_id="300")

    await EventDispatcher([100]).handle_event(group_payload)

    assert len(caplog.records) == 1
    assert "group_id=100 user_id=200" in caplog.text


@pytest.mark.asyncio
async def test_malformed_event_is_logged_without_input_then_next_event_works(
    group_payload: dict, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="app.onebot.events")
    malformed = deepcopy(group_payload)
    malformed["message_id"] = "confidential malformed input"
    dispatcher = EventDispatcher([100])

    await dispatcher.handle_event(malformed)
    await dispatcher.handle_event(group_payload)

    assert [record.levelno for record in caplog.records] == [logging.WARNING, logging.INFO]
    assert "validation_errors=1" in caplog.text
    assert "confidential" not in caplog.text
    assert "private" not in caplog.text


@pytest.mark.asyncio
async def test_debug_logging_uses_normalized_event_only(
    group_payload: dict, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG, logger="app.onebot.events")
    group_payload["extension"] = "unmodeled extension data"

    await EventDispatcher([100]).handle_event(group_payload)

    assert "private question" in caplog.text
    assert "private raw message" in caplog.text
    assert "unmodeled extension data" not in caplog.text


@pytest.mark.asyncio
async def test_unknown_segment_does_not_stop_event_dispatch(
    group_payload: dict, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="app.onebot.events")
    group_payload["message"].append({"type": "future_segment", "data": {}})

    await EventDispatcher([100]).handle_event(group_payload)

    assert "segment_count=3" in caplog.text
    assert "segment_types=at,future_segment,text" in caplog.text


async def test_dispatcher_only_passes_valid_allowed_events_to_business_handler(group_payload):
    handler = AsyncMock()
    dispatcher = EventDispatcher([100], event_handler=handler)
    for override in (
        {"group_id": 999},
        {"user_id": 300},
        {"message_type": "private"},
        {"message": "not an array"},
    ):
        await dispatcher.handle_event({**group_payload, **override})
    handler.assert_not_called()
    await dispatcher.handle_event(group_payload)
    handler.assert_awaited_once_with(GroupMessageEvent.model_validate(group_payload))
