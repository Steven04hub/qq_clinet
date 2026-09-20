"""Route only real bot mentions and exact, eligible commands."""

from unittest.mock import AsyncMock, Mock

import pytest

from app.handlers.router import MessageRouter
from app.onebot.models import GroupMessageEvent
from app.services.dedup import MessageDeduplicator
from app.services.rate_limit import RateLimiter


def make_event(*, text="echo hello", mention=True, **overrides):
    segments = []
    if mention:
        segments.append({"type": "at", "data": {"qq": "100"}})
    segments.append({"type": "text", "data": {"text": text}})
    payload = {
        "post_type": "message",
        "message_type": "group",
        "message_id": 1,
        "group_id": 10,
        "user_id": 200,
        "self_id": 100,
        "message": segments,
        "sender": {"user_id": 200, "nickname": "Test user"},
        **overrides,
    }
    return GroupMessageEvent.model_validate(payload)


@pytest.fixture
def routing():
    chat = AsyncMock()
    commands = AsyncMock()
    dedup = Mock(wraps=MessageDeduplicator())
    rate = Mock(wraps=RateLimiter(interval_seconds=5))
    router = MessageRouter([10, 20], chat, commands, deduplicator=dedup, rate_limiter=rate)
    return router, chat, commands, dedup, rate


@pytest.mark.parametrize(
    "event",
    [
        make_event(group_id=30),
        make_event(user_id=100),
        make_event(mention=False),
        make_event(mention=False, text="@机器人 echo hello"),
        make_event(mention=False, text="[CQ:at,qq=100] echo hello"),
        make_event(text="  "),
        make_event(message=[{"type": "at", "data": {"qq": "100"}}]),
        make_event(message=[{"type": "image", "data": {"file": "image.png"}}]),
        make_event().model_copy(update={"message_type": "private"}),
        make_event().model_copy(update={"post_type": "notice"}),
    ],
)
async def test_ineligible_event_does_not_route_or_consume_controls(routing, event):
    router, chat, commands, dedup, rate = routing
    await router.handle(event)
    chat.handle.assert_not_awaited()
    commands.handle.assert_not_awaited()
    dedup.seen.assert_not_called()
    rate.allow.assert_not_called()


async def test_real_bot_mention_routes_chat(routing):
    router, chat, commands, dedup, rate = routing
    event = make_event()
    await router.handle(event)
    chat.handle.assert_awaited_once_with(event)
    commands.handle.assert_not_awaited()
    dedup.seen.assert_called_once_with((10, 1))
    rate.allow.assert_called_once_with(200)


@pytest.mark.parametrize("command", ["/help", "/clear", "/status"])
@pytest.mark.parametrize("mention", [False, True])
async def test_exact_commands_route_with_or_without_bot_mention(routing, command, mention):
    router, chat, commands, _, _ = routing
    event = make_event(text=f" {command} ", mention=mention)
    await router.handle(event)
    commands.handle.assert_awaited_once_with(event)
    chat.handle.assert_not_awaited()


@pytest.mark.parametrize("other_target", ["300", "all", None])
@pytest.mark.parametrize("mention_bot", [False, True])
async def test_command_addressed_to_others_is_ignored(routing, other_target, mention_bot):
    router, chat, commands, dedup, rate = routing
    event = make_event(text="/clear", mention=mention_bot)
    segments = [segment.model_dump() for segment in event.message]
    segments.insert(0, {"type": "at", "data": {"qq": other_target}})
    event = make_event(message=segments)
    await router.handle(event)
    commands.handle.assert_not_awaited()
    chat.handle.assert_not_awaited()
    dedup.seen.assert_not_called()
    rate.allow.assert_not_called()


@pytest.mark.parametrize("text", ["/help extra", "/helpful", "/status now", "/CLEAR"])
async def test_command_prefixes_without_mention_are_not_commands(routing, text):
    router, chat, commands, dedup, rate = routing
    await router.handle(make_event(text=text, mention=False))
    commands.handle.assert_not_awaited()
    chat.handle.assert_not_awaited()
    dedup.seen.assert_not_called()
    rate.allow.assert_not_called()


async def test_other_user_mention_does_not_trigger_chat(routing):
    router, chat, commands, dedup, rate = routing
    await router.handle(
        make_event(
            message=[
                {"type": "at", "data": {"qq": "300"}},
                {"type": "text", "data": {"text": "echo hello"}},
            ]
        )
    )
    chat.handle.assert_not_awaited()
    commands.handle.assert_not_awaited()
    dedup.seen.assert_not_called()
    rate.allow.assert_not_called()


async def test_replayed_message_is_deduplicated_before_rate_limit(routing):
    router, chat, _, dedup, rate = routing
    event = make_event()
    await router.handle(event)
    await router.handle(event)
    chat.handle.assert_awaited_once_with(event)
    assert dedup.seen.call_count == 2
    rate.allow.assert_called_once_with(200)


async def test_message_ids_are_deduplicated_per_group(routing):
    router, chat, _, _, _ = routing
    first = make_event(group_id=10, user_id=200)
    second = make_event(group_id=20, user_id=300)
    await router.handle(first)
    await router.handle(second)
    assert chat.handle.await_count == 2


async def test_user_rate_limit_is_shared_across_groups_and_commands(routing):
    router, chat, commands, _, rate = routing
    await router.handle(make_event())
    await router.handle(make_event(message_id=2, group_id=20))
    await router.handle(make_event(message_id=3, text="/help", mention=False))
    assert chat.handle.await_count == 1
    commands.handle.assert_not_awaited()
    assert rate.allow.call_count == 3


async def test_ignored_message_does_not_suppress_next_valid_request(routing):
    router, chat, _, _, _ = routing
    await router.handle(make_event(mention=False))
    await router.handle(make_event())
    chat.handle.assert_awaited_once()


async def test_different_users_have_independent_rate_limits(routing):
    router, chat, _, _, _ = routing
    await router.handle(make_event(user_id=200, message_id=1))
    await router.handle(make_event(user_id=300, message_id=2))
    assert chat.handle.await_count == 2
