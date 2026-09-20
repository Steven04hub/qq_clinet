"""Select eligible commands or mentions before consuming abuse controls."""

from collections.abc import Iterable
from typing import Protocol

from app.onebot.messages import extract_plain_text, is_at_bot
from app.onebot.models import GroupMessageEvent
from app.services.dedup import MessageDeduplicator
from app.services.rate_limit import RateLimiter


class EventHandler(Protocol):
    async def handle(self, event: GroupMessageEvent) -> None: ...


class MessageRouter:
    def __init__(
        self,
        allowed_group_ids: Iterable[int],
        chat_handler: EventHandler,
        command_handler: EventHandler,
        *,
        deduplicator: MessageDeduplicator,
        rate_limiter: RateLimiter,
    ) -> None:
        self.allowed_group_ids = frozenset(allowed_group_ids)
        self.chat_handler = chat_handler
        self.command_handler = command_handler
        self.deduplicator = deduplicator
        self.rate_limiter = rate_limiter

    async def handle(self, event: GroupMessageEvent) -> None:
        if event.post_type != "message" or event.message_type != "group":
            return
        if event.user_id == event.self_id or event.group_id not in self.allowed_group_ids:
            return
        text = extract_plain_text(event.message, event.self_id)
        if not text:
            return
        handler: EventHandler
        if text in {"/help", "/clear", "/status"}:
            if any(
                segment.type == "at" and str(segment.data.get("qq")) != str(event.self_id)
                for segment in event.message
            ):
                return
            handler = self.command_handler
        elif is_at_bot(event.message, event.self_id):
            handler = self.chat_handler
        else:
            return
        if self.deduplicator.seen((event.group_id, event.message_id)):
            return
        if not self.rate_limiter.allow(event.user_id):
            return
        await handler.handle(event)
