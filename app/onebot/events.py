"""Validate eligible group events before optionally routing them to the bot."""

import logging
from collections.abc import Awaitable, Callable

from pydantic import TypeAdapter, ValidationError

from app.onebot.models import GroupMessageEvent

logger = logging.getLogger(__name__)
_id_adapter = TypeAdapter(int)


class EventDispatcher:
    def __init__(
        self,
        allowed_group_ids: list[int] | set[int],
        event_handler: Callable[[GroupMessageEvent], Awaitable[None]] | None = None,
    ) -> None:
        self.allowed_group_ids = frozenset(allowed_group_ids)
        self._event_handler = event_handler

    async def handle_event(self, payload: dict) -> None:
        if not self.allowed_group_ids:
            return
        if payload.get("post_type") != "message" or payload.get("message_type") != "group":
            return

        # Filter using envelope IDs before parsing content. Match the event model's
        # integer coercion so numeric-string IDs receive the same filtering.
        envelope_ids = {}
        for name in ("group_id", "user_id", "self_id"):
            try:
                envelope_ids[name] = _id_adapter.validate_python(payload.get(name))
            except ValidationError:
                # Let full validation report malformed eligible events safely.
                pass
        group_id = envelope_ids.get("group_id")
        if group_id is not None and group_id not in self.allowed_group_ids:
            return
        user_id = envelope_ids.get("user_id")
        if user_id is not None and user_id == envelope_ids.get("self_id"):
            return

        try:
            event = GroupMessageEvent.model_validate(payload)
        except ValidationError as exc:
            # ValidationError's text includes the rejected input, which may be chat
            # content or NapCat extension data. Log only the number of failures.
            logger.warning(
                "Ignoring invalid group message event: validation_errors=%s",
                exc.error_count(),
            )
            return

        if event.user_id == event.self_id:
            return
        if event.group_id not in self.allowed_group_ids:
            return

        logger.info(
            "Group message event group_id=%s user_id=%s message_id=%s "
            "segment_count=%s segment_types=%s",
            event.group_id,
            event.user_id,
            event.message_id,
            len(event.message),
            ",".join(sorted({segment.type for segment in event.message})),
        )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Group message event payload=%s", event.model_dump())
        if self._event_handler is not None:
            await self._event_handler(event)
