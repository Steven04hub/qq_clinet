"""Small in-memory group session store for shared group context."""

import asyncio
from collections import deque
from collections.abc import Mapping
from typing import Protocol


class SessionStore(Protocol):
    async def get(self, session_id: str) -> list[dict[str, str]]: ...
    async def append(self, session_id: str, message: Mapping[str, str]) -> None: ...
    async def clear(self, session_id: str) -> None: ...
    async def size(self, session_id: str) -> int: ...


class MemorySessionStore:
    def __init__(self, max_messages: int = 20) -> None:
        if isinstance(max_messages, bool) or not isinstance(max_messages, int) or max_messages < 2:
            raise ValueError("max_messages must be an integer of at least 2")
        self.max_messages = max_messages
        self._sessions: dict[str, deque[dict[str, str]]] = {}
        self._lock = asyncio.Lock()

    async def get(self, session_id: str) -> list[dict[str, str]]:
        async with self._lock:
            return [dict(message) for message in self._sessions.get(session_id, ())]

    async def append(self, session_id: str, message: Mapping[str, str]) -> None:
        role = message.get("role")
        content = message.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str):
            raise ValueError("session messages require a user/assistant role and text content")
        async with self._lock:
            entries = self._sessions.setdefault(session_id, deque(maxlen=self.max_messages))
            entries.append({"role": role, "content": content})

    async def clear(self, session_id: str) -> None:
        async with self._lock:
            self._sessions.pop(session_id, None)

    async def size(self, session_id: str) -> int:
        async with self._lock:
            return len(self._sessions.get(session_id, ()))
