"""Queue-backed WebSocket doubles; these never open a network connection."""

import asyncio
import json
from typing import Any


class FakeWebSocket:
    """Emulate the small async WebSocket interface used by OneBotClient."""

    def __init__(self) -> None:
        self.incoming: asyncio.Queue[Any] = asyncio.Queue()
        self.sent: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.send_started = asyncio.Event()
        self.send_gate: asyncio.Event | None = None
        self.closed = False
        self.receivers = 0
        self.max_receivers = 0

    def __aiter__(self) -> "FakeWebSocket":
        return self

    async def __anext__(self) -> str | bytes:
        self.receivers += 1
        self.max_receivers = max(self.max_receivers, self.receivers)
        try:
            item = await self.incoming.get()
        finally:
            self.receivers -= 1
        if item is None:
            raise StopAsyncIteration
        if isinstance(item, BaseException):
            raise item
        return item

    async def send(self, data: str | bytes) -> None:
        self.send_started.set()
        if self.send_gate is not None:
            await self.send_gate.wait()
        if self.closed:
            raise ConnectionError("Fake socket is closed")
        await self.sent.put(json.loads(data))

    async def close(self) -> None:
        if not self.closed:
            self.closed = True
            self.incoming.put_nowait(None)

    def feed(self, payload: Any) -> None:
        self.incoming.put_nowait(json.dumps(payload))

    def feed_raw(self, payload: str | bytes) -> None:
        self.incoming.put_nowait(payload)

    def disconnect(self, error: Exception | None = None) -> None:
        self.incoming.put_nowait(error)

    async def next_request(self) -> dict[str, Any]:
        return await asyncio.wait_for(self.sent.get(), timeout=1)

    def respond(self, request: dict[str, Any], **overrides: Any) -> dict[str, Any]:
        payload = {
            "status": "ok",
            "retcode": 0,
            "data": {},
            "echo": request["echo"],
            **overrides,
        }
        self.feed(payload)
        return payload
