"""OneBot 11 forward WebSocket client with one receive loop per connection."""

import asyncio
import json
import logging
import math
from collections.abc import Awaitable, Callable
from typing import Any

from websockets.asyncio.client import ClientConnection
from websockets.asyncio.client import connect as ws_connect
from websockets.exceptions import WebSocketException

from app.onebot.actions import (
    ActionManager,
    OneBotActionError,
    OneBotProtocolError,
    OneBotTimeoutError,
)

logger = logging.getLogger(__name__)
type EventHandler = Callable[[dict[str, Any]], Awaitable[None]]


class OneBotClient:
    def __init__(
        self,
        url: str,
        access_token: str,
        *,
        event_handler: EventHandler | None = None,
        connect_timeout: float = 10.0,
        action_timeout: float = 10.0,
    ) -> None:
        self.url = url
        self._access_token = access_token
        self._event_handler = event_handler
        self._connect_timeout = connect_timeout
        self._action_timeout = action_timeout
        self._ws: ClientConnection | None = None
        self._actions = ActionManager()
        self._receiver_task: asyncio.Task[None] | None = None
        self._connecting_task: asyncio.Task[ClientConnection] | None = None
        self._runner_task: asyncio.Task | None = None
        self._event_tasks: set[asyncio.Task[None]] = set()
        self._connect_lock = asyncio.Lock()
        self._close_lock = asyncio.Lock()
        self._connected = asyncio.Event()
        self._closed = asyncio.Event()

    @property
    def pending_requests(self) -> dict[str, asyncio.Future[dict[str, Any]]]:
        return self._actions.pending_requests

    @property
    def is_connected(self) -> bool:
        return self._connected.is_set() and not self._closed.is_set()

    async def _open_connection(self) -> ClientConnection:
        return await ws_connect(
            self.url,
            additional_headers={"Authorization": f"Bearer {self._access_token}"},
            proxy=None,
            open_timeout=self._connect_timeout,
            close_timeout=3,
            ping_interval=20,
            ping_timeout=20,
        )

    async def connect(self) -> None:
        """Connect once and start receiving; use run() for automatic reconnect."""
        async with self._connect_lock:
            if self._closed.is_set():
                raise ConnectionError("OneBotClient is closed")
            if self.is_connected:
                return
            self._connecting_task = asyncio.create_task(
                self._open_connection(), name="onebot-connect"
            )
            try:
                ws = await self._connecting_task
            finally:
                self._connecting_task = None
            if self._closed.is_set():
                await ws.close()
                raise ConnectionError("OneBotClient is closed")
            self._ws = ws
            self._connected.set()
            self._receiver_task = asyncio.create_task(
                self._receive_loop(ws), name="onebot-receiver"
            )
            logger.info("NapCat connected")

    async def run(self, event_handler: EventHandler | None = None) -> None:
        """Maintain the connection until close() or cancellation; never retry actions."""
        if self._runner_task is not None:
            raise RuntimeError("OneBotClient.run() is already active")
        if event_handler is not None:
            self._event_handler = event_handler
        self._runner_task = asyncio.current_task()
        delay = 1.0
        try:
            while not self._closed.is_set():
                try:
                    await self.connect()
                    delay = 1.0
                    assert self._receiver_task is not None
                    await self._receiver_task
                except asyncio.CancelledError:
                    if self._closed.is_set():
                        break
                    raise
                except (OSError, WebSocketException, TimeoutError) as exc:
                    # Some transport errors contain headers or URLs; log only their type.
                    logger.warning("NapCat connection failed error_type=%s", type(exc).__name__)
                if self._closed.is_set():
                    break
                logger.info("NapCat reconnecting delay_seconds=%s", delay)
                await self._wait_backoff(delay)
                delay = min(delay * 2, 15.0)
        finally:
            await self.close()
            self._runner_task = None

    async def _wait_backoff(self, delay: float) -> None:
        try:
            await asyncio.wait_for(self._closed.wait(), timeout=delay)
        except TimeoutError:
            pass

    async def wait_until_connected(self, timeout: float = 10.0) -> None:
        """Wait for run() to connect, also waking immediately if the client closes."""
        connected = asyncio.create_task(self._connected.wait())
        closed = asyncio.create_task(self._closed.wait())
        try:
            done, _ = await asyncio.wait(
                {connected, closed}, timeout=timeout, return_when=asyncio.FIRST_COMPLETED
            )
            if self._closed.is_set():
                raise ConnectionError("OneBotClient is closed")
            if not done:
                raise OneBotTimeoutError("Timed out waiting for NapCat connection")
        finally:
            for task in (connected, closed):
                task.cancel()
            await asyncio.gather(connected, closed, return_exceptions=True)

    async def _receive_loop(self, ws: ClientConnection) -> None:
        try:
            async for raw in ws:
                try:
                    payload = json.loads(raw)
                except (ValueError, TypeError):
                    logger.warning("Ignoring invalid OneBot JSON frame")
                    continue
                if not isinstance(payload, dict):
                    logger.warning("Ignoring non-object OneBot frame")
                    continue
                if self._actions.resolve(payload):
                    continue
                if "post_type" in payload:
                    self._schedule_event(payload)
                else:
                    logger.debug("Ignoring unmatched OneBot action response")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("NapCat receive failed error_type=%s", type(exc).__name__)
        finally:
            if self._ws is ws:
                self._ws = None
                self._connected.clear()
                if self._closed.is_set():
                    self._actions.cancel_all()
                else:
                    self._actions.fail_all()
                logger.info("NapCat disconnected")
            await ws.close()

    def _schedule_event(self, payload: dict[str, Any]) -> None:
        if self._event_handler is None or self._closed.is_set():
            return
        # Do not let an event flood allocate unbounded background tasks. Crucially,
        # waiting for a slot here would block action responses on the same socket.
        if len(self._event_tasks) >= 100:
            logger.warning("Dropping OneBot event: event handlers at capacity")
            return
        task = asyncio.create_task(self._dispatch_event(payload), name="onebot-event")
        self._event_tasks.add(task)
        task.add_done_callback(self._event_tasks.discard)

    async def _dispatch_event(self, payload: dict[str, Any]) -> None:
        try:
            assert self._event_handler is not None
            await self._event_handler(payload)
        except Exception as exc:
            logger.error("Event handler failed error_type=%s", type(exc).__name__)

    async def call_action(self, action: str, params: dict, timeout: float = 10.0) -> dict[str, Any]:
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Action timeout must be finite and positive")
        ws = self._ws
        if ws is None or not self.is_connected:
            raise ConnectionError("NapCat is not connected")
        echo, future = self._actions.create()
        try:
            request = json.dumps({"action": action, "params": params, "echo": echo})
            try:
                async with asyncio.timeout(timeout):
                    await ws.send(request)
                    response = await future
            except TimeoutError as exc:
                logger.warning("Action failed action=%s reason=timeout", action)
                raise OneBotTimeoutError(f"OneBot action timed out: {action}") from exc
            except (OSError, WebSocketException) as exc:
                logger.warning("Action failed action=%s reason=connection_lost", action)
                raise ConnectionError("NapCat connection lost while sending action") from exc
            if response.get("status") != "ok" or response.get("retcode") != 0:
                logger.warning("Action failed action=%s reason=server_response", action)
                raise OneBotActionError(action, response)
            return response
        finally:
            self._actions.discard(echo)
            if future.done() and not future.cancelled():
                # A send failure can race with disconnect failing this Future.
                future.exception()

    async def send_group_message(self, group_id: int, message: list[dict] | str) -> int:
        if isinstance(group_id, bool) or not isinstance(group_id, int) or group_id <= 0:
            raise ValueError("group_id must be a positive integer")
        if isinstance(message, str):
            message = [{"type": "text", "data": {"text": message}}]
        response = await self.call_action(
            "send_group_msg", {"group_id": group_id, "message": message}, self._action_timeout
        )
        data = response.get("data")
        message_id = data.get("message_id") if isinstance(data, dict) else None
        if not isinstance(message_id, int) or isinstance(message_id, bool):
            raise OneBotProtocolError("send_group_msg response has no integer message_id")
        return message_id

    async def close(self) -> None:
        """Stop reconnecting, close the socket and await all owned background tasks."""
        self._closed.set()
        self._connected.clear()
        self._actions.cancel_all()
        async with self._close_lock:
            current = asyncio.current_task()
            connecting = self._connecting_task
            if connecting is not None and connecting is not current:
                connecting.cancel()
                await asyncio.gather(connecting, return_exceptions=True)
            ws = self._ws
            if ws is not None:
                try:
                    await ws.close()
                except Exception as exc:
                    logger.warning("NapCat close failed error_type=%s", type(exc).__name__)
            tasks = set(self._event_tasks)
            if self._receiver_task is not None:
                tasks.add(self._receiver_task)
            tasks.discard(current)
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            self._event_tasks.difference_update(tasks)
            self._ws = None
