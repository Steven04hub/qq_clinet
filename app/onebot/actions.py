"""Action errors and echo-to-Future correlation; no socket reads here."""

import asyncio
from typing import Any
from uuid import uuid4


class OneBotTimeoutError(TimeoutError):
    """An action or connection did not complete within its deadline."""


class OneBotActionError(RuntimeError):
    """NapCat returned a non-success action response."""

    def __init__(self, action: str, response: dict[str, Any]) -> None:
        self.action = action
        self.status = response.get("status")
        self.retcode = response.get("retcode")
        # Do not include server wording, request params or response bodies in errors.
        super().__init__(f"OneBot action failed: {action}")


class OneBotProtocolError(RuntimeError):
    """A response is missing fields required by the requested action."""


class ActionManager:
    """Manage requests on a single connection (never replay them on reconnect)."""

    def __init__(self) -> None:
        self.pending_requests: dict[str, asyncio.Future[dict[str, Any]]] = {}

    def create(self) -> tuple[str, asyncio.Future[dict[str, Any]]]:
        echo = str(uuid4())
        future = asyncio.get_running_loop().create_future()
        self.pending_requests[echo] = future
        return echo, future

    def resolve(self, response: dict[str, Any]) -> bool:
        echo = response.get("echo")
        if not isinstance(echo, str):
            return False
        future = self.pending_requests.pop(echo, None)
        if future is None:
            return False
        if not future.done():
            future.set_result(response)
        return True

    def discard(self, echo: str) -> None:
        future = self.pending_requests.pop(echo, None)
        if future is not None and not future.done():
            future.cancel()

    def fail_all(self) -> None:
        for future in self.pending_requests.values():
            if not future.done():
                future.set_exception(ConnectionError("NapCat connection lost"))
        self.pending_requests.clear()

    def cancel_all(self) -> None:
        for future in self.pending_requests.values():
            if not future.done():
                future.cancel()
        self.pending_requests.clear()
