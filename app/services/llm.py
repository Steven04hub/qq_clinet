"""Bounded OpenAI-compatible chat calls with safe errors and metadata-only logs."""

import asyncio
import json
import logging
import math
import time
from typing import Protocol

from openai import (
    APIConnectionError,
    APIResponseValidationError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
)

logger = logging.getLogger(__name__)

_ERROR_MESSAGES = {
    "timeout": "Model request timed out.",
    "auth": "Model authentication failed.",
    "balance": "Model account has insufficient balance.",
    "rate_limit": "Model service rate limit reached.",
    "api": "Model service returned an HTTP error.",
    "connection": "Model service connection failed.",
    "empty": "Model service returned no text.",
    "malformed": "Model service returned an invalid response.",
    "unknown": "Model request failed.",
}

TRUNCATION_NOTICE = "\n\n（回复达到长度上限，已截断。）"


class LLMService(Protocol):
    async def chat(self, messages: list[dict[str, str]]) -> str: ...


class LLMError(Exception):
    """A safe public error that never incorporates a provider error body."""

    def __init__(self, kind: str, status_code: int | None = None) -> None:
        self.kind = kind if kind in _ERROR_MESSAGES else "unknown"
        self.status_code = status_code
        super().__init__(_ERROR_MESSAGES[self.kind])


class OpenAICompatibleLLM:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        *,
        timeout_seconds: float = 60,
        max_concurrency: int = 3,
        max_tokens: int = 1024,
        thinking: str = "",
        client: AsyncOpenAI | None = None,
    ) -> None:
        if (
            isinstance(timeout_seconds, bool)
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be finite and positive")
        for name, value in (("max_concurrency", max_concurrency), ("max_tokens", max_tokens)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if thinking not in {"", "enabled", "disabled"}:
            raise ValueError("thinking must be empty, enabled, or disabled")

        self._client = (
            AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout_seconds,
                max_retries=0,
            )
            if client is None
            else client.with_options(max_retries=0)
        )
        self._model = model
        self._timeout_seconds = timeout_seconds
        self._max_tokens = max_tokens
        self._thinking = thinking
        # Share one service across groups so this bounds all model requests.
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._models_lock = asyncio.Lock()
        self._models_cache: tuple[float, list[str]] | None = None

    async def list_models(self) -> list[str]:
        """Read provider model IDs, without exposing credentials or provider errors."""
        async with self._models_lock:
            if self._models_cache and time.monotonic() - self._models_cache[0] < 60:
                return list(self._models_cache[1])
            try:
                async with asyncio.timeout(min(self._timeout_seconds, 10)):
                    page = await self._client.models.list()
                models = sorted(
                    {
                        item.id
                        for item in page.data[:200]
                        if isinstance(item.id, str)
                        and 0 < len(item.id) <= 150
                        and all(character.isprintable() for character in item.id)
                    }
                )
                self._models_cache = (time.monotonic(), models)
                return list(models)
            except Exception as exc:
                raise self._safe_error(exc) from None

    async def chat(self, messages: list[dict[str, str]]) -> str:
        return await self._chat(messages, json_mode=False)

    async def chat_json(self, messages: list[dict[str, str]]) -> str:
        """Request a JSON object for bot decisions; still validate its schema downstream."""
        return await self._chat(messages, json_mode=True)

    async def _chat(self, messages: list[dict[str, str]], *, json_mode: bool) -> str:
        started = time.monotonic()
        kind = "unknown"
        status_code = None
        try:
            # Include queueing and response-body reads in the same hard deadline.
            # Provider keep-alives must not extend a request indefinitely.
            async with asyncio.timeout(self._timeout_seconds):
                async with self._semaphore:
                    options = {}
                    if json_mode:
                        options["response_format"] = {"type": "json_object"}
                    if self._thinking:
                        options["extra_body"] = {"thinking": {"type": self._thinking}}
                    completion = await self._client.chat.completions.create(
                        model=self._model,
                        messages=messages,
                        stream=False,
                        max_tokens=self._max_tokens,
                        **options,
                    )
                    reply = self._extract_reply(completion)
            kind = "ok"
            return reply
        except asyncio.CancelledError:
            kind = "cancelled"
            raise
        except Exception as exc:
            error = self._safe_error(exc)
            kind, status_code = error.kind, error.status_code
            # Neither provider messages nor their chained traceback escape here.
            raise error from None
        finally:
            logger.info(
                "LLM request kind=%s status_code=%s elapsed_ms=%d",
                kind,
                status_code,
                round((time.monotonic() - started) * 1000),
            )

    @staticmethod
    def _extract_reply(completion: object) -> str:
        choices = getattr(completion, "choices", None)
        if not isinstance(choices, list) or not choices:
            raise LLMError("malformed")
        message = getattr(choices[0], "message", None)
        if message is None or not hasattr(message, "content"):
            raise LLMError("malformed")
        fields_set = getattr(message, "model_fields_set", set())
        if "content" not in fields_set:
            raise LLMError("malformed")
        content = message.content
        if content is None:
            raise LLMError("empty")
        if not isinstance(content, str):
            raise LLMError("malformed")
        if not content.strip():
            raise LLMError("empty")
        reply = content.strip()
        if getattr(choices[0], "finish_reason", None) == "length":
            reply += TRUNCATION_NOTICE
        return reply

    @staticmethod
    def _safe_error(exc: Exception) -> LLMError:
        if isinstance(exc, LLMError):
            return exc
        if isinstance(exc, (APITimeoutError, TimeoutError)):
            return LLMError("timeout")
        if isinstance(exc, APIStatusError):
            kind = {401: "auth", 402: "balance", 429: "rate_limit"}.get(exc.status_code, "api")
            return LLMError(kind, exc.status_code)
        if isinstance(exc, APIConnectionError):
            return LLMError("connection")
        if isinstance(exc, (APIResponseValidationError, json.JSONDecodeError)):
            return LLMError("malformed")
        return LLMError("unknown")

    async def close(self) -> None:
        await self._client.close()
