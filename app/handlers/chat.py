"""Serial group conversations and quoted replies through injected services."""

import asyncio
import logging
from typing import Protocol

from app.config import DEFAULT_SYSTEM_PROMPT
from app.onebot.messages import extract_plain_text, make_reply, split_text
from app.onebot.models import GroupMessageEvent
from app.services.llm import LLMError, LLMService
from app.services.session import SessionStore

logger = logging.getLogger(__name__)


class GroupMessageSender(Protocol):
    async def send_group_message(self, group_id: int, message: list[dict] | str) -> int: ...


class ChatHandler:
    def __init__(
        self,
        client: GroupMessageSender,
        *,
        max_reply_chars: int = 1500,
        llm: LLMService | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        group_locks: dict[int, asyncio.Lock] | None = None,
        max_input_chars: int = 4000,
        queue_timeout_seconds: float = 30,
        session_store: SessionStore | None = None,
        max_context_messages: int = 20,
    ) -> None:
        if max_reply_chars <= 0:
            raise ValueError("max_reply_chars must be positive")
        self.client = client
        self.max_reply_chars = max_reply_chars
        self.llm = llm
        self.system_prompt = system_prompt
        self.group_locks = group_locks if group_locks is not None else {}
        self.max_input_chars = max_input_chars
        self.queue_timeout_seconds = queue_timeout_seconds
        self.session_store = session_store
        self.max_context_messages = max_context_messages

    async def handle(self, event: GroupMessageEvent) -> None:
        text = extract_plain_text(event.message, event.self_id)
        if not text:
            return
        if len(text) > self.max_input_chars:
            await self._reply(event, f"消息太长了，请控制在 {self.max_input_chars} 字以内。")
            return
        lock = self.group_locks.setdefault(event.group_id, asyncio.Lock())
        try:
            await asyncio.wait_for(lock.acquire(), timeout=self.queue_timeout_seconds)
        except TimeoutError:
            await self._reply(event, "当前群的请求较多，请稍后再试。")
            return
        try:
            await self._handle_locked(event, text)
        finally:
            lock.release()

    async def _handle_locked(self, event: GroupMessageEvent, text: str) -> None:
        parts = text.split(maxsplit=1)
        if parts[0] == "echo" and len(parts) == 2:
            await self._reply(event, parts[1])
            return
        if self.llm is None:
            await self._reply(event, "当前为 echo 测试模式，请发送：@机器人 echo hello")
            return
        nickname = event.sender.card or event.sender.nickname or "群友"
        nickname = " ".join(nickname.split())[:80]
        user_message = {"role": "user", "content": f"[{nickname}/{event.user_id}] {text}"}
        session_id = f"group:{event.group_id}"
        history = await self.session_store.get(session_id) if self.session_store else []
        messages = [{"role": "system", "content": self.system_prompt}, *history, user_message]
        store_turn = False
        try:
            answer = await self.llm.chat(messages)
            if not isinstance(answer, str) or not answer.strip():
                raise LLMError("empty")
            store_turn = True
        except LLMError as exc:
            logger.warning("Chat model failed kind=%s", exc.kind)
            answer = {
                "timeout": "模型响应超时，请稍后再试。",
                "rate_limit": "模型繁忙，请稍后再试。",
            }.get(exc.kind, "模型暂时不可用，请稍后再试。")
        except Exception as exc:
            logger.error("Chat model failed error_type=%s", type(exc).__name__)
            answer = "模型暂时不可用，请稍后再试。"
        # A send failure propagates to the event supervisor. Do not retry, since
        # NapCat may already have delivered a reply whose acknowledgement was lost.
        if self.session_store is not None and store_turn:
            await self.session_store.append(session_id, user_message)
            await self.session_store.append(session_id, {"role": "assistant", "content": answer})
        await self._reply(event, answer)

    async def _reply(self, event: GroupMessageEvent, answer: str) -> None:
        for chunk in split_text(answer, self.max_reply_chars, max_chunks=3):
            await self.client.send_group_message(
                event.group_id, make_reply(event.message_id, chunk)
            )
