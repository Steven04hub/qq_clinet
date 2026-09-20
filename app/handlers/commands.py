"""Explicit group commands; none of these commands call the language model."""

from app.handlers.chat import GroupMessageSender
from app.onebot.messages import extract_plain_text, make_reply, split_text
from app.onebot.models import GroupMessageEvent
from app.services.session import SessionStore


class CommandHandler:
    def __init__(
        self,
        client: GroupMessageSender,
        *,
        model_name: str = "",
        max_reply_chars: int = 1500,
        session_store: SessionStore | None = None,
    ) -> None:
        self.client = client
        self.model_name = model_name
        self.max_reply_chars = max_reply_chars
        self.session_store = session_store

    async def handle(self, event: GroupMessageEvent) -> None:
        command = extract_plain_text(event.message, event.self_id)
        session_id = f"group:{event.group_id}"
        if command == "/clear" and self.session_store is not None:
            await self.session_store.clear(session_id)
        context_size = await self.session_store.size(session_id) if self.session_store else 0
        answers = {
            "/help": (
                "@机器人 echo hello：回复 hello\n"
                "/clear：清除当前群上下文（记忆暂未开启）\n"
                "/status：查看机器人状态"
            ),
            "/clear": "当前为 echo 测试模式，记忆暂未开启，无需清除上下文。",
            "/status": f"Bot online\nMode: echo\nContext: {context_size}/20",
        }
        if self.model_name:
            answers.update(
                {
                    "/help": (
                        "@机器人 + 问题：和 AI 对话\n"
                        "@机器人 echo hello：测试消息收发\n"
                        "/clear：清除当前群上下文\n/status：查看机器人状态\n"
                        "命令可直接发送，也可以 @机器人 后发送。"
                    ),
                    "/clear": "当前群上下文已清除。",
                    "/status": f"Bot online\nModel: {self.model_name}\nContext: {context_size}/20",
                }
            )
        answer = answers.get(command)
        if answer is not None:
            for chunk in split_text(answer, self.max_reply_chars):
                await self.client.send_group_message(
                    event.group_id, make_reply(event.message_id, chunk)
                )
