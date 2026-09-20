"""Compose a group bot, a read-only listener, or a one-shot send smoke test."""

import argparse
import asyncio
import logging
import signal
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from app.config import get_settings
from app.handlers.chat import ChatHandler
from app.handlers.commands import CommandHandler
from app.handlers.router import MessageRouter
from app.onebot.client import OneBotClient
from app.onebot.events import EventDispatcher
from app.services.dedup import MessageDeduplicator
from app.services.llm import LLMError, OpenAICompatibleLLM
from app.services.rate_limit import RateLimiter
from app.services.session import MemorySessionStore
from app.utils.logging import configure_logging

logger = logging.getLogger(__name__)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QQ group bot with NapCat / OneBot 11")
    subparsers = parser.add_subparsers(dest="command")
    parser.set_defaults(command="run")
    subparsers.add_parser("run", help="Run the AI group chat bot (default)")
    web = subparsers.add_parser("web", help="Run the local control panel and single-group bot")
    web.add_argument("--port", type=int, default=8765)
    web.add_argument("--state-file", type=Path, default=Path(".state/control.json"))
    export = subparsers.add_parser(
        "memory-export", help="Export portable local memory without credentials"
    )
    export.add_argument("--state-file", type=Path, default=Path(".state/control.json"))
    export.add_argument("--output", type=Path, required=True)
    restore = subparsers.add_parser(
        "memory-import", help="Restore memory into a new state directory"
    )
    restore.add_argument("--archive", type=Path, required=True)
    restore.add_argument("--state-dir", type=Path, default=Path(".state"))
    subparsers.add_parser("check-model", help="Check AI credentials with one short model request")
    subparsers.add_parser("listen", help="Log eligible group events without auto-replies")
    subparsers.add_parser("echo", help="Reply to @bot echo messages without calling an AI model")
    send = subparsers.add_parser("send", help="Send exactly one test message to an allowed group")
    send.add_argument("--group-id", type=int, required=True)
    send.add_argument("--message", required=True, help="Literal text; CQ codes are not parsed")
    send.add_argument("--reply-to", type=int, help="Optional source message_id from listener logs")
    args = parser.parse_args(argv)
    if args.command == "web" and not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.command == "send":
        if args.group_id <= 0:
            parser.error("--group-id must be positive")
        if not args.message.strip():
            parser.error("--message must not be empty")
    return args


async def send_once(client: OneBotClient, args: argparse.Namespace, timeout: float) -> None:
    await client.wait_until_connected(timeout)
    message: list[dict] = [{"type": "text", "data": {"text": args.message}}]
    if args.reply_to is not None:
        message.insert(0, {"type": "reply", "data": {"id": str(args.reply_to)}})
    message_id = await client.send_group_message(args.group_id, message)
    logger.info("Sent group message group_id=%s message_id=%s", args.group_id, message_id)


async def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging()
    if args.command in {"memory-export", "memory-import"}:
        from app.services.portable import export_memory, restore_memory

        try:
            if args.command == "memory-export":
                path = await asyncio.to_thread(export_memory, args.state_file, args.output)
            else:
                path = await asyncio.to_thread(restore_memory, args.archive, args.state_dir)
            logger.info("Memory migration completed: %s", path)
            return 0
        except (ValueError, OSError) as exc:
            logger.error("Memory migration failed: %s", str(exc))
            return 2
    try:
        settings = get_settings()
    except ValidationError as exc:
        # Even validation errors may contain raw secrets; print only field locations.
        fields = ", ".join(".".join(map(str, e["loc"])) for e in exc.errors())
        logger.error("Configuration invalid; check .env fields: %s", fields)
        return 2
    configure_logging(
        settings.log_level,
        secrets=[
            settings.napcat_access_token.get_secret_value(),
            settings.llm_api_key.get_secret_value(),
            settings.typesafe_api_key.get_secret_value(),
        ],
    )
    if args.command == "web":
        from app.web.runner import run_control

        return await run_control(settings, port=args.port, state_file=args.state_file)
    if args.command == "send" and args.group_id not in settings.allowed_group_ids:
        logger.error("Refusing send: group_id is not in ALLOWED_GROUP_IDS")
        return 2
    if not settings.allowed_group_ids:
        logger.warning("ALLOWED_GROUP_IDS is empty: all group events will be ignored")

    llm = None
    if args.command in {"run", "check-model"}:
        try:
            settings.validate_llm_config()
        except ValueError as exc:
            logger.error("AI configuration invalid: %s", exc)
            return 2
        llm = OpenAICompatibleLLM(
            settings.llm_api_key.get_secret_value(),
            settings.llm_base_url,
            settings.llm_model,
            timeout_seconds=settings.llm_timeout_seconds,
            max_concurrency=settings.max_llm_concurrency,
            max_tokens=settings.llm_max_tokens,
            thinking=settings.llm_thinking,
        )
    if args.command == "check-model":
        assert llm is not None
        try:
            answer = await llm.chat([{"role": "user", "content": "请仅回复：连接成功"}])
            logger.info(
                "Model check succeeded model=%s response_chars=%s", settings.llm_model, len(answer)
            )
            return 0
        except LLMError as exc:
            logger.error("Model check failed kind=%s status_code=%s", exc.kind, exc.status_code)
            return 1
        finally:
            await llm.close()

    client = OneBotClient(
        settings.napcat_ws_url,
        settings.napcat_access_token.get_secret_value(),
        connect_timeout=settings.connect_timeout_seconds,
        action_timeout=settings.action_timeout_seconds,
    )
    router = None
    session_store = MemorySessionStore(settings.max_context_messages) if llm else None
    if args.command in {"run", "echo"}:
        group_locks = {group_id: asyncio.Lock() for group_id in settings.allowed_group_ids}
        router = MessageRouter(
            settings.allowed_group_ids,
            ChatHandler(
                client,
                llm=llm,
                system_prompt=settings.system_prompt,
                group_locks=group_locks,
                max_reply_chars=settings.max_reply_chars,
                max_input_chars=settings.max_input_chars,
                queue_timeout_seconds=settings.group_queue_timeout_seconds,
                session_store=session_store,
                max_context_messages=settings.max_context_messages,
            ),
            CommandHandler(
                client,
                model_name=settings.llm_model if llm else "",
                max_reply_chars=settings.max_reply_chars,
                session_store=session_store,
            ),
            deduplicator=MessageDeduplicator(),
            rate_limiter=RateLimiter(settings.user_rate_limit_seconds),
        )
        logger.info("Bot mode=%s allowed_groups=%s", args.command, len(settings.allowed_group_ids))
    dispatcher = EventDispatcher(
        settings.allowed_group_ids, event_handler=router.handle if router else None
    )
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    installed_signals = []
    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signum, stop.set)
            installed_signals.append(signum)
        except (NotImplementedError, RuntimeError):
            # asyncio.run handles Ctrl+C on platforms without add_signal_handler.
            pass

    runner = asyncio.create_task(client.run(dispatcher.handle_event), name="onebot-runner")
    stopper = asyncio.create_task(stop.wait(), name="onebot-stop")
    tasks = {runner, stopper}
    sender = None
    if args.command == "send":
        sender = asyncio.create_task(
            send_once(client, args, settings.connect_timeout_seconds), name="onebot-send-once"
        )
        tasks.add(sender)
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        if sender in done:
            await sender
        if runner in done:
            await runner
        return 0
    except Exception as exc:
        logger.error("Bot operation failed error_type=%s", type(exc).__name__)
        return 1
    finally:
        await client.close()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if llm is not None:
            await llm.close()
        for signum in installed_signals:
            loop.remove_signal_handler(signum)
        logger.info("Bot stopped")


def cli() -> None:
    try:
        result = asyncio.run(main())
    except KeyboardInterrupt:
        result = 0
    raise SystemExit(result)


if __name__ == "__main__":
    cli()
