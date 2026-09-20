"""Start one local control server and its supervised bot connection."""

import logging
import socket
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn

from app.config import Settings
from app.services.control import BotController
from app.services.profile_store import GroupMemoryStore
from app.services.stickers import StickerLibrary
from app.web.server import create_app
from app.web.settings import ControlSettings, ControlSettingsStore

logger = logging.getLogger(__name__)


async def run_control(config: Settings, *, port: int, state_file: Path) -> int:
    # Bind before starting OneBot: an occupied port must not create a second bot.
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("127.0.0.1", port))
        sock.listen(128)
        sock.setblocking(False)
    except OSError:
        sock.close()
        logger.error("Control panel port unavailable port=%s", port)
        return 2
    try:
        # An ambiguous legacy allowlist requires choosing one target in the panel.
        target = str(config.allowed_group_ids[0]) if len(config.allowed_group_ids) == 1 else ""
        initial = ControlSettings(
            target_group_id=target,
            model=config.llm_model or "deepseek-flash",
            enabled=bool(target and config.llm_api_key.get_secret_value() and config.llm_base_url),
        )
        controller = BotController(
            config,
            ControlSettingsStore(state_file, initial),
            memory=GroupMemoryStore(state_file.parent / "memory" / "groups.sqlite3"),
            stickers=StickerLibrary(state_file.parent / "stickers"),
        )
    except ValueError:
        sock.close()
        logger.error("Cannot load control settings; check the local state file")
        return 2

    @asynccontextmanager
    async def lifespan(_app):
        await controller.start()
        try:
            yield
        finally:
            await controller.close()

    app = create_app(controller, lifespan=lifespan)
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host="127.0.0.1",
            port=port,
            log_config=None,
            access_log=False,
            server_header=False,
            timeout_graceful_shutdown=5,
        )
    )
    logger.info("Control panel: http://127.0.0.1:%s", port)
    try:
        await server.serve(sockets=[sock])
    finally:
        sock.close()
    return 0
