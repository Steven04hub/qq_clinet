"""A same-origin, loopback-only control API and its bundled frontend."""

import base64
import logging
import shutil
import tempfile
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlsplit

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from starlette.background import BackgroundTask

from app.web.settings import ControlSettings

logger = logging.getLogger(__name__)


class BotController(Protocol):
    def status(self) -> dict[str, Any]: ...

    async def update(self, settings: ControlSettings) -> dict[str, Any]: ...

    async def clear_context(self) -> dict[str, Any]: ...

    async def list_models(self) -> dict[str, Any]: ...

    async def profiles(self) -> dict[str, Any]: ...

    async def refresh_profiles(self, group_id: str) -> dict[str, Any]: ...

    async def correct_alias(
        self, group_id: str, alias: str, user_ids: list[str]
    ) -> dict[str, Any]: ...

    async def search_memory(self, group_id: str, query: str) -> dict[str, Any]: ...

    async def list_stickers(self) -> dict[str, Any]: ...

    async def add_sticker(
        self, data: bytes, filename: str, label: str, tags: list[str]
    ) -> dict: ...

    async def sticker_image(self, sticker_id: str) -> tuple[bytes, str]: ...

    async def export_memory(self, destination: Path) -> Path: ...


class ProfileRefreshRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, hide_input_in_errors=True)
    group_id: str = Field(pattern=r"^[1-9][0-9]{0,19}$")


class AliasRequest(ProfileRefreshRequest):
    alias: str = Field(min_length=1, max_length=100)
    user_ids: list[str] = Field(min_length=1, max_length=10)


class MemorySearchRequest(ProfileRefreshRequest):
    query: str = Field(min_length=1, max_length=500)


class StickerUploadRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, hide_input_in_errors=True)
    filename: str = Field(default="", max_length=200)
    label: str = Field(min_length=1, max_length=80)
    tags: list[str] = Field(default_factory=list, max_length=12)
    data_base64: str = Field(min_length=1, max_length=11_184_812)


def _authority(value: str) -> tuple[str, int | None] | None:
    if not value or any(character.isspace() for character in value):
        return None
    try:
        parsed = urlsplit(f"//{value}")
        if (
            parsed.username is not None
            or parsed.password is not None
            or parsed.path
            or parsed.query
            or parsed.fragment
        ):
            return None
        if not parsed.hostname:
            return None
        return parsed.hostname.lower(), parsed.port
    except ValueError:
        return None


def _origin_matches(request: Request, authority: tuple[str, int | None]) -> bool:
    origin = request.headers.get("origin")
    if origin is None:
        # Non-browser clients may omit Origin but must still supply the custom header.
        return True
    try:
        parsed = urlsplit(origin)
        if parsed.path or parsed.query or parsed.fragment or parsed.scheme != request.url.scheme:
            return False
        source = _authority(parsed.netloc)
        if source is None:
            return False
        default_port = 443 if parsed.scheme == "https" else 80
        return (source[0], default_port if source[1] is None else source[1]) == (
            authority[0],
            default_port if authority[1] is None else authority[1],
        )
    except ValueError:
        return False


def _error(message: str, status: int) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status)


async def _run(operation: Callable[[], Awaitable[dict[str, Any]]]) -> dict[str, Any] | JSONResponse:
    try:
        return await operation()
    except ValueError:
        return _error("设置无法应用，请检查目标群和模型。", 400)
    except Exception as exc:
        logger.warning("Control request failed: %s", type(exc).__name__)
        return _error("操作未完成，请稍后重试。", 500)


def create_app(
    controller: BotController,
    *,
    lifespan: Any = None,
    allowed_hosts: Sequence[str] = ("127.0.0.1", "localhost", "::1"),
) -> FastAPI:
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    trusted_hosts = frozenset(host.lower() for host in allowed_hosts)
    static_path = Path(__file__).parent / "static"

    @app.middleware("http")
    async def local_requests(request: Request, call_next: Callable) -> Any:
        authority = _authority(request.headers.get("host", ""))
        length = request.headers.get("content-length", "0")
        if not length.isdecimal() or int(length) > 12_000_000:
            response = _error("请求过大，图片最大支持 8 MB。", 413)
        elif authority is None or authority[0] not in trusted_hosts:
            response = _error("仅允许通过本机地址访问控制界面。", 403)
        elif request.method not in {"GET", "HEAD"} and (
            request.headers.get("x-qq-control") != "1" or not _origin_matches(request, authority)
        ):
            response = _error("请求来源校验失败，请从控制界面重试。", 403)
        else:
            response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; "
            "base-uri 'none'; form-action 'self'"
        )
        return response

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_: Request, exc: RequestValidationError) -> JSONResponse:
        known_fields = ControlSettings.model_fields
        fields = sorted(
            {
                str(error["loc"][-1])
                for error in exc.errors()
                if error["loc"] and error["loc"][-1] in known_fields
            }
        )
        return JSONResponse(
            {"error": "设置格式有误，请检查填写内容。", "fields": fields}, status_code=422
        )

    @app.get("/api/status", response_model=None)
    async def status() -> dict[str, Any] | JSONResponse:
        try:
            return controller.status()
        except Exception as exc:
            logger.warning("Control status failed: %s", type(exc).__name__)
            return _error("暂时无法读取运行状态。", 500)

    @app.get("/api/models", response_model=None)
    async def models() -> dict[str, Any] | JSONResponse:
        return await _run(controller.list_models)

    @app.put("/api/settings", response_model=None)
    async def update(settings: ControlSettings) -> dict[str, Any] | JSONResponse:
        return await _run(lambda: controller.update(settings))

    @app.post("/api/context/clear", response_model=None)
    async def clear() -> dict[str, Any] | JSONResponse:
        return await _run(controller.clear_context)

    @app.get("/api/profiles", response_model=None)
    async def profiles() -> dict[str, Any] | JSONResponse:
        return await _run(controller.profiles)

    @app.post("/api/profiles/refresh", response_model=None)
    async def refresh_profiles(body: ProfileRefreshRequest) -> dict[str, Any] | JSONResponse:
        return await _run(lambda: controller.refresh_profiles(body.group_id))

    @app.put("/api/profiles/alias", response_model=None)
    async def correct_alias(body: AliasRequest) -> dict[str, Any] | JSONResponse:
        return await _run(
            lambda: controller.correct_alias(body.group_id, body.alias, body.user_ids)
        )

    @app.post("/api/memory/search", response_model=None)
    async def memory_search(body: MemorySearchRequest) -> dict | JSONResponse:
        return await _run(lambda: controller.search_memory(body.group_id, body.query))

    @app.get("/api/stickers", response_model=None)
    async def list_stickers() -> dict | JSONResponse:
        return await _run(controller.list_stickers)

    @app.post("/api/stickers", response_model=None)
    async def upload_sticker(body: StickerUploadRequest) -> dict | JSONResponse:
        try:
            data = base64.b64decode(body.data_base64, validate=True)
            return await controller.add_sticker(data, body.filename, body.label, body.tags)
        except ValueError:
            return _error("图片无效，请选择不超过 8 MB 的 PNG、JPEG、GIF 或 WebP。", 400)

    @app.get("/api/stickers/{sticker_id}/image", response_model=None)
    async def sticker_image(sticker_id: str) -> Response:
        try:
            data, mime = await controller.sticker_image(sticker_id)
            return Response(data, media_type=mime)
        except ValueError:
            return _error("素材不存在。", 404)

    @app.post("/api/memory/export", response_model=None)
    async def memory_export() -> Response:
        folder = Path(tempfile.mkdtemp(prefix="qq-memory-export-"))
        try:
            archive = await controller.export_memory(folder / "qq-memory.zip")
            return FileResponse(
                archive,
                media_type="application/zip",
                filename="qq-memory.zip",
                background=BackgroundTask(shutil.rmtree, folder, ignore_errors=True),
            )
        except Exception as exc:
            shutil.rmtree(folder, ignore_errors=True)
            logger.warning("Memory export failed error_type=%s", type(exc).__name__)
            return _error("迁移包导出失败，请检查本地数据。", 500)

    @app.get("/", response_class=FileResponse)
    async def index() -> FileResponse:
        return FileResponse(static_path / "index.html", media_type="text/html")

    # check_dir=False allows importing the API before frontend assets are generated.
    app.mount("/static", StaticFiles(directory=static_path, check_dir=False), name="static")
    return app
