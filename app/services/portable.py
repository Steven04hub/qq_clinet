"""Versioned, credential-free ZIP exports of portable bot memory and stickers."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import stat
import tempfile
import zipfile
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

from app.services.profile_store import SCHEMA_VERSION, GroupMemoryStore
from app.services.stickers import MAX_STICKER_BYTES, StickerLibrary
from app.web.settings import ControlSettings

FORMAT = "qq-ai-bot-memory"
MANIFEST_VERSION = 1
MAX_FILES = 2048
MAX_TOTAL_BYTES = 512 * 1024 * 1024
MAX_DATABASE_BYTES = 256 * 1024 * 1024
MAX_JSON_BYTES = 1024 * 1024
MAX_ARCHIVE_BYTES = MAX_TOTAL_BYTES + 16 * 1024 * 1024
_ASSET = re.compile(r"stickers/[0-9a-f]{64}\.(?:png|jpg|gif|webp)")
_REQUIRED_TABLES = {
    "members": {
        "group_id",
        "user_id",
        "display_name",
        "names",
        "profile",
        "message_count",
        "emoji_habits",
        "last_seen",
    },
    "messages": {"sequence", "group_id", "message_id", "time", "payload"},
    "seen_messages": {"group_id", "message_id"},
    "aliases": {"group_id", "alias", "source", "payload"},
    "memes": {"group_id", "name", "payload"},
    "memory_metadata": {"key", "value"},
    "memory_documents": {"sequence", "group_id", "user_id", "is_bot", "tokens"},
}


def _json(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _regular_file(path: Path, limit: int) -> None:
    try:
        info = path.lstat()
    except OSError:
        raise ValueError("迁移文件缺失或无法读取。") from None
    if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
        raise ValueError("迁移文件必须是大小合规的普通文件，不能是符号链接。")


def _directory(path: Path) -> None:
    if path.is_symlink() or not path.is_dir():
        raise ValueError("迁移目录必须是普通目录，不能是符号链接。")


def _read_json(path: Path) -> object:
    _regular_file(path, MAX_JSON_BYTES)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        raise ValueError("迁移文件中的 JSON 格式无效。") from None


def _paused_control(value: object) -> bytes:
    try:
        settings = ControlSettings.model_validate(value)
    except (ValueError, TypeError):
        raise ValueError("迁移包中的控制设置无效或包含未知字段。") from None
    return _json(settings.model_copy(update={"enabled": False}).model_dump())


def _entry_limit(name: str) -> int:
    if name in {"manifest.json", "control.json", "stickers/stickers.json"}:
        return MAX_JSON_BYTES
    if name == "memory/groups.sqlite3":
        return MAX_DATABASE_BYTES
    if _ASSET.fullmatch(name):
        return MAX_STICKER_BYTES
    raise ValueError("迁移包包含不允许的文件路径。")


def _validate_database(path: Path, expected_version: int) -> None:
    _regular_file(path, MAX_DATABASE_BYTES)
    try:
        with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as db:
            db.execute("PRAGMA query_only=ON")
            db.execute("PRAGMA trusted_schema=OFF")
            if db.execute("PRAGMA user_version").fetchone()[0] != expected_version:
                raise ValueError("数据库版本与迁移包不一致。")
            if db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise ValueError("迁移数据库完整性校验失败。")
            objects = dict(db.execute("SELECT name,type FROM sqlite_master"))
            for table, columns in _REQUIRED_TABLES.items():
                if objects.get(table) != "table":
                    raise ValueError("迁移数据库缺少必要的数据表。")
                actual = {row[1] for row in db.execute(f'PRAGMA table_info("{table}")')}
                if not columns <= actual:
                    raise ValueError("迁移数据库的数据表格式无效。")
    except sqlite3.Error:
        raise ValueError("迁移数据库无法读取或完整性校验失败。") from None


def _sticker_index(path: Path) -> tuple[bytes, dict[str, str]]:
    """Validate the portable metadata whitelist and map assets to content hashes."""
    if not path.exists() and not path.is_symlink():
        return _json([]), {}
    _directory(path)
    # A partially uploaded temporary regular file is ignored; a symbolic link is
    # never followed, including links which are not listed in the library index.
    for child in path.rglob("*"):
        if child.is_symlink():
            raise ValueError("表情素材目录不能包含符号链接。")
    index = path / "stickers.json"
    records = _read_json(index) if index.exists() or index.is_symlink() else []
    if not isinstance(records, list) or len(records) > MAX_FILES - 4:
        raise ValueError("表情素材索引无效或数量过多。")
    expected = {}
    allowed = {"id", "label", "tags", "extension", "mime_type", "size"}
    for item in records:
        if (
            not isinstance(item, dict)
            or set(item) - allowed
            or not isinstance(item.get("id"), str)
            or not re.fullmatch(r"img_[0-9a-f]{64}", item["id"])
            or not isinstance(item.get("extension"), str)
            or item.get("extension") not in {"png", "jpg", "gif", "webp"}
            or not isinstance(item.get("label"), str)
            or len(item["label"]) > 80
            or not isinstance(item.get("tags"), list)
            or len(item["tags"]) > 8
            or any(not isinstance(tag, str) or len(tag) > 30 for tag in item["tags"])
            or type(item.get("size")) is not int
            or not 0 < item["size"] <= MAX_STICKER_BYTES
        ):
            raise ValueError("表情素材索引格式无效。")
        digest = item["id"][4:]
        name = f"stickers/{digest}.{item['extension']}"
        if name in expected:
            raise ValueError("表情素材索引含有重复项目。")
        expected[name] = digest
    if records:
        library = StickerLibrary(path)
        for item in records:
            data, mime = library.read(item["id"])
            if len(data) != item["size"] or item.get("mime_type", mime) != mime:
                raise ValueError("表情素材内容与索引不一致。")
    return _json(records), expected


def export_memory(state_file: Path, destination: Path) -> Path:
    """Write a new migration ZIP; neither the original controls nor secrets change."""
    state_file = Path(state_file).absolute()
    destination = Path(destination).absolute()
    root = state_file.parent
    _directory(root)
    control = _paused_control(_read_json(state_file))
    memory_dir = root / "memory"
    _directory(memory_dir)
    database = memory_dir / "groups.sqlite3"
    _regular_file(database, MAX_DATABASE_BYTES)
    sticker_index, assets = _sticker_index(root / "stickers")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("迁移包目标文件已经存在，请换一个文件名。")
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    _directory(destination.parent)
    temporary: Path | None = None
    try:
        with tempfile.TemporaryDirectory(
            prefix=".qq-memory-export-", dir=destination.parent
        ) as work:
            work_path = Path(work)
            snapshot = work_path / "groups.sqlite3"
            memory = GroupMemoryStore(database)
            memory.backup_to(snapshot)
            _validate_database(snapshot, memory.schema_version)
            sources: dict[str, Path | bytes] = {
                "control.json": control,
                "memory/groups.sqlite3": snapshot,
                "stickers/stickers.json": sticker_index,
            }
            sources.update({name: root / name for name in sorted(assets)})
            with tempfile.NamedTemporaryFile(
                prefix=".qq-memory-", suffix=".zip", dir=destination.parent, delete=False
            ) as handle:
                temporary = Path(handle.name)
                os.fchmod(handle.fileno(), 0o600)
            manifest = {
                "format": FORMAT,
                "schema_version": MANIFEST_VERSION,
                "database_schema_version": memory.schema_version,
                "created_at": datetime.now(UTC).isoformat(),
                "files": {},
            }
            total = 0
            with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for name, source in sources.items():
                    limit = _entry_limit(name)
                    size = len(source) if isinstance(source, bytes) else source.stat().st_size
                    total += size
                    if size > limit or total > MAX_TOTAL_BYTES:
                        raise ValueError("迁移数据超过允许的大小上限。")
                    digest = hashlib.sha256()
                    if isinstance(source, bytes):
                        archive.writestr(name, source)
                        digest.update(source)
                    else:
                        _regular_file(source, limit)
                        copied = 0
                        with source.open("rb") as reader, archive.open(name, "w") as writer:
                            while chunk := reader.read(1024 * 1024):
                                copied += len(chunk)
                                if copied > size:
                                    raise ValueError("迁移过程中源文件发生变化，请重试。")
                                digest.update(chunk)
                                writer.write(chunk)
                        if copied != size:
                            raise ValueError("迁移过程中源文件发生变化，请重试。")
                    hexdigest = digest.hexdigest()
                    if name in assets and assets[name] != hexdigest:
                        raise ValueError("迁移过程中表情素材发生变化，请重试。")
                    manifest["files"][name] = {"size": size, "sha256": hexdigest}
                manifest_bytes = _json(manifest)
                if (
                    len(manifest_bytes) > MAX_JSON_BYTES
                    or total + len(manifest_bytes) > MAX_TOTAL_BYTES
                ):
                    raise ValueError("迁移数据超过允许的大小上限。")
                archive.writestr("manifest.json", manifest_bytes)
            # A hard link publishes the completed archive atomically and, unlike
            # replace(), cannot overwrite a destination created concurrently.
            os.link(temporary, destination)
            os.chmod(destination, 0o600)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def _manifest(archive: zipfile.ZipFile) -> tuple[dict, dict[str, zipfile.ZipInfo]]:
    infos = archive.infolist()
    if not infos or len(infos) > MAX_FILES:
        raise ValueError("迁移包条目数量无效或超过上限。")
    entries = {}
    total = 0
    for info in infos:
        name = info.filename
        if (
            name in entries
            or name != info.orig_filename
            or "\\" in name
            or "\x00" in name
            or info.is_dir()
            or info.flag_bits & 1
            or info.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
        ):
            raise ValueError("迁移包包含重复、加密或非法路径条目。")
        mode = info.external_attr >> 16
        if stat.S_IFMT(mode) not in {0, stat.S_IFREG}:
            raise ValueError("迁移包不能包含符号链接或特殊文件。")
        limit = _entry_limit(name)
        total += info.file_size
        if info.file_size > limit or total > MAX_TOTAL_BYTES:
            raise ValueError("迁移包解压大小超过上限。")
        entries[name] = info
    if "manifest.json" not in entries:
        raise ValueError("迁移包缺少 manifest.json。")
    try:
        manifest = json.loads(archive.read(entries["manifest.json"]))
    except (ValueError, UnicodeError):
        raise ValueError("迁移包清单格式无效。") from None
    if (
        not isinstance(manifest, dict)
        or manifest.get("format") != FORMAT
        or type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] != MANIFEST_VERSION
        or type(manifest.get("database_schema_version")) is not int
        or manifest["database_schema_version"] != SCHEMA_VERSION
        or not isinstance(manifest.get("files"), dict)
    ):
        raise ValueError("迁移包类型或版本不受支持。")
    files = manifest["files"]
    if set(files) != set(entries) - {"manifest.json"} or not {
        "control.json",
        "memory/groups.sqlite3",
        "stickers/stickers.json",
    } <= set(files):
        raise ValueError("迁移包文件与清单不一致。")
    for name, item in files.items():
        if (
            not isinstance(item, dict)
            or set(item) != {"size", "sha256"}
            or type(item.get("size")) is not int
            or item["size"] != entries[name].file_size
            or not isinstance(item.get("sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"])
        ):
            raise ValueError("迁移包文件校验信息无效。")
    return manifest, entries


def restore_memory(archive: Path, state_dir: Path) -> Path:
    """Validate in a private staging directory, then publish a new paused state."""
    archive = Path(archive).absolute()
    state_dir = Path(state_dir).absolute()
    _regular_file(archive, MAX_ARCHIVE_BYTES)
    if state_dir.exists() or state_dir.is_symlink():
        raise FileExistsError("还原目录已经存在；请指定一个全新的目录。")
    state_dir.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    _directory(state_dir.parent)
    reserved = False
    try:
        with tempfile.TemporaryDirectory(
            prefix=".qq-memory-restore-", dir=state_dir.parent
        ) as work:
            staged = Path(work) / "state"
            staged.mkdir(mode=0o700)
            with zipfile.ZipFile(archive, "r") as package:
                manifest, entries = _manifest(package)
                for name, item in manifest["files"].items():
                    target = staged / name
                    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                    digest = hashlib.sha256()
                    copied = 0
                    descriptor = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                    with (
                        os.fdopen(descriptor, "wb") as writer,
                        package.open(entries[name]) as reader,
                    ):
                        while chunk := reader.read(1024 * 1024):
                            copied += len(chunk)
                            if copied > item["size"]:
                                raise ValueError("迁移包条目的实际解压大小与清单不符。")
                            digest.update(chunk)
                            writer.write(chunk)
                    if copied != item["size"] or digest.hexdigest() != item["sha256"]:
                        raise ValueError("迁移包文件校验失败，文件可能不完整。")
            control = staged / "control.json"
            control.write_bytes(_paused_control(_read_json(control)))
            _validate_database(
                staged / "memory" / "groups.sqlite3", manifest["database_schema_version"]
            )
            _, assets = _sticker_index(staged / "stickers")
            actual_assets = set(manifest["files"]) - {
                "control.json",
                "memory/groups.sqlite3",
                "stickers/stickers.json",
            }
            if actual_assets != set(assets):
                raise ValueError("迁移包的表情索引与素材文件不一致。")
            # Exclusive reservation prevents replacing a directory that another
            # import or user created while this package was being verified.
            state_dir.mkdir(mode=0o700)
            reserved = True
            os.replace(staged, state_dir)
            reserved = False
    except (zipfile.BadZipFile, zipfile.LargeZipFile, RuntimeError):
        raise ValueError("迁移包不是可读取的 ZIP 文件。") from None
    finally:
        if reserved:
            try:
                state_dir.rmdir()
            except OSError:
                pass
    return state_dir
