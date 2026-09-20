import base64
import hashlib
import json
import sqlite3
import stat
import zipfile
from concurrent.futures import ThreadPoolExecutor

import pytest

from app.services import portable
from app.services.portable import export_memory, restore_memory
from app.services.profile_store import GroupMemoryStore
from app.services.stickers import StickerLibrary
from app.web.settings import ControlSettings, ControlSettingsStore

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/mioAAAAASUVORK5CYII="
)


def message(key, text="这家火锅蘸料不错"):
    return {"message_id": str(key), "user_id": "10", "nickname": "小张", "text": text, "time": key}


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "source"
    settings = ControlSettings(
        model="example-chat", mode="chat", target_group_id="100", enabled=True
    )
    control = root / "control.json"
    ControlSettingsStore(control, settings).save(settings)
    memory = GroupMemoryStore(root / "memory" / "groups.sqlite3")
    memory.ingest("100", [message(1)])
    memory.set_alias("100", "火锅王", ["10"], explanation="用户确认，这是老昵称")
    memory.save_analysis(
        "100",
        {
            "members": [
                {
                    "user_id": "10",
                    "summary": "经常聊火锅",
                    "facts": [
                        {
                            "category": "topics",
                            "text": "最近聊过火锅蘸料",
                            "evidence_ids": ["1"],
                            "confidence": "observed",
                        }
                    ],
                }
            ],
            "memes": [
                {
                    "name": "蘸料",
                    "meaning": "聚餐时的老话题",
                    "user_ids": ["10"],
                    "evidence_ids": ["1"],
                }
            ],
        },
    )
    sticker = StickerLibrary(root / "stickers").register(PNG, label="看热闹", tags=["吃瓜"])
    return root, control, memory, sticker


def rewrite_archive(original, destination, *, mutate=None, extra=None, symlink=None, rehash=True):
    with zipfile.ZipFile(original) as package:
        files = {name: package.read(name) for name in package.namelist()}
    manifest = json.loads(files.pop("manifest.json"))
    if mutate:
        mutate(files, manifest)
    if rehash:
        manifest["files"] = {
            name: {"size": len(content), "sha256": hashlib.sha256(content).hexdigest()}
            for name, content in files.items()
        }
    files["manifest.json"] = json.dumps(manifest).encode()
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as package:
        for name, content in files.items():
            if name == symlink:
                info = zipfile.ZipInfo(name)
                info.create_system = 3
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
                package.writestr(info, content)
            else:
                package.writestr(name, content)
        if extra:
            package.writestr(*extra)
    return destination


def test_export_restore_profiles_history_aliases_and_portable_stickers(source, tmp_path):
    root, control, memory, sticker = source
    (root / ".env").write_text("SECRET_DO_NOT_EXPORT=this-is-private")
    (root / "private.log").write_text("private log")
    (root / "memory" / "unrelated.txt").write_text("not memory")
    destination = tmp_path / "move.zip"
    assert export_memory(control, destination) == destination
    assert stat.S_IMODE(destination.stat().st_mode) == 0o600
    assert json.loads(control.read_text())["enabled"] is True
    with zipfile.ZipFile(destination) as package:
        manifest = json.loads(package.read("manifest.json"))
        assert manifest["schema_version"] == 1
        assert json.loads(package.read("control.json"))["enabled"] is False
        assert ".env" not in package.namelist()
        assert "private.log" not in package.namelist()
        assert "memory/unrelated.txt" not in package.namelist()
        assert all(not name.startswith("/") for name in package.namelist())
        for name, metadata in manifest["files"].items():
            content = package.read(name)
            assert metadata == {"size": len(content), "sha256": hashlib.sha256(content).hexdigest()}
    restored = tmp_path / "new-machine" / "state"
    assert restore_memory(destination, restored) == restored
    moved = GroupMemoryStore(restored / "memory" / "groups.sqlite3")
    assert moved.overview("100") == memory.overview("100")
    assert moved.search("100", "火锅王")[0]["message_id"] == "1"
    assert StickerLibrary(restored / "stickers").read(sticker["id"]) == (PNG, "image/png")
    settings = json.loads((restored / "control.json").read_text())
    assert settings["enabled"] is False
    assert settings["target_group_id"] == "100"
    assert str(root) not in (restored / "stickers" / "stickers.json").read_text()
    for path in restored.rglob("*"):
        assert stat.S_IMODE(path.stat().st_mode) == (0o700 if path.is_dir() else 0o600)
    assert not list(tmp_path.glob(".qq-memory-*"))


def test_export_without_uploaded_stickers_is_supported(source, tmp_path):
    root, control, _, sticker = source
    StickerLibrary(root / "stickers").delete(sticker["id"])
    (root / "stickers" / "stickers.json").unlink()
    (root / "stickers").rmdir()
    package = export_memory(control, tmp_path / "empty-stickers.zip")
    restored = restore_memory(package, tmp_path / "restored")
    assert len(StickerLibrary(restored / "stickers").list()) == 3


def test_export_uses_consistent_sqlite_backup_while_memory_receives_messages(source, tmp_path):
    _, control, memory, _ = source
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = pool.submit(
            memory.ingest, "100", [message(i, f"第{i}次吃火锅") for i in range(2, 100)]
        )
        package = export_memory(control, tmp_path / "live.zip")
        pending.result()
    restored = restore_memory(package, tmp_path / "restored")
    with sqlite3.connect(restored / "memory" / "groups.sqlite3") as db:
        assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        count = db.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        assert count in {1, 99}
        assert db.execute("SELECT COUNT(*) FROM memory_documents").fetchone()[0] == count


def test_export_never_overwrites_existing_target(source, tmp_path):
    _, control, _, _ = source
    target = tmp_path / "existing.zip"
    target.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        export_memory(control, target)
    assert target.read_bytes() == b"keep"


def test_restore_never_overwrites_existing_directory(source, tmp_path):
    _, control, _, _ = source
    package = export_memory(control, tmp_path / "memory.zip")
    existing = tmp_path / "existing"
    existing.mkdir()
    (existing / "keep").write_text("keep")
    with pytest.raises(FileExistsError):
        restore_memory(package, existing)
    assert (existing / "keep").read_text() == "keep"
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileExistsError):
        restore_memory(package, empty)


@pytest.mark.parametrize("which", ["control", "database", "memory_dir", "sticker", "stickers_dir"])
def test_export_rejects_symbolic_links(source, tmp_path, which):
    root, control, _, sticker = source
    paths = {
        "control": control,
        "database": root / "memory" / "groups.sqlite3",
        "memory_dir": root / "memory",
        "sticker": root / "stickers" / (sticker["id"][4:] + ".png"),
        "stickers_dir": root / "stickers",
    }
    path = paths[which]
    moved = tmp_path / "moved"
    path.rename(moved)
    path.symlink_to(moved, target_is_directory=moved.is_dir())
    with pytest.raises(ValueError):
        export_memory(control, tmp_path / "rejected.zip")
    assert not (tmp_path / "rejected.zip").exists()


@pytest.mark.parametrize(
    "name",
    [
        "../escape",
        "/absolute",
        "C:/windows",
        "stickers\\escape.png",
        "memory/../escape",
        ".env",
        "stickers/private.txt",
        "stickers/",
    ],
)
def test_restore_rejects_zip_path_traversal_and_unlisted_file_types(source, tmp_path, name):
    _, control, _, _ = source
    package = export_memory(control, tmp_path / "valid.zip")
    malicious = rewrite_archive(package, tmp_path / "bad.zip", extra=(name, b"bad"))
    target = tmp_path / "state"
    with pytest.raises(ValueError):
        restore_memory(malicious, target)
    assert not target.exists()
    assert not list(tmp_path.glob(".qq-memory-restore-*"))
    assert not (tmp_path / "escape").exists()


def test_restore_rejects_symbolic_link_zip_entries(source, tmp_path):
    _, control, _, _ = source
    package = export_memory(control, tmp_path / "valid.zip")
    bad = rewrite_archive(package, tmp_path / "link.zip", symlink="stickers/stickers.json")
    with pytest.raises(ValueError, match="符号链接"):
        restore_memory(bad, tmp_path / "state")
    assert not (tmp_path / "state").exists()


def test_restore_rejects_checksum_mismatch_and_leaves_no_partial_state(source, tmp_path):
    _, control, _, _ = source
    package = export_memory(control, tmp_path / "valid.zip")

    def change(files, _manifest):
        files["control.json"] = files["control.json"].replace(b"example-chat", b"corrupt-chat")

    bad = rewrite_archive(package, tmp_path / "bad.zip", mutate=change, rehash=False)
    with pytest.raises(ValueError, match="校验失败"):
        restore_memory(bad, tmp_path / "state")
    assert not (tmp_path / "state").exists()
    assert not list(tmp_path.glob(".qq-memory-restore-*"))


def test_restore_forces_pause_even_if_archive_settings_enable_bot(source, tmp_path):
    _, control, _, _ = source
    package = export_memory(control, tmp_path / "valid.zip")

    def change(files, _manifest):
        settings = json.loads(files["control.json"])
        settings["enabled"] = True
        files["control.json"] = json.dumps(settings).encode()

    active = rewrite_archive(package, tmp_path / "active.zip", mutate=change)
    restored = restore_memory(active, tmp_path / "state")
    assert json.loads((restored / "control.json").read_text())["enabled"] is False


@pytest.mark.parametrize("kind", ["corrupt", "empty", "newer"])
def test_restore_validates_sqlite_content_not_only_zip_hash(source, tmp_path, kind):
    _, control, _, _ = source
    package = export_memory(control, tmp_path / "valid.zip")
    if kind == "corrupt":
        replacement = b"this is not sqlite"
    else:
        database = tmp_path / "other.sqlite3"
        if kind == "newer":
            with zipfile.ZipFile(package) as archive:
                database.write_bytes(archive.read("memory/groups.sqlite3"))
        with sqlite3.connect(database) as db:
            db.execute(f"PRAGMA user_version={999 if kind == 'newer' else 1}")
        replacement = database.read_bytes()

    def change(files, _manifest):
        files["memory/groups.sqlite3"] = replacement

    bad = rewrite_archive(package, tmp_path / "bad.zip", mutate=change)
    with pytest.raises(ValueError, match="数据库"):
        restore_memory(bad, tmp_path / "state")
    assert not (tmp_path / "state").exists()


@pytest.mark.parametrize(
    "limit", ["MAX_FILES", "MAX_TOTAL_BYTES", "MAX_DATABASE_BYTES", "MAX_JSON_BYTES"]
)
def test_restore_enforces_entry_and_uncompressed_size_limits(source, tmp_path, monkeypatch, limit):
    _, control, _, _ = source
    package = export_memory(control, tmp_path / "valid.zip")
    monkeypatch.setattr(portable, limit, 1)
    with pytest.raises(ValueError):
        restore_memory(package, tmp_path / "state")
    assert not (tmp_path / "state").exists()


def test_export_rejects_unknown_control_fields_instead_of_copying_credentials(source, tmp_path):
    _, control, _, _ = source
    settings = json.loads(control.read_text())
    settings["api_key"] = "do-not-copy-secret"
    control.write_text(json.dumps(settings))
    with pytest.raises(ValueError, match="未知字段"):
        export_memory(control, tmp_path / "bad.zip")
    assert not (tmp_path / "bad.zip").exists()


def test_restore_validates_sticker_index_and_material_content(source, tmp_path):
    _, control, _, _ = source
    package = export_memory(control, tmp_path / "valid.zip")

    def change(files, _manifest):
        image = next(name for name in files if name.endswith(".png"))
        files[image] = files[image][:-1] + b"0"

    bad = rewrite_archive(package, tmp_path / "bad.zip", mutate=change)
    with pytest.raises(ValueError):
        restore_memory(bad, tmp_path / "state")
    assert not (tmp_path / "state").exists()


def test_unsupported_archive_version_is_rejected(source, tmp_path):
    _, control, _, _ = source
    package = export_memory(control, tmp_path / "valid.zip")
    bad = rewrite_archive(
        package,
        tmp_path / "bad.zip",
        mutate=lambda _files, manifest: manifest.update(schema_version=999),
    )
    with pytest.raises(ValueError, match="版本"):
        restore_memory(bad, tmp_path / "state")


def test_restore_failed_publish_removes_reservation(source, tmp_path, monkeypatch):
    _, control, _, _ = source
    package = export_memory(control, tmp_path / "valid.zip")

    def fail(*_args):
        raise OSError("simulated disk error")

    monkeypatch.setattr(portable.os, "replace", fail)
    with pytest.raises(OSError):
        restore_memory(package, tmp_path / "state")
    assert not (tmp_path / "state").exists()
    assert not list(tmp_path.glob(".qq-memory-restore-*"))
