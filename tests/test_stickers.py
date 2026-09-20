import base64
import json
import shutil
import stat
import struct

import pytest

from app.services.stickers import MAX_STICKER_BYTES, StickerLibrary

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGNgYGAAAAAEAAH2FzhVAAAAAElFTkSuQmCC"
)
GIF = base64.b64decode("R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7")


def test_builtins_resolve_only_known_face_ids(tmp_path):
    library = StickerLibrary(tmp_path / "assets")
    assert [item["id"] for item in library.list()] == ["face_14", "face_1", "face_13"]
    assert library.resolve("face_13") == {"type": "face", "data": {"id": "13"}}
    with pytest.raises(ValueError):
        library.resolve("face_999")
    with pytest.raises(ValueError):
        library.delete("face_14")


def test_register_and_reopen_sticker_uses_content_id_and_literal_base64(tmp_path):
    path = tmp_path / "assets"
    library = StickerLibrary(path)
    item = library.register(
        PNG, filename="../../other.png", label="看热闹", tags=["吃瓜", " 吃瓜 "]
    )
    assert item["kind"] == "image"
    assert item["mime_type"] == "image/png"
    assert item["tags"] == ["吃瓜"]
    assert item["id"].startswith("img_")
    assert len(item["id"]) == 68
    reopened = StickerLibrary(path)
    assert reopened.read(item["id"]) == (PNG, "image/png")
    segment = reopened.resolve(item["id"])
    assert segment["type"] == "image"
    assert base64.b64decode(segment["data"]["file"].split("base64://", 1)[1]) == PNG
    assert len(reopened.list()) == 4
    assert stat.S_IMODE((path / "stickers.json").stat().st_mode) == 0o600
    assert not (tmp_path / "other.png").exists()
    assert not list(path.glob(".upload-*"))
    assert str(tmp_path) not in (path / "stickers.json").read_text()


def test_duplicate_upload_relabels_same_asset_and_directory_is_portable(tmp_path):
    original = tmp_path / "original"
    library = StickerLibrary(original)
    first = library.register(GIF, filename="好笑.gif")
    second = library.register(GIF, label="更好笑", tags=["搞怪"])
    assert first["id"] == second["id"]
    assert len(library.list()) == 4
    relocated = tmp_path / "new-machine" / "stickers"
    shutil.copytree(original, relocated)
    shutil.rmtree(original)
    migrated = StickerLibrary(relocated)
    assert migrated.read(first["id"]) == (GIF, "image/gif")
    assert migrated.list()[-1]["label"] == "更好笑"
    migrated.delete(first["id"])
    assert len(migrated.list()) == 3
    with pytest.raises(ValueError):
        migrated.resolve(first["id"])


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"<svg><script>alert(1)</script></svg>",
        b"<html>not an image</html>",
        b"GIF89a",
        b"\x89PNG\r\n\x1a\n",
        b"\xff\xd8\xff",
        b"RIFF" + b"\0" * 4 + b"WEBP",
        b"x" * (MAX_STICKER_BYTES + 1),
    ],
)
def test_invalid_images_never_create_assets_or_index(tmp_path, data):
    library = StickerLibrary(tmp_path)
    with pytest.raises(ValueError):
        library.register(data, filename="pretend.png")
    assert list(tmp_path.iterdir()) == []


def test_rejects_giant_dimensions_and_corrupted_registered_images(tmp_path):
    library = StickerLibrary(tmp_path)
    giant = PNG[:16] + struct.pack(">II", 100000, 100000) + PNG[24:]
    with pytest.raises(ValueError):
        library.register(giant)
    item = library.register(PNG)
    image_path = next(tmp_path.glob("*.png"))
    image_path.write_bytes(GIF)
    with pytest.raises(ValueError, match="校验"):
        library.resolve(item["id"])


def test_unknown_paths_and_symlinks_cannot_read_arbitrary_files(tmp_path):
    root = tmp_path / "assets"
    library = StickerLibrary(root)
    for value in ("../secret", "file:///etc/passwd", "https://example.org/a.png", "all"):
        with pytest.raises(ValueError):
            library.resolve(value)
    item = library.register(PNG)
    image_path = next(root.glob("*.png"))
    image_path.unlink()
    secret = tmp_path / "outside"
    secret.write_bytes(PNG)
    image_path.symlink_to(secret)
    with pytest.raises(ValueError, match="不可用"):
        library.read(item["id"])
    with pytest.raises(ValueError, match="符号链接"):
        library.register(PNG)


def test_sticker_index_cannot_inject_filename_or_non_hash_id(tmp_path):
    library = StickerLibrary(tmp_path)
    (tmp_path / "stickers.json").write_text(
        json.dumps(
            [
                {"id": "../../outside", "extension": "png", "label": "x", "tags": []},
                {"id": "img_" + "a" * 64, "extension": "../../outside", "label": "x", "tags": []},
            ]
        )
    )
    assert len(library.list()) == 3


@pytest.mark.parametrize("kwargs", [{"label": "x" * 81}, {"tags": ["x"] * 9}, {"tags": [1]}])
def test_invalid_metadata_is_rejected_before_writing(tmp_path, kwargs):
    with pytest.raises(ValueError):
        StickerLibrary(tmp_path).register(PNG, **kwargs)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "data,extension,mime",
    [
        (
            base64.b64decode(
                "/9j/wAARCAABAAEDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL"
                "/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS"
                "0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlq"
                "c3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJ"
                "ytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAA"
                "AAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMi"
                "MoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RV"
                "VldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0"
                "tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9sAQwACAgIC"
                "AgIDAgIDBQMDAwUGBQUFBQYIBgYGBgYICggICAgICAoKCgoKCgoKDAwMDAwMDg4ODg4P"
                "Dw8PDw8PDw8P/9sAQwECAgIEBAQHBAQHEAsJCxAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQ"
                "EBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQ/90ABAAB/9oADAMBAAIRAxEAPwD+"
                "f+iiigD/2Q=="
            ),
            "jpg",
            "image/jpeg",
        ),
        (
            base64.b64decode("UklGRiIAAABXRUJQVlA4IBYAAAAwAQCdASoBAAEADsD+JaQAA3AAAAAA"),
            "webp",
            "image/webp",
        ),
    ],
)
def test_jpeg_and_webp_are_registered_by_bytes_not_filename(tmp_path, data, extension, mime):
    library = StickerLibrary(tmp_path)
    item = library.register(data, filename="misleading.svg")
    assert item["mime_type"] == mime
    assert list(tmp_path.glob(f"*.{extension}"))
    assert library.read(item["id"]) == (data, mime)
