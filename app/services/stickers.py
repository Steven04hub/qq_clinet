"""Portable local sticker assets and explicitly selected OneBot face segments."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import struct
import tempfile
from pathlib import Path
from threading import RLock

MAX_STICKER_BYTES = 8 * 1024 * 1024
MAX_STICKER_PIXELS = 16_000_000

# Classic QQ face IDs, not Unicode emoji codes or network image URLs.
_BUILTINS = (
    {"id": "face_14", "label": "微笑", "tags": ["微笑", "礼貌", "无语"], "emoji": "🙂"},
    {"id": "face_1", "label": "撇嘴", "tags": ["嫌弃", "不服", "吐槽"], "emoji": "😒"},
    {"id": "face_13", "label": "呲牙", "tags": ["笑", "搞怪", "得逞"], "emoji": "😁"},
)


def _image_type(data: bytes) -> tuple[str, str]:
    """Recognize bounded raster containers; never accept SVG, HTML or arbitrary files."""
    if not isinstance(data, bytes) or not data or len(data) > MAX_STICKER_BYTES:
        raise ValueError("图片不能为空，且不能超过 8 MB。")
    dimensions: tuple[int, int] | None = None
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        if len(data) < 45 or data[12:16] != b"IHDR" or b"IEND" not in data[-16:]:
            raise ValueError("PNG 图片格式不完整。")
        dimensions = struct.unpack(">II", data[16:24])
        extension, mime = "png", "image/png"
    elif data.startswith((b"GIF87a", b"GIF89a")):
        if len(data) < 14 or data[-1:] != b";":
            raise ValueError("GIF 图片格式不完整。")
        dimensions = struct.unpack("<HH", data[6:10])
        extension, mime = "gif", "image/gif"
    elif data.startswith(b"\xff\xd8\xff"):
        if len(data) < 20 or not data.endswith(b"\xff\xd9"):
            raise ValueError("JPEG 图片格式不完整。")
        position = 2
        while position + 4 <= len(data):
            if data[position] != 0xFF:
                break
            while position < len(data) and data[position] == 0xFF:
                position += 1
            if position >= len(data):
                break
            marker = data[position]
            position += 1
            if marker in (0xD9, 0xDA):
                break
            if marker == 0x01 or 0xD0 <= marker <= 0xD8:
                continue
            if position + 2 > len(data):
                break
            length = int.from_bytes(data[position : position + 2], "big")
            if length < 2 or position + length > len(data):
                break
            if marker in (
                0xC0,
                0xC1,
                0xC2,
                0xC3,
                0xC5,
                0xC6,
                0xC7,
                0xC9,
                0xCA,
                0xCB,
                0xCD,
                0xCE,
                0xCF,
            ):
                if length >= 8:
                    height, width = struct.unpack(">HH", data[position + 3 : position + 7])
                    dimensions = (width, height)
                break
            position += length
        if dimensions is None:
            raise ValueError("JPEG 图片缺少有效尺寸。")
        extension, mime = "jpg", "image/jpeg"
    elif data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        if len(data) < 30 or int.from_bytes(data[4:8], "little") + 8 != len(data):
            raise ValueError("WebP 图片格式不完整。")
        kind = data[12:16]
        payload = data[20:]
        if kind == b"VP8X" and len(payload) >= 10:
            dimensions = (
                1 + int.from_bytes(payload[4:7], "little"),
                1 + int.from_bytes(payload[7:10], "little"),
            )
        elif kind == b"VP8L" and len(payload) >= 5 and payload[0] == 0x2F:
            value = int.from_bytes(payload[1:5], "little")
            dimensions = ((value & 0x3FFF) + 1, ((value >> 14) & 0x3FFF) + 1)
        elif kind == b"VP8 " and len(payload) >= 10 and payload[3:6] == b"\x9d\x01\x2a":
            dimensions = (
                int.from_bytes(payload[6:8], "little") & 0x3FFF,
                int.from_bytes(payload[8:10], "little") & 0x3FFF,
            )
        if dimensions is None:
            raise ValueError("WebP 图片缺少有效尺寸。")
        extension, mime = "webp", "image/webp"
    else:
        raise ValueError("只支持 PNG、JPEG、GIF 和 WebP 图片。")
    width, height = dimensions
    if not 1 <= width <= 8192 or not 1 <= height <= 8192 or width * height > MAX_STICKER_PIXELS:
        raise ValueError("图片尺寸过大或无效，请使用不超过 1600 万像素的图片。")
    return extension, mime


class StickerLibrary:
    """Keep JSON metadata and hash-named assets together, ready to copy to another host."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._index = self.path / "stickers.json"
        self._lock = RLock()

    def _read(self) -> list[dict]:
        if not self._index.exists():
            return []
        if self._index.is_symlink():
            raise ValueError("表情目录文件不能使用符号链接。")
        try:
            value = json.loads(self._index.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ValueError("无法读取表情目录，请检查素材目录。") from error
        if not isinstance(value, list):
            raise ValueError("表情目录格式无效。")
        return [
            item
            for item in value
            if isinstance(item, dict)
            and isinstance(item.get("id"), str)
            and re.fullmatch(r"img_[0-9a-f]{64}", item["id"])
            and item.get("extension") in {"png", "jpg", "gif", "webp"}
            and isinstance(item.get("label"), str)
            and isinstance(item.get("tags"), list)
        ]

    def _write_file(self, destination: Path, data: bytes) -> None:
        if destination.is_symlink():
            raise ValueError("表情文件不能使用符号链接。")
        temporary: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=self.path, prefix=".upload-", delete=False
            ) as handle:
                temporary = handle.name
                os.chmod(temporary, 0o600)
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        finally:
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)

    @staticmethod
    def _public(item: dict) -> dict:
        return {
            "id": item["id"],
            "label": item["label"],
            "tags": item["tags"],
            "kind": "image",
            "mime_type": item.get("mime_type", f"image/{item['extension']}"),
            "size": item.get("size", 0),
        }

    def list(self) -> list[dict]:
        with self._lock:
            return [dict(item, tags=list(item["tags"]), kind="face") for item in _BUILTINS] + [
                self._public(item) for item in self._read()
            ]

    def register(
        self,
        data: bytes,
        *,
        filename: str = "",
        label: str = "",
        tags: list[str] | None = None,
    ) -> dict:
        extension, mime = _image_type(data)
        if not isinstance(label, str) or not isinstance(filename, str):
            raise ValueError("表情名称和文件名必须是文字。")
        label = label.strip() or Path(filename).stem.strip() or "自定义表情"
        if len(label) > 80:
            raise ValueError("表情名称最多 80 个字。")
        if tags is None:
            tags = []
        if (
            not isinstance(tags, list)
            or len(tags) > 8
            or any(not isinstance(tag, str) or len(tag.strip()) > 30 for tag in tags)
        ):
            raise ValueError("最多设置 8 个标签，每个标签最多 30 个字。")
        digest = hashlib.sha256(data).hexdigest()
        item = {
            "id": f"img_{digest}",
            "label": label,
            "tags": list(dict.fromkeys(tag.strip() for tag in tags if tag.strip())),
            "extension": extension,
            "mime_type": mime,
            "size": len(data),
        }
        with self._lock:
            records = self._read()
            records = [record for record in records if record["id"] != item["id"]]
            self._write_file(self.path / f"{digest}.{extension}", data)
            records.append(item)
            self._write_file(
                self._index,
                json.dumps(records, ensure_ascii=False, indent=2).encode("utf-8"),
            )
        return self._public(item)

    def read(self, sticker_id: str) -> tuple[bytes, str]:
        """Return an uploaded image for same-origin browser preview."""
        with self._lock:
            item = next((item for item in self._read() if item["id"] == sticker_id), None)
            if item is None:
                raise ValueError("表情素材不存在。")
            image_path = self.path / f"{item['id'][4:]}.{item['extension']}"
            if image_path.is_symlink() or not image_path.is_file():
                raise ValueError("表情素材文件不可用。")
            if image_path.stat().st_size > MAX_STICKER_BYTES:
                raise ValueError("表情素材文件过大。")
            data = image_path.read_bytes()
            _, mime = _image_type(data)
            if hashlib.sha256(data).hexdigest() != item["id"][4:]:
                raise ValueError("表情素材文件校验失败。")
            return data, mime

    def resolve(self, sticker_id: str) -> dict:
        """Produce only fixed-ID QQ faces or registered, verified local image bytes."""
        builtin = next((item for item in _BUILTINS if item["id"] == sticker_id), None)
        if builtin:
            return {"type": "face", "data": {"id": builtin["id"][5:]}}
        data, _ = self.read(sticker_id)
        return {
            "type": "image",
            "data": {"file": "base64://" + base64.b64encode(data).decode("ascii")},
        }

    def delete(self, sticker_id: str) -> None:
        if any(item["id"] == sticker_id for item in _BUILTINS):
            raise ValueError("内置 QQ 表情不能删除。")
        with self._lock:
            records = self._read()
            item = next((item for item in records if item["id"] == sticker_id), None)
            if item is None:
                raise ValueError("表情素材不存在。")
            self._write_file(
                self._index,
                json.dumps(
                    [record for record in records if record["id"] != sticker_id],
                    ensure_ascii=False,
                    indent=2,
                ).encode("utf-8"),
            )
            image_path = self.path / f"{item['id'][4:]}.{item['extension']}"
            if image_path.is_file() and not image_path.is_symlink():
                image_path.unlink()
