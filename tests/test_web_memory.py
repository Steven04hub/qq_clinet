"""New memory/sticker HTTP routes use only temporary files and local fake services."""

import base64
import io
import json
import zipfile
from pathlib import Path

import httpx
import pytest

from app.services.profile_store import GroupMemoryStore
from app.services.stickers import MAX_STICKER_BYTES, StickerLibrary
from app.web.server import create_app
from tests.test_stickers import GIF, PNG
from tests.test_web import FakeController

WRITE_HEADERS = {"X-QQ-Control": "1", "Origin": "http://127.0.0.1:8765"}


class MemoryController(FakeController):
    def __init__(self, root: Path):
        super().__init__()
        self.memory = GroupMemoryStore(root / "memory" / "groups.sqlite3")
        self.stickers = StickerLibrary(root / "assets")
        self.search_calls = []
        self.upload_calls = []
        self.image_calls = []
        self.export_paths = []
        self.export_error = None

    async def search_memory(self, group_id, query):
        self.search_calls.append((group_id, query))
        if group_id != self.settings.target_group_id:
            raise ValueError("PRIVATE_ERROR_OTHER_GROUP")
        return {"group_id": group_id, "results": self.memory.search(group_id, query, limit=8)}

    async def list_stickers(self):
        return {"stickers": self.stickers.list()}

    async def add_sticker(self, data, filename, label, tags):
        self.upload_calls.append((len(data), filename, label, tags))
        self.stickers.register(data, filename=filename, label=label, tags=tags)
        return await self.list_stickers()

    async def sticker_image(self, sticker_id):
        self.image_calls.append(sticker_id)
        return self.stickers.read(sticker_id)

    async def export_memory(self, destination):
        self.export_paths.append(destination)
        with zipfile.ZipFile(destination, "w") as archive:
            archive.writestr("manifest.json", json.dumps({"version": 1, "kind": "test-memory"}))
            archive.writestr("memory.json", json.dumps({"fictional": True}))
        if self.export_error:
            raise self.export_error
        return destination


@pytest.fixture
def memory_controller(tmp_path):
    return MemoryController(tmp_path)


@pytest.fixture
async def memory_browser(memory_controller):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_app(memory_controller)),
        base_url="http://127.0.0.1:8765",
    ) as client:
        yield client


def upload(data=PNG, **changes):
    body = {
        "filename": "funny.png",
        "label": "看热闹",
        "tags": ["吃瓜"],
        "data_base64": base64.b64encode(data).decode("ascii"),
    }
    body.update(changes)
    return body


@pytest.mark.parametrize(
    "path,body",
    [
        ("/api/memory/search", {"group_id": "123", "query": "测试"}),
        ("/api/stickers", upload()),
        ("/api/memory/export", None),
    ],
)
@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Origin": "http://127.0.0.1:8765"},
        {"X-QQ-Control": "1", "Origin": "https://untrusted.example"},
        {"X-QQ-Control": "1", "Origin": "http://127.0.0.1:9999"},
    ],
)
async def test_memory_writes_require_custom_header_and_matching_origin(
    memory_browser, memory_controller, path, body, headers
):
    response = await memory_browser.post(path, json=body, headers=headers)
    assert response.status_code == 403
    assert memory_controller.search_calls == []
    assert memory_controller.upload_calls == []
    assert memory_controller.export_paths == []
    assert "access-control-allow-origin" not in response.headers


async def test_search_passes_only_requested_group_and_query_and_returns_group_local_evidence(
    memory_browser, memory_controller
):
    for group_id, text in [("123", "海边烧烤是周五"), ("456", "海边烧烤外群秘密")]:
        memory_controller.memory.ingest(
            group_id, [{"message_id": "1", "user_id": "200", "text": text, "nickname": "群友"}]
        )
    response = await memory_browser.post(
        "/api/memory/search", json={"group_id": "123", "query": "海边烧烤"}, headers=WRITE_HEADERS
    )
    assert response.status_code == 200
    assert memory_controller.search_calls == [("123", "海边烧烤")]
    assert response.json()["group_id"] == "123"
    assert response.json()["results"][0]["text"] == "海边烧烤是周五"
    assert "外群秘密" not in response.text


async def test_search_of_non_target_group_returns_safe_error(memory_browser, memory_controller):
    response = await memory_browser.post(
        "/api/memory/search",
        json={"group_id": "456", "query": "PRIVATE_QUERY"},
        headers=WRITE_HEADERS,
    )
    assert response.status_code == 400
    assert memory_controller.search_calls == [("456", "PRIVATE_QUERY")]
    assert "PRIVATE" not in response.text


@pytest.mark.parametrize(
    "body",
    [
        {"group_id": 123, "query": "PRIVATE_QUERY"},
        {"group_id": "../private", "query": "PRIVATE_QUERY"},
        {"group_id": "123", "query": ""},
        {"group_id": "123", "query": "q" * 501},
        {"group_id": "123", "query": "PRIVATE_QUERY", "path": "/PRIVATE_PATH/database"},
    ],
)
async def test_search_payload_validation_does_not_echo_query_or_path(
    memory_browser, memory_controller, body
):
    response = await memory_browser.post("/api/memory/search", json=body, headers=WRITE_HEADERS)
    assert response.status_code == 422
    assert "PRIVATE" not in response.text
    assert memory_controller.search_calls == []


@pytest.mark.parametrize("data,mime", [(PNG, "image/png"), (GIF, "image/gif")])
async def test_sticker_upload_list_and_preview_use_registered_bytes_and_correct_mime(
    memory_browser, memory_controller, tmp_path, data, mime
):
    listed = await memory_browser.get("/api/stickers")
    assert listed.status_code == 200 and len(listed.json()["stickers"]) == 3
    uploaded = await memory_browser.post(
        "/api/stickers",
        json=upload(data, filename="../../PRIVATE_FILE.png"),
        headers=WRITE_HEADERS,
    )
    assert uploaded.status_code == 200
    item = next(item for item in uploaded.json()["stickers"] if item["kind"] == "image")
    assert item["mime_type"] == mime
    assert "PRIVATE_FILE" not in uploaded.text and str(tmp_path) not in uploaded.text
    assert not (tmp_path / "PRIVATE_FILE.png").exists()
    preview = await memory_browser.get(f"/api/stickers/{item['id']}/image")
    assert preview.status_code == 200 and preview.content == data
    assert preview.headers["content-type"] == mime
    assert preview.headers["cache-control"] == "no-store"
    assert preview.headers["x-content-type-options"] == "nosniff"
    assert memory_controller.image_calls == [item["id"]]


@pytest.mark.parametrize(
    "invalid", ["not base64 PRIVATE", "%%%%", "Zm8", "data:image/png;base64,AAAA"]
)
async def test_invalid_base64_fails_before_controller_without_echoing_input(
    memory_browser, memory_controller, invalid
):
    response = await memory_browser.post(
        "/api/stickers", json=upload(data_base64=invalid), headers=WRITE_HEADERS
    )
    assert response.status_code == 400
    assert "PRIVATE" not in response.text
    assert memory_controller.upload_calls == []


@pytest.mark.parametrize("length", ["12000001", "999999999999999999999", "-1", "invalid"])
async def test_oversize_or_invalid_content_length_rejected_before_body_is_used(
    memory_browser, memory_controller, length
):
    response = await memory_browser.post(
        "/api/stickers",
        content=b"PRIVATE_BODY_NEVER_PARSE",
        headers={**WRITE_HEADERS, "Content-Length": length, "Content-Type": "application/json"},
    )
    assert response.status_code == 413
    assert "PRIVATE" not in response.text and memory_controller.upload_calls == []


async def test_decoded_image_size_limit_is_enforced_even_within_json_request_limit(
    memory_browser, memory_controller
):
    response = await memory_browser.post(
        "/api/stickers", json=upload(b"x" * (MAX_STICKER_BYTES + 1)), headers=WRITE_HEADERS
    )
    assert response.status_code == 400
    assert memory_controller.upload_calls[0][0] == MAX_STICKER_BYTES + 1
    assert not list(memory_controller.stickers.path.iterdir())


@pytest.mark.parametrize("sticker_id", ["PRIVATE_ABSOLUTE_PATH", "..%5cPRIVATE_FILE", "img_0000"])
async def test_unknown_or_path_like_preview_ids_return_no_private_path(memory_browser, sticker_id):
    response = await memory_browser.get(f"/api/stickers/{sticker_id}/image")
    assert response.status_code == 404
    assert "PRIVATE" not in response.text


async def test_preview_rejects_symlink_without_exposing_target_file(
    memory_browser, memory_controller, tmp_path
):
    item = memory_controller.stickers.register(PNG, label="测试")
    asset = next(memory_controller.stickers.path.glob("*.png"))
    outside = tmp_path / "PRIVATE_OUTSIDE_FILE"
    outside.write_bytes(b"PRIVATE_OUTSIDE_CONTENT")
    asset.unlink()
    asset.symlink_to(outside)
    response = await memory_browser.get(f"/api/stickers/{item['id']}/image")
    assert response.status_code == 404
    assert "PRIVATE" not in response.text


async def test_sticker_reads_reject_untrusted_host(memory_browser, memory_controller):
    response = await memory_browser.get(
        "/api/stickers/img_missing/image", headers={"Host": "evil.example"}
    )
    assert response.status_code == 403 and memory_controller.image_calls == []


async def test_export_downloads_valid_zip_and_cleans_temporary_folder_after_response(
    memory_browser, memory_controller
):
    response = await memory_browser.post("/api/memory/export", headers=WRITE_HEADERS)
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    assert 'filename="qq-memory.zip"' in response.headers["content-disposition"]
    assert response.headers["cache-control"] == "no-store"
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert set(archive.namelist()) == {"manifest.json", "memory.json"}
        assert json.loads(archive.read("memory.json")) == {"fictional": True}
    destination = memory_controller.export_paths[0]
    assert not destination.exists() and not destination.parent.exists()
    assert str(destination.parent) not in str(response.headers)


async def test_export_failure_also_cleans_partial_archive_and_hides_local_path(
    memory_browser, memory_controller, caplog
):
    memory_controller.export_error = OSError("/PRIVATE_PATH/private-db.sqlite")
    response = await memory_browser.post("/api/memory/export", headers=WRITE_HEADERS)
    assert response.status_code == 500
    destination = memory_controller.export_paths[0]
    assert not destination.parent.exists()
    assert "PRIVATE" not in response.text and "PRIVATE" not in caplog.text
