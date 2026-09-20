import json
import stat
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest
from pydantic import ValidationError

from app.web.server import create_app
from app.web.settings import ControlSettings, ControlSettingsStore


class FakeController:
    def __init__(self) -> None:
        self.settings = ControlSettings(model="test-model", target_group_id="123")
        self.updates = 0
        self.cleared = 0
        self.fail: Exception | None = None

    def status(self) -> dict[str, Any]:
        if self.fail:
            raise self.fail
        return {"settings": self.settings.model_dump(), "connected": True}

    async def update(self, settings: ControlSettings) -> dict[str, Any]:
        if self.fail:
            raise self.fail
        self.settings = settings
        self.updates += 1
        return self.status()

    async def clear_context(self) -> dict[str, Any]:
        if self.fail:
            raise self.fail
        self.cleared += 1
        return self.status()

    async def list_models(self) -> dict[str, Any]:
        if self.fail:
            raise self.fail
        return {"models": ["test-model", "second-model"]}


@pytest.fixture
def controller() -> FakeController:
    return FakeController()


@pytest.fixture
async def browser(controller: FakeController) -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_app(controller)),
        base_url="http://127.0.0.1:8765",
    ) as client:
        yield client


async def test_status_and_models(browser: httpx.AsyncClient) -> None:
    response = await browser.get("/api/status")
    assert response.status_code == 200
    assert response.json()["settings"]["reply_limit_per_minute"] == 10
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert "frame-ancestors 'none'" in response.headers["content-security-policy"]
    assert "access-control-allow-origin" not in response.headers
    assert (await browser.get("/api/models")).json() == {"models": ["test-model", "second-model"]}


async def test_update_and_clear(browser: httpx.AsyncClient, controller: FakeController) -> None:
    payload = controller.settings.model_dump()
    payload.update(mode="chat", model="second-model", reply_limit_per_minute=10)
    response = await browser.put(
        "/api/settings",
        json=payload,
        headers={"X-QQ-Control": "1", "Origin": "http://127.0.0.1:8765"},
    )
    assert response.status_code == 200
    assert controller.settings.mode == "chat"
    assert controller.settings.model == "second-model"
    assert controller.updates == 1
    cleared = await browser.post("/api/context/clear", headers={"X-QQ-Control": "1"})
    assert cleared.status_code == 200
    assert controller.cleared == 1


async def test_profile_routes_preserve_target_scope_and_use_write_guard(browser, controller):
    snapshot = {"group_id": "123", "members": [], "aliases": [], "memes": []}
    controller.profiles = AsyncMock(return_value=snapshot)
    controller.refresh_profiles = AsyncMock(return_value=snapshot)
    controller.correct_alias = AsyncMock(return_value=snapshot)
    assert (await browser.get("/api/profiles")).json() == snapshot
    denied = await browser.post("/api/profiles/refresh", json={"group_id": "123"})
    assert denied.status_code == 403
    controller.refresh_profiles.assert_not_awaited()
    refreshed = await browser.post(
        "/api/profiles/refresh", json={"group_id": "123"}, headers={"X-QQ-Control": "1"}
    )
    assert refreshed.status_code == 200
    controller.refresh_profiles.assert_awaited_once_with("123")
    corrected = await browser.put(
        "/api/profiles/alias",
        json={"group_id": "123", "alias": "小明", "user_ids": ["2"]},
        headers={"X-QQ-Control": "1"},
    )
    assert corrected.status_code == 200
    controller.correct_alias.assert_awaited_once_with("123", "小明", ["2"])


async def test_alias_payload_validation_never_echoes_private_input(browser):
    result = await browser.put(
        "/api/profiles/alias",
        json={"group_id": "123", "alias": "private-nickname", "user_ids": [2]},
        headers={"X-QQ-Control": "1"},
    )
    assert result.status_code == 422
    assert "private-nickname" not in result.text


@pytest.mark.parametrize("path", ["/api/settings", "/api/context/clear"])
async def test_write_requires_custom_header(
    browser: httpx.AsyncClient, controller: FakeController, path: str
) -> None:
    method = "PUT" if "settings" in path else "POST"
    result = await browser.request(method, path, json=controller.settings.model_dump())
    assert result.status_code == 403
    assert controller.updates == controller.cleared == 0


@pytest.mark.parametrize(
    "origin",
    [
        "https://evil.example",
        "http://localhost:8765",
        "http://127.0.0.1:9999",
        "https://127.0.0.1:8765",
        "http://127.0.0.1:8765/path",
        "http://127.0.0.1:8765?unexpected=true",
        "http://@127.0.0.1:8765",
        "null",
    ],
)
async def test_write_rejects_cross_origin(
    browser: httpx.AsyncClient, controller: FakeController, origin: str
) -> None:
    response = await browser.put(
        "/api/settings",
        json=controller.settings.model_dump(),
        headers={"X-QQ-Control": "1", "Origin": origin},
    )
    assert response.status_code == 403
    assert controller.updates == 0


@pytest.mark.parametrize(
    "host",
    [
        "evil.example:8765",
        "127.0.0.1.evil.example",
        "localhost.evil.example",
        "user@localhost",
        "@localhost",
        "localhost:999999",
        "localhost/path",
        "localhost?query=1",
        "127.0.0.1#fragment",
        "",
    ],
)
async def test_untrusted_hosts_are_rejected(browser: httpx.AsyncClient, host: str) -> None:
    response = await browser.get("/api/status", headers={"host": host})
    assert response.status_code == 403
    assert response.headers["cache-control"] == "no-store"


@pytest.mark.parametrize("url", ["http://localhost:8765", "http://[::1]:8765"])
async def test_other_loopback_hosts(controller: FakeController, url: str) -> None:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_app(controller)), base_url=url
    ) as client:
        response = await client.put(
            "/api/settings",
            headers={"X-QQ-Control": "1", "Origin": url},
            json=controller.settings.model_dump(),
        )
        assert response.status_code == 200


async def test_cors_preflight_is_not_allowed(browser: httpx.AsyncClient) -> None:
    response = await browser.options(
        "/api/settings",
        headers={
            "Origin": "https://evil.example",
            "Access-Control-Request-Method": "PUT",
            "Access-Control-Request-Headers": "X-QQ-Control",
        },
    )
    assert response.status_code == 403
    assert "access-control-allow-origin" not in response.headers


async def test_validation_does_not_echo_secret_input(
    browser: httpx.AsyncClient, controller: FakeController
) -> None:
    payload = controller.settings.model_dump()
    payload.update(target_group_id="secret-group-invalid", api_key="never-expose-me")
    response = await browser.put("/api/settings", json=payload, headers={"X-QQ-Control": "1"})
    assert response.status_code == 422
    assert response.json()["fields"] == ["target_group_id"]
    assert "never-expose-me" not in response.text
    assert "secret-group-invalid" not in response.text
    assert "api_key" not in response.text
    assert controller.updates == 0


async def test_invalid_json_returns_safe_error(browser: httpx.AsyncClient) -> None:
    response = await browser.put(
        "/api/settings",
        content=b'{"secret": "private',
        headers={"X-QQ-Control": "1", "Content-Type": "application/json"},
    )
    assert response.status_code == 422
    assert "private" not in response.text


@pytest.mark.parametrize("exception", [ValueError("do-not-leak"), RuntimeError("do-not-leak")])
async def test_controller_errors_are_safe(
    browser: httpx.AsyncClient, controller: FakeController, exception: Exception
) -> None:
    controller.fail = exception
    response = await browser.post("/api/context/clear", headers={"X-QQ-Control": "1"})
    assert response.status_code == (400 if isinstance(exception, ValueError) else 500)
    assert "do-not-leak" not in response.text
    status_response = await browser.get("/api/status")
    assert status_response.status_code == 500
    assert "do-not-leak" not in status_response.text


@pytest.mark.parametrize(
    "field,value",
    [
        ("mode", "unknown"),
        ("target_group_id", "-123"),
        ("target_group_id", "0"),
        ("target_group_id", "１２３"),
        ("target_group_id", "1" * 21),
        ("target_group_id", 123),
        ("model", " "),
        ("model", "secret\nmodel"),
        ("model", "a" * 151),
        ("chat_style", " "),
        ("chat_style", "a" * 1001),
        ("reply_limit_per_minute", 0),
        ("reply_limit_per_minute", 61),
        ("reply_limit_per_minute", "10"),
        ("reply_limit_per_minute", True),
        ("enabled", "true"),
        ("api_key", "secret"),
    ],
)
def test_control_settings_reject_invalid_values(field: str, value: Any) -> None:
    data = {"model": "test-model", field: value}
    with pytest.raises(ValidationError):
        ControlSettings.model_validate(data)


def test_control_settings_strip_text_and_canonicalize_id() -> None:
    settings = ControlSettings(model=" model ", target_group_id=" 00123 ", chat_style=" 风趣 ")
    assert settings.model == "model"
    assert settings.target_group_id == "123"
    assert settings.chat_style == "风趣"
    assert settings.reply_limit_per_minute == 10
    assert settings.mode == "assistant"


def test_store_initial_value_is_an_independent_copy(tmp_path: Path) -> None:
    initial = ControlSettings(model="test-model")
    store = ControlSettingsStore(tmp_path / "missing.json", initial)
    initial.model = "changed"
    loaded = store.load()
    assert loaded.model == "test-model"
    loaded.model = "changed-again"
    assert store.load().model == "test-model"


def test_store_round_trip_private_permissions(tmp_path: Path) -> None:
    path = tmp_path / "controls" / "settings.json"
    settings = ControlSettings(model="test-model", mode="chat", target_group_id="123")
    store = ControlSettingsStore(path, settings)
    store.save(settings)
    assert store.load() == settings
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert set(json.loads(path.read_text())) == set(ControlSettings.model_fields)


@pytest.mark.parametrize("contents", ["broken-secret-json", "null", '{"model":""}'])
def test_corrupt_file_is_not_silently_replaced(tmp_path: Path, contents: str) -> None:
    path = tmp_path / "settings.json"
    path.write_text(contents)
    store = ControlSettingsStore(path, ControlSettings(model="test-model"))
    with pytest.raises(ValueError, match="Saved control settings are invalid") as captured:
        store.load()
    assert "secret" not in str(captured.value)
    assert path.read_text() == contents


def test_failed_atomic_replace_keeps_existing_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "settings.json"
    initial = ControlSettings(model="initial")
    store = ControlSettingsStore(path, initial)
    store.save(initial)

    def fail_replace(*_: object) -> None:
        raise OSError("simulated write failure")

    monkeypatch.setattr("app.web.settings.os.replace", fail_replace)
    with pytest.raises(OSError):
        store.save(ControlSettings(model="changed"))
    assert store.load() == initial
    assert list(tmp_path.iterdir()) == [path]


def test_store_does_not_save_unvalidated_copy(tmp_path: Path) -> None:
    settings = ControlSettings(model="test-model")
    store = ControlSettingsStore(tmp_path / "settings.json", settings)
    with pytest.raises(ValidationError):
        store.save(settings.model_copy(update={"reply_limit_per_minute": 0}))
    assert not store.path.exists()
