import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.config import Settings, get_settings


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch):
    """Keep tests independent of the developer's real configuration."""
    field_names = {name.upper() for name in Settings.model_fields}
    for name in os.environ:
        if name.upper() in field_names:
            monkeypatch.delenv(name)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_dotenv_comma_separated_ids_and_optional_llm(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "NAPCAT_ACCESS_TOKEN=test-token\n"
        "ALLOWED_GROUP_IDS=123456, 987654\n"
        "ADMIN_QQ_IDS=456789\n"
        "LLM_API_KEY=\nLLM_BASE_URL=\nLLM_MODEL=\n"
        "MAX_CONTEXT_MESSAGES=30\n"
        "ACTION_TIMEOUT_SECONDS=2.5\n"
        "LOG_LEVEL=debug\n",
        encoding="utf-8",
    )
    settings = Settings(_env_file=env_file)
    assert settings.allowed_group_ids == [123456, 987654]
    assert settings.admin_qq_ids == [456789]
    assert settings.llm_api_key.get_secret_value() == ""
    assert settings.max_context_messages == 30
    assert settings.action_timeout_seconds == 2.5
    assert settings.log_level == "DEBUG"


@pytest.mark.parametrize("value", ["", "   "])
def test_empty_id_lists(value, monkeypatch):
    monkeypatch.setenv("ALLOWED_GROUP_IDS", value)
    monkeypatch.setenv("ADMIN_QQ_IDS", value)
    settings = Settings(napcat_access_token="test-token", _env_file=None)
    assert settings.allowed_group_ids == []
    assert settings.admin_qq_ids == []


def test_environment_overrides_dotenv(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("NAPCAT_ACCESS_TOKEN=file-token\nALLOWED_GROUP_IDS=123\n")
    monkeypatch.setenv("ALLOWED_GROUP_IDS", "456,789")
    assert Settings(_env_file=env_file).allowed_group_ids == [456, 789]


@pytest.mark.parametrize("value", ["abc", "0", "-2", "1,,2", "1.2", "1,"])
def test_invalid_id_lists(value, monkeypatch):
    monkeypatch.setenv("ALLOWED_GROUP_IDS", value)
    with pytest.raises(ValidationError):
        Settings(napcat_access_token="test-token", _env_file=None)


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_context_messages", 0),
        ("max_context_messages", 1),
        ("max_llm_concurrency", -1),
        ("max_reply_chars", 0),
        ("user_rate_limit_seconds", -1),
        ("action_timeout_seconds", 0),
        ("connect_timeout_seconds", float("inf")),
        ("action_timeout_seconds", float("nan")),
        ("llm_timeout_seconds", 0),
        ("llm_timeout_seconds", float("inf")),
        ("llm_timeout_seconds", float("nan")),
        ("llm_max_tokens", 0),
        ("llm_max_tokens", 32769),
        ("llm_max_tokens", 1.5),
        ("max_input_chars", 0),
        ("max_input_chars", -1),
        ("group_queue_timeout_seconds", 0),
        ("group_queue_timeout_seconds", float("inf")),
        ("group_queue_timeout_seconds", float("nan")),
        ("llm_thinking", "unsupported"),
        ("log_level", "not-a-level"),
    ],
)
def test_invalid_limits(field, value):
    with pytest.raises(ValidationError):
        Settings(napcat_access_token="test-token", _env_file=None, **{field: value})


@pytest.mark.parametrize("token", ["", "  ", "token\nwith-newline"])
def test_access_token_must_be_nonempty_and_header_safe(token):
    with pytest.raises(ValidationError):
        Settings(napcat_access_token=token, _env_file=None)


def test_access_token_required():
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


@pytest.mark.parametrize(
    "url", ["ws://127.0.0.1:3001", "wss://localhost:3001/onebot", "ws://[::1]:3001"]
)
def test_loopback_websocket_urls(url):
    assert (
        Settings(napcat_access_token="test-token", napcat_ws_url=url, _env_file=None).napcat_ws_url
        == url
    )


@pytest.mark.parametrize(
    "url",
    [
        "ws://0.0.0.0:3001",
        "ws://192.168.1.5:3001",
        "ws://example.com:3001",
        "http://127.0.0.1:3001",
        "ws://localhost.evil.test:3001",
        "ws://user:password@127.0.0.1:3001",
        "ws://127.0.0.1:3001?access_token=secret",
        "ws://127.0.0.1:3001#secret",
        "ws://127.0.0.1:0",
        "ws://127.0.0.1:99999",
        "ws://127.0.0.1:\n3001",
    ],
)
def test_nonlocal_or_unsafe_websocket_urls_are_rejected(url):
    with pytest.raises(ValidationError):
        Settings(napcat_access_token="test-token", napcat_ws_url=url, _env_file=None)


def test_secret_values_are_hidden_in_representations():
    settings = Settings(
        napcat_access_token="private-napcat-token", llm_api_key="private-llm-key", _env_file=None
    )
    for output in (repr(settings), str(settings), settings.model_dump_json()):
        assert "private-napcat-token" not in output
        assert "private-llm-key" not in output


def test_validation_error_does_not_echo_secrets():
    with pytest.raises(ValidationError) as exc:
        Settings(napcat_access_token="sensitive-token invalid", _env_file=None)
    assert "sensitive-token" not in str(exc.value)


def test_get_settings_loads_dotenv_once(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("NAPCAT_ACCESS_TOKEN=test-token\n")
    first = get_settings()
    assert get_settings() is first


def test_connection_commands_can_load_without_llm_configuration():
    settings = Settings(napcat_access_token="test-token", _env_file=None)
    assert settings.llm_api_key.get_secret_value() == ""
    assert settings.llm_base_url == ""
    assert settings.llm_model == ""
    assert settings.llm_thinking == ""
    assert settings.llm_timeout_seconds == 60
    assert settings.llm_max_tokens == 1024
    assert settings.max_input_chars == 4000
    assert settings.group_queue_timeout_seconds == 30
    assert "昵称" in settings.system_prompt
    assert "没有联网搜索或操作本机的工具" in settings.system_prompt


@pytest.mark.parametrize("field", ["llm_api_key", "llm_base_url", "llm_model"])
def test_ai_run_validation_requires_each_provider_field_without_echoing_values(field):
    values = {
        "llm_api_key": "private-provider-key",
        "llm_base_url": "https://provider.example/private-endpoint",
        "llm_model": "private-model",
    }
    values[field] = ""
    settings = Settings(napcat_access_token="test-token", _env_file=None, **values)
    with pytest.raises(ValueError) as exc:
        settings.validate_llm_config()
    assert str(exc.value) == "Missing LLM configuration: " + field.upper()
    assert "private" not in str(exc.value)


def test_ai_run_validation_reports_all_missing_fields():
    settings = Settings(napcat_access_token="test-token", _env_file=None)
    with pytest.raises(ValueError) as exc:
        settings.validate_llm_config()
    assert str(exc.value) == "Missing LLM configuration: LLM_API_KEY, LLM_BASE_URL, LLM_MODEL"


@pytest.mark.parametrize(
    "url",
    [
        "https://api.deepseek.com",
        "https://provider.example/v1",
        "https://provider.example:8443/custom/v1/",
        "http://localhost:8000/v1",
        "http://127.0.0.1:11434/v1",
        "http://[::1]:8080/v1",
        "https://localhost:8443/v1",
    ],
)
def test_valid_provider_urls_support_ai_run(url):
    settings = Settings(
        napcat_access_token="test-token",
        llm_api_key="test-key",
        llm_base_url=url,
        llm_model="  custom-model  ",
        _env_file=None,
    )
    assert settings.llm_base_url == url
    assert settings.llm_model == "custom-model"
    assert settings.validate_llm_config() is None


@pytest.mark.parametrize(
    "url",
    [
        " ",
        "https://",
        "https://:443/v1",
        "provider.example/v1",
        "ftp://provider.example/v1",
        "ws://127.0.0.1:8000/v1",
        "http://provider.example/v1",
        "http://192.168.1.2/v1",
        "http://0.0.0.0:8000/v1",
        "http://localhost.example/v1",
        "https://user:private-password@provider.example/v1",
        "https://provider.example/v1?token=private-token",
        "https://provider.example/v1#private-token",
        "https://provider.example/v1?",
        "https://provider.example/v1#",
        "https://provider.example:0/v1",
        "https://provider.example:65536/v1",
        "https://provider.example:not-a-port/v1",
        "https://provider.example:/v1",
        "https://provider.example/with space",
        "https://provider.example/\nprivate-path",
        "https://provider.example/\x7fprivate-path",
        "https://provider.example/\x85private-path",
        "https://provider.example\\private-path",
        "https://[invalid-host]/v1",
        "https://[::1]invalid-suffix/v1",
        "https://invalid^host/v1",
    ],
)
def test_unsafe_provider_urls_fail_without_echoing_input(url):
    with pytest.raises(ValidationError) as exc:
        Settings(napcat_access_token="test-token", llm_base_url=url, _env_file=None)
    error = str(exc.value)
    assert "LLM_BASE_URL" in error
    assert "private-" not in error
    assert "input_value" not in error


@pytest.mark.parametrize(
    "key",
    [
        "private-key ",
        "private\tkey",
        "private\nkey",
        "private\rkey",
        "private\x00key",
        "private\x7fkey",
        "private\x85key",
        "private-密钥",
    ],
)
def test_llm_api_keys_must_be_header_safe_without_echoing_input(key):
    with pytest.raises(ValidationError) as exc:
        Settings(napcat_access_token="test-token", llm_api_key=key, _env_file=None)
    assert "LLM_API_KEY" in str(exc.value)
    assert "private" not in str(exc.value)


@pytest.mark.parametrize("prompt", ["", "  ", "\n\t"])
def test_system_prompt_cannot_be_blank(prompt):
    with pytest.raises(ValidationError, match="SYSTEM_PROMPT"):
        Settings(napcat_access_token="test-token", system_prompt=prompt, _env_file=None)


def test_whitespace_model_is_empty_and_requires_configuration_for_run():
    settings = Settings(
        napcat_access_token="test-token",
        llm_api_key="test-key",
        llm_base_url="https://provider.example/v1",
        llm_model=" \n ",
        _env_file=None,
    )
    assert settings.llm_model == ""
    with pytest.raises(ValueError, match="LLM_MODEL"):
        settings.validate_llm_config()


@pytest.mark.parametrize("thinking", ["", "disabled", "enabled"])
def test_llm_options_from_environment(thinking, monkeypatch):
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "45.5")
    monkeypatch.setenv("LLM_MAX_TOKENS", "32768")
    monkeypatch.setenv("LLM_THINKING", thinking)
    monkeypatch.setenv("MAX_INPUT_CHARS", "2000")
    monkeypatch.setenv("GROUP_QUEUE_TIMEOUT_SECONDS", "12.5")
    monkeypatch.setenv("MAX_CONTEXT_MESSAGES", "2")
    monkeypatch.setenv("SYSTEM_PROMPT", "  自定义中文提示词  ")
    settings = Settings(napcat_access_token="test-token", _env_file=None)
    assert settings.llm_timeout_seconds == 45.5
    assert settings.llm_max_tokens == 32768
    assert settings.llm_thinking == thinking
    assert settings.max_input_chars == 2000
    assert settings.group_queue_timeout_seconds == 12.5
    assert settings.max_context_messages == 2
    assert settings.system_prompt == "自定义中文提示词"


def test_example_configuration_has_only_placeholders_and_no_allowed_groups():
    settings = Settings(_env_file=Path(__file__).parents[1] / ".env.example")
    assert settings.napcat_access_token.get_secret_value() == "change-me"
    assert settings.allowed_group_ids == []
    assert settings.admin_qq_ids == []
    assert settings.llm_api_key.get_secret_value() == ""
    assert settings.llm_base_url == "https://api.deepseek.com"
    assert settings.llm_model == "deepseek-flash"
    assert settings.llm_thinking == "disabled"
    with pytest.raises(ValueError, match="LLM_API_KEY"):
        settings.validate_llm_config()
