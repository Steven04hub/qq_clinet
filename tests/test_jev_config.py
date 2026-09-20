"""TypeSafe configuration stays optional and secrets remain masked."""

import pytest
from pydantic import ValidationError

from app.config import Settings


def test_jev_dotenv_configuration(tmp_path):
    path = tmp_path / ".env"
    path.write_text(
        "NAPCAT_ACCESS_TOKEN=test\nTYPESAFE_API_KEY=test-typesafe-secret\n"
        "JEV_MODE=shadow\nJEV_MODEL=jev-1.13.0\nJEV_TIMEOUT_SECONDS=2.5\n"
    )
    settings = Settings(_env_file=path)
    assert settings.typesafe_api_key.get_secret_value() == "test-typesafe-secret"
    assert "test-typesafe-secret" not in repr(settings)
    assert "test-typesafe-secret" not in settings.model_dump_json()
    assert settings.jev_mode == "shadow"
    assert settings.jev_timeout_seconds == 2.5


@pytest.mark.parametrize(
    "field,value",
    [
        ("typesafe_api_key", "two words"),
        ("typesafe_api_key", "中文"),
        ("typesafe_api_key", "key\n"),
        ("jev_timeout_seconds", 0),
        ("jev_timeout_seconds", 16),
        ("jev_timeout_seconds", float("nan")),
        ("jev_timeout_seconds", float("inf")),
        ("jev_mode", "invalid"),
        ("jev_model", ""),
        ("jev_model", "  "),
        ("jev_model", "model\nid"),
    ],
)
def test_invalid_jev_configuration(field, value):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, napcat_access_token="test", **{field: value})


def test_missing_typesafe_key_is_optional():
    settings = Settings(_env_file=None, napcat_access_token="test")
    assert settings.jev_mode == "off"
    assert settings.typesafe_api_key.get_secret_value() == ""
