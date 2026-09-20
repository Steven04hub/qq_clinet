"""Public bot controls, persisted separately from credentials."""

import json
import os
import tempfile
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.prompts import DEFAULT_CHAT_STYLE


class ControlSettings(BaseModel):
    """Only non-secret settings that are safe to show in the browser."""

    model_config = ConfigDict(extra="forbid", strict=True, hide_input_in_errors=True)

    mode: Literal["assistant", "chat"] = "assistant"
    target_group_id: str = Field(default="", max_length=20)
    model: str = Field(min_length=1, max_length=150)
    reply_limit_per_minute: int = Field(default=10, ge=1, le=60)
    chat_style: str = Field(default=DEFAULT_CHAT_STYLE, min_length=1, max_length=1000)
    enabled: bool = True

    @field_validator("target_group_id", "model", "chat_style", mode="before")
    @classmethod
    def strip_strings(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    @field_validator("target_group_id")
    @classmethod
    def validate_group_id(cls, value: str) -> str:
        if not value:
            return value
        if not value.isascii() or not value.isdigit() or int(value) <= 0:
            raise ValueError("Target group must be a positive numeric ID")
        return str(int(value))

    @field_validator("model")
    @classmethod
    def validate_model(cls, value: str) -> str:
        if any(ord(character) < 32 or ord(character) == 127 for character in value):
            raise ValueError("Model name must not contain control characters")
        return value


class ControlSettingsStore:
    """Atomically save public controls without ever copying the .env file."""

    def __init__(self, path: str | Path, initial: ControlSettings) -> None:
        self.path = Path(path)
        self.initial = initial.model_copy(deep=True)

    def load(self) -> ControlSettings:
        try:
            data = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return self.initial.model_copy(deep=True)
        except (OSError, UnicodeError):
            raise ValueError("Cannot read saved control settings") from None
        try:
            return ControlSettings.model_validate(json.loads(data))
        except (ValueError, TypeError):
            raise ValueError("Saved control settings are invalid") from None

    def save(self, settings: ControlSettings) -> None:
        # Revalidate even if a caller used Pydantic's unchecked model_copy(update=...).
        settings = ControlSettings.model_validate(settings.model_dump())
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                prefix=f".{self.path.name}.",
                dir=self.path.parent,
                delete=False,
            ) as output:
                temporary_path = Path(output.name)
                os.fchmod(output.fileno(), 0o600)
                json.dump(settings.model_dump(), output, ensure_ascii=False, indent=2)
                output.write("\n")
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary_path, self.path)
            temporary_path = None
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
