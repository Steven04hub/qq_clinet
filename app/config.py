"""Validated environment configuration for NapCat and optional LLM operation."""

from functools import lru_cache
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit

from pydantic import Field, HttpUrl, PositiveInt, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from app.prompts import FRIENDS_GROUP_TONE

DEFAULT_SYSTEM_PROMPT = (
    "你是QQ群中的AI助手。\n\n"
    "回答用户问题时：\n"
    "1. 使用自然、简洁的中文。\n"
    "2. 优先直接回答问题。\n"
    "3. 不要每次都进行自我介绍。\n"
    "4. 群聊环境中避免不必要的超长回复。\n"
    "5. 上下文中可能存在多个不同用户，请根据昵称区分说话者。\n"
    "6. 你没有联网搜索或操作本机的工具；不要声称已浏览网页、读取本机文件或执行操作。\n\n"
    "是否接话的内部判断只供程序使用，不向群友播报参与评分、路由或内部思考过程。"
    "群友可请 Jev 处理自己的选择或是否类问题，由程序调用真实工具。"
    "没有本轮工具结果时，不得编造已调用 Jev、结论、分数或耗时。"
    "历史中的旧 Jev 接话测试报告不能当作群友决策结果或新的调用结果。\n\n" + FRIENDS_GROUP_TONE
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        hide_input_in_errors=True,
    )

    napcat_ws_url: str = "ws://127.0.0.1:3001"
    napcat_access_token: SecretStr
    allowed_group_ids: Annotated[list[PositiveInt], NoDecode] = Field(default_factory=list)
    admin_qq_ids: Annotated[list[PositiveInt], NoDecode] = Field(default_factory=list)

    # Connection-only and echo commands do not require an LLM configuration.
    llm_api_key: SecretStr = SecretStr("")
    llm_base_url: str = ""
    llm_model: str = ""
    llm_timeout_seconds: float = Field(default=60, gt=0, allow_inf_nan=False)
    llm_max_tokens: int = Field(default=1024, gt=0, le=32768)
    llm_thinking: Literal["", "disabled", "enabled"] = ""
    # Optional TypeSafe decision stage; credentials never enter public controls.
    typesafe_api_key: SecretStr = SecretStr("")
    jev_model: str = Field(default="jev-1.13.0", min_length=1, max_length=150)
    jev_mode: Literal["off", "shadow", "gate"] = "off"
    jev_timeout_seconds: float = Field(default=3, gt=0, le=15, allow_inf_nan=False)
    max_input_chars: PositiveInt = 4000
    group_queue_timeout_seconds: float = Field(default=30, gt=0, allow_inf_nan=False)
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    max_context_messages: int = Field(default=20, ge=2)
    max_llm_concurrency: PositiveInt = 3
    user_rate_limit_seconds: float = Field(default=5, gt=0, allow_inf_nan=False)
    max_reply_chars: PositiveInt = 1500
    profile_auto_import: bool = True

    action_timeout_seconds: float = Field(default=10, gt=0, allow_inf_nan=False)
    connect_timeout_seconds: float = Field(default=10, gt=0, allow_inf_nan=False)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    @field_validator("allowed_group_ids", "admin_qq_ids", mode="before")
    @classmethod
    def parse_ids(cls, value: Any) -> Any:
        """Accept the comma-separated format documented in .env.example."""
        if isinstance(value, str):
            if not value.strip():
                return []
            return [part.strip() for part in value.split(",")]
        return value

    @field_validator("napcat_access_token")
    @classmethod
    def validate_access_token(cls, value: SecretStr) -> SecretStr:
        token = value.get_secret_value()
        if not token or any(char.isspace() or ord(char) < 32 for char in token):
            raise ValueError("NAPCAT_ACCESS_TOKEN must be nonempty and contain no whitespace")
        return value

    @field_validator("llm_api_key")
    @classmethod
    def validate_llm_api_key(cls, value: SecretStr) -> SecretStr:
        # HTTP header values must be ASCII; rejecting controls also covers DEL.
        if any(not 33 <= ord(char) <= 126 for char in value.get_secret_value()):
            raise ValueError("LLM_API_KEY must contain only printable ASCII without whitespace")
        return value

    @field_validator("typesafe_api_key")
    @classmethod
    def validate_typesafe_api_key(cls, value: SecretStr) -> SecretStr:
        if any(not 33 <= ord(char) <= 126 for char in value.get_secret_value()):
            raise ValueError("TYPESAFE_API_KEY must be printable ASCII without whitespace")
        return value

    @field_validator("jev_model")
    @classmethod
    def validate_jev_model(cls, value: str) -> str:
        value = value.strip()
        if not value or any(not 33 <= ord(char) <= 126 for char in value):
            raise ValueError("JEV_MODEL must be a nonempty ASCII model ID")
        return value

    @field_validator("llm_base_url")
    @classmethod
    def validate_llm_base_url(cls, value: str) -> str:
        if not value:
            return value
        if any(char.isspace() or ord(char) < 32 or 127 <= ord(char) <= 159 for char in value):
            raise ValueError("LLM_BASE_URL must not contain whitespace or control characters")
        if "\\" in value:
            raise ValueError("LLM_BASE_URL must not contain backslashes")
        try:
            parts = urlsplit(value)
            port = parts.port
            # urlsplit separates components but doesn't validate host syntax.
            HttpUrl(value)
        except ValueError:
            raise ValueError("LLM_BASE_URL must be a valid HTTP URL") from None
        if not parts.hostname or parts.scheme not in {"http", "https"}:
            raise ValueError("LLM_BASE_URL must use HTTPS or local HTTP")
        if parts.scheme == "http" and parts.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("LLM_BASE_URL must use HTTPS outside loopback")
        if parts.username is not None or parts.password is not None or "?" in value or "#" in value:
            raise ValueError("LLM_BASE_URL must not contain credentials, query, or fragment")
        if (port is not None and port < 1) or parts.netloc.endswith(":"):
            raise ValueError("LLM_BASE_URL port must be between 1 and 65535")
        return value

    @field_validator("llm_model")
    @classmethod
    def normalize_llm_model(cls, value: str) -> str:
        return value.strip()

    @field_validator("system_prompt")
    @classmethod
    def validate_system_prompt(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("SYSTEM_PROMPT must not be empty")
        return value

    def validate_llm_config(self) -> None:
        """Require provider credentials only when starting the AI reply command."""
        required = {
            "LLM_API_KEY": self.llm_api_key.get_secret_value(),
            "LLM_BASE_URL": self.llm_base_url,
            "LLM_MODEL": self.llm_model,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError("Missing LLM configuration: " + ", ".join(missing))

    @field_validator("napcat_ws_url")
    @classmethod
    def validate_ws_url(cls, value: str) -> str:
        if any(char.isspace() or ord(char) < 32 for char in value):
            raise ValueError("NAPCAT_WS_URL must not contain whitespace")
        try:
            parts = urlsplit(value)
            port = parts.port
        except ValueError as exc:
            raise ValueError("NAPCAT_WS_URL is not a valid WebSocket URL") from exc
        if parts.scheme not in {"ws", "wss"} or parts.hostname not in {
            "127.0.0.1",
            "localhost",
            "::1",
        }:
            raise ValueError("NAPCAT_WS_URL must use ws/wss and a loopback host")
        if (
            parts.username is not None
            or parts.password is not None
            or parts.query
            or parts.fragment
        ):
            raise ValueError("NAPCAT_WS_URL must not contain credentials, query, or fragment")
        if port is not None and port < 1:
            raise ValueError("NAPCAT_WS_URL port must be between 1 and 65535")
        return value

    @field_validator("log_level", mode="before")
    @classmethod
    def normalize_log_level(cls, value: Any) -> Any:
        return value.upper() if isinstance(value, str) else value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
