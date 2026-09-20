"""Standard logging with secret redaction at the final output boundary."""

import logging
import re
from collections.abc import Iterable


class RedactingFormatter(logging.Formatter):
    """Redact messages and exception traces, including third-party output."""

    def __init__(self, secrets: Iterable[str] = ()) -> None:
        super().__init__("%(asctime)s %(levelname)s %(name)s: %(message)s")
        self._secrets = sorted({secret for secret in secrets if secret}, key=len, reverse=True)

    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        for secret in self._secrets:
            message = message.replace(secret, "[REDACTED]")
        # Hide the whole header value, including schemes such as Bearer or Basic.
        message = re.sub(
            r"(?i)(\bauthorization\b[\"']?\s*[:=]\s*)[^\r\n]+",
            r"\1[REDACTED]",
            message,
        )
        return re.sub(
            r"(?i)(\b(?:api[_-]?key|access[_-]?token|token)\b[\"']?\s*[:=]\s*)"
            r"(?:[\"'][^\"'\r\n]*[\"']|[^\s,;}&]+)",
            r"\1[REDACTED]",
            message,
        )


def configure_logging(level: str = "INFO", secrets: Iterable[str] = ()) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(RedactingFormatter(secrets))
    logging.basicConfig(level=level.upper(), handlers=[handler], force=True)
    # WebSocket DEBUG logs include HTTP headers and complete message frames.
    for name in (
        "websockets",
        "websockets.client",
        "websockets.server",
        "openai",
        "httpx",
        "httpcore",
        "httpx2",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)
