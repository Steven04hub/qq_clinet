import logging

import pytest

from app.utils.logging import configure_logging


@pytest.fixture
def restore_logging():
    root = logging.getLogger()
    old_handlers, old_level = root.handlers[:], root.level
    websocket_levels = {
        name: logging.getLogger(name).level
        for name in (
            "websockets",
            "websockets.client",
            "websockets.server",
            "openai",
            "httpx",
            "httpcore",
            "httpx2",
        )
    }
    yield
    for handler in root.handlers:
        if handler not in old_handlers:
            handler.close()
    root.handlers = old_handlers
    root.setLevel(old_level)
    for name, level in websocket_levels.items():
        logging.getLogger(name).setLevel(level)


def test_logging_redacts_secrets_and_exception_traces(capsys, restore_logging):
    configure_logging("DEBUG", secrets=("private-token", "private-key", ""))
    logger = logging.getLogger("app.test")
    logger.info("Connected with %s", "private-token")
    try:
        raise RuntimeError("Failed with private-key")
    except RuntimeError:
        logger.exception("Action failed")
    output = capsys.readouterr().err
    assert "Connected" in output
    assert "Action failed" in output
    assert "[REDACTED]" in output
    assert "private-token" not in output
    assert "private-key" not in output


@pytest.mark.parametrize(
    "message,secret",
    [
        ("Authorization: Bearer unknown-token", "unknown-token"),
        ("{'Authorization': 'Basic unknown-credential'}", "unknown-credential"),
        ("api_key=unknown-key", "unknown-key"),
        ("access_token=unknown-token", "unknown-token"),
        ('{"token": "unknown-token"}', "unknown-token"),
    ],
)
def test_logging_redacts_header_and_key_values(message, secret, capsys, restore_logging):
    configure_logging()
    logging.getLogger("app.test").warning(message)
    output = capsys.readouterr().err
    assert secret not in output
    assert "[REDACTED]" in output


def test_debug_does_not_enable_websocket_wire_logging(capsys, restore_logging):
    configure_logging("DEBUG")
    logging.getLogger("app.test").debug("application diagnostic")
    logging.getLogger("websockets.client").debug("Authorization: wire-token")
    logging.getLogger("websockets.client").debug("private message frame")
    output = capsys.readouterr().err
    assert "application diagnostic" in output
    assert "wire-token" not in output
    assert "private message frame" not in output


def test_logging_configuration_does_not_duplicate_output(capsys, restore_logging):
    configure_logging()
    configure_logging()
    logging.getLogger("app.test").info("one-record-only")
    assert capsys.readouterr().err.count("one-record-only") == 1


def test_debug_suppresses_model_request_and_response_bodies(capsys, restore_logging):
    configure_logging("DEBUG")
    for name in ("openai._base_client", "httpx", "httpcore", "httpx2"):
        logging.getLogger(name).debug("private model request body")
    assert "private model request body" not in capsys.readouterr().err
