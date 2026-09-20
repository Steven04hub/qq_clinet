"""Exercise the web CLI with real local HTTP and WebSocket mock services."""

import asyncio
import json
import signal
import socket
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest
from websockets.asyncio.server import serve

from tests.test_runtime import (
    TEST_EVENT,
    TEST_GROUP_ID,
    TEST_TOKEN,
    assert_clean_output,
    cleanup_process,
    server_url,
    start_cli,
)


@pytest.fixture
def web_model_server():
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def respond(self, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            requests.append(("GET", self.path, self.headers.get("Authorization"), None))
            self.respond(
                {
                    "object": "list",
                    "data": [
                        {"id": name, "object": "model", "created": 1, "owned_by": "test"}
                        for name in ("test-model", "new-model")
                    ],
                }
            )

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(("POST", self.path, self.headers.get("Authorization"), body))
            self.respond(
                {
                    "id": "local-test-completion",
                    "object": "chat.completion",
                    "created": 1,
                    "model": body["model"],
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {
                                "role": "assistant",
                                "content": json.dumps(
                                    {"should_reply": True, "reply": "测试主动回复"}
                                ),
                            },
                        }
                    ],
                }
            )

        def log_message(self, *_: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02})
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def unused_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


async def wait_status(client: httpx.AsyncClient, predicate: Callable[[dict], bool]) -> dict:
    async with asyncio.timeout(8):
        while True:
            try:
                response = await client.get("/api/status")
            except httpx.TransportError:
                await asyncio.sleep(0.05)
                continue
            assert response.status_code == 200, response.text
            state = response.json()
            if predicate(state):
                return state
            await asyncio.sleep(0.05)


def event(message_id: int, text: str, *, group_id: int = TEST_GROUP_ID) -> dict:
    return {
        **TEST_EVENT,
        "message_id": message_id,
        "group_id": group_id,
        "message": [{"type": "text", "data": {"text": text}}],
    }


async def test_web_cli_controls_chat_model_filter_budget_and_pause(
    tmp_path: Path, web_model_server
) -> None:
    model_url, model_requests = web_model_server
    connected = asyncio.get_running_loop().create_future()
    actions = asyncio.Queue()

    async def handle_ws(connection) -> None:
        assert connection.request.headers["Authorization"] == f"Bearer {TEST_TOKEN}"
        connected.set_result(connection)
        async for packet in connection:
            request = json.loads(packet)
            await connection.send(
                json.dumps(
                    {
                        "status": "ok",
                        "retcode": 0,
                        "echo": request["echo"],
                        "data": {"message_id": 999},
                    }
                )
            )
            await actions.put(request)

    async with asyncio.timeout(20):
        async with serve(handle_ws, "127.0.0.1", 0) as ws_server:
            port = unused_port()
            state_file = tmp_path / "control.json"
            process = await start_cli(
                tmp_path,
                server_url(ws_server),
                "web",
                "--port",
                str(port),
                "--state-file",
                str(state_file),
                overrides={
                    "LLM_API_KEY": "web-model-test-key",
                    "LLM_BASE_URL": model_url,
                    "LLM_MODEL": "test-model",
                    "LLM_THINKING": "disabled",
                    "PROFILE_AUTO_IMPORT": "false",
                },
            )
            try:
                async with httpx.AsyncClient(
                    base_url=f"http://127.0.0.1:{port}", timeout=2, trust_env=False
                ) as browser:
                    initial = await wait_status(browser, lambda state: state["connected"])
                    connection = await connected
                    assert initial["settings"]["mode"] == "assistant"
                    assert initial["settings"]["target_group_id"] == str(TEST_GROUP_ID)
                    assert initial["settings"]["enabled"] is True
                    assert initial["model_configured"] is True
                    assert initial["replies_last_minute"] == 0
                    page = await browser.get("/")
                    assert page.status_code == 200
                    assert "text/html" in page.headers["content-type"]

                    models = await browser.get("/api/models")
                    assert models.json() == {
                        "models": ["new-model", "test-model"],
                        "error": None,
                    }
                    assert model_requests == [
                        ("GET", "/v1/models", "Bearer web-model-test-key", None)
                    ]
                    settings = {
                        **initial["settings"],
                        "mode": "chat",
                        "model": "new-model",
                        "reply_limit_per_minute": 10,
                    }
                    response = await browser.put(
                        "/api/settings", json=settings, headers={"X-QQ-Control": "1"}
                    )
                    assert response.status_code == 200, response.text
                    assert response.json()["settings"] == settings
                    assert json.loads(state_file.read_text()) == settings

                    # Unrelated groups must not schedule a model decision or enter
                    # visible activity/history.
                    other_text = "unrelated-group-private-content"
                    await connection.send(
                        json.dumps(event(800, other_text, group_id=TEST_GROUP_ID + 1))
                    )
                    await asyncio.sleep(1.7)
                    unrelated_state = (await browser.get("/api/status")).json()
                    assert unrelated_state["context_messages"] == 0
                    assert other_text not in json.dumps(unrelated_state)
                    assert len(model_requests) == 1
                    assert actions.empty()

                    # No @ segment: chat mode chooses to reply on its own.
                    message_received_at = asyncio.get_running_loop().time()
                    await connection.send(json.dumps(event(801, "今天这个段子有点好笑")))
                    request = await actions.get()
                    assert asyncio.get_running_loop().time() - message_received_at >= 4.9
                    assert request["action"] == "send_group_msg"
                    assert request["params"] == {
                        "group_id": TEST_GROUP_ID,
                        "message": [{"type": "text", "data": {"text": "测试主动回复"}}],
                    }
                    assert len(model_requests) == 2
                    method, path, auth, body = model_requests[-1]
                    assert (method, path, auth) == (
                        "POST",
                        "/v1/chat/completions",
                        "Bearer web-model-test-key",
                    )
                    assert body["model"] == "new-model"
                    assert body["response_format"] == {"type": "json_object"}
                    assert body["thinking"] == {"type": "disabled"}
                    assert body["messages"][0]["role"] == "system"
                    assert "今天这个段子有点好笑" in body["messages"][-1]["content"]
                    assert other_text not in json.dumps(body)
                    await wait_status(browser, lambda state: state["context_messages"] == 2)

                    # Lowering the limit must retain the reply already sent in
                    # the rolling window, and skip the next model call entirely.
                    settings["reply_limit_per_minute"] = 1
                    limited = await browser.put(
                        "/api/settings", json=settings, headers={"X-QQ-Control": "1"}
                    )
                    assert limited.status_code == 200
                    assert limited.json()["replies_last_minute"] == 1
                    await connection.send(json.dumps(event(802, "再接一个梗吧")))
                    await wait_status(
                        browser,
                        lambda state: any(item["kind"] == "limited" for item in state["activity"]),
                    )
                    assert len(model_requests) == 2
                    assert actions.empty()

                    settings["enabled"] = False
                    paused = await browser.put(
                        "/api/settings", json=settings, headers={"X-QQ-Control": "1"}
                    )
                    assert paused.status_code == 200
                    paused_state = paused.json()
                    paused_text = "ignored-while-paused"
                    await connection.send(json.dumps(event(803, paused_text)))
                    await asyncio.sleep(0.2)
                    after_pause = (await browser.get("/api/status")).json()
                    assert after_pause["settings"]["enabled"] is False
                    assert after_pause["context_messages"] == paused_state["context_messages"]
                    assert paused_text not in json.dumps(after_pause)
                    assert len(model_requests) == 2
                    assert actions.empty()
                    assert "web-model-test-key" not in json.dumps(after_pause)

                process.send_signal(signal.SIGTERM)
                raw_output, _ = await process.communicate()
                output = raw_output.decode()
                # Uvicorn may re-raise the captured signal after its complete
                # lifespan cleanup; both exit statuses represent a clean stop.
                assert process.returncode in (0, -signal.SIGTERM), output
                assert "Application shutdown complete" in output
                assert "NapCat disconnected" in output
                assert "web-model-test-key" not in output
                assert "unrelated-group-private-content" not in output
                assert "ignored-while-paused" not in output
                assert_clean_output(output)
            finally:
                await cleanup_process(process)
