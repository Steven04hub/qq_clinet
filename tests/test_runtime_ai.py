"""Exercise the CLI across real local OneBot and model HTTP protocols."""

import asyncio
import json
import signal
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from websockets.asyncio.server import serve

from tests.test_runtime import (
    TEST_EVENT,
    TEST_GROUP_ID,
    assert_clean_output,
    cleanup_process,
    server_url,
    start_cli,
)


@pytest.fixture
def model_server():
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, self.headers.get("Authorization"), body))
            response = json.dumps(
                {
                    "id": "test-completion",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "test-model",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": "你好，这是模拟模型回答。"},
                        }
                    ],
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, *args):
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


async def test_cli_run_calls_model_and_replies_once_for_duplicate_event(tmp_path, model_server):
    base_url, model_requests = model_server
    received = asyncio.Queue()

    async def handle_ws(connection):
        await connection.send(json.dumps(TEST_EVENT))  # Unmentioned chatter stays local.
        event = {
            **TEST_EVENT,
            "message_id": 790,
            "message": [
                {"type": "at", "data": {"qq": "111"}},
                {"type": "text", "data": {"text": " 你好"}},
            ],
        }
        for _ in range(2):
            await connection.send(json.dumps(event))
        request = json.loads(await connection.recv())
        await connection.send(
            json.dumps(
                {
                    "status": "ok",
                    "retcode": 0,
                    "echo": request["echo"],
                    "data": {"message_id": 901},
                }
            )
        )
        await received.put(request)
        await connection.wait_closed()

    async with asyncio.timeout(10):
        async with serve(handle_ws, "127.0.0.1", 0) as server:
            process = await start_cli(
                tmp_path,
                server_url(server),
                "run",
                overrides={
                    "LLM_API_KEY": "model-test-key",
                    "LLM_BASE_URL": base_url,
                    "LLM_MODEL": "test-model",
                    "LLM_THINKING": "disabled",
                },
            )
            try:
                request = await received.get()
                assert request["action"] == "send_group_msg"
                assert request["params"] == {
                    "group_id": TEST_GROUP_ID,
                    "message": [
                        {"type": "reply", "data": {"id": "790"}},
                        {"type": "text", "data": {"text": "你好，这是模拟模型回答。"}},
                    ],
                }
                assert len(model_requests) == 1
                path, authorization, body = model_requests[0]
                assert path == "/v1/chat/completions"
                assert authorization == "Bearer model-test-key"
                assert body["model"] == "test-model"
                assert body["thinking"] == {"type": "disabled"}
                assert body["messages"][0]["role"] == "system"
                assert body["messages"][-1]["content"].endswith("你好")
                assert len(body["messages"]) == 2
                process.send_signal(signal.SIGTERM)
                raw_output, _ = await process.communicate()
                output = raw_output.decode()
                assert process.returncode == 0, output
                assert "model-test-key" not in output
                assert_clean_output(output)
            finally:
                await cleanup_process(process)


async def test_cli_run_requires_explicit_model_credentials(tmp_path):
    process = await start_cli(tmp_path, "ws://127.0.0.1:1", "run")
    try:
        async with asyncio.timeout(5):
            raw_output, _ = await process.communicate()
        assert process.returncode == 2
        assert "LLM_API_KEY" in raw_output.decode()
        assert "NapCat connected" not in raw_output.decode()
        assert_clean_output(raw_output.decode())
    finally:
        await cleanup_process(process)
