"""Exercise the real SDK over an in-memory HTTP transport, without credentials."""

import asyncio
import json
import logging
import traceback

import httpx
import pytest
from openai import AsyncOpenAI

from app.services.llm import LLMError, OpenAICompatibleLLM

TEST_KEY = "fake-unit-test-key"
MESSAGES = [{"role": "system", "content": "Be concise."}, {"role": "user", "content": "你好"}]


def completion(content="你好！", *, finish_reason="stop", **message_fields):
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1,
        "model": "deepseek-flash",
        "choices": [
            {
                "index": 0,
                "finish_reason": finish_reason,
                "message": {"role": "assistant", "content": content, **message_fields},
            }
        ],
    }


@pytest.fixture
async def make_llm():
    services = []

    def factory(handler, *, sdk_retries=0, strict=False, **kwargs):
        http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        sdk = AsyncOpenAI(
            api_key=TEST_KEY,
            base_url="https://llm.invalid/v1",
            http_client=http_client,
            max_retries=sdk_retries,
            _strict_response_validation=strict,
        )
        service = OpenAICompatibleLLM(
            TEST_KEY, "https://llm.invalid/v1", "deepseek-flash", client=sdk, **kwargs
        )
        services.append(service)
        return service, sdk, http_client

    yield factory
    for service in services:
        await service.close()


async def test_sdk_sends_auth_messages_and_top_level_thinking(make_llm):
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(
            200, json=completion("  你好！  ", reasoning_content="private reasoning")
        )

    service, _, _ = make_llm(handle, thinking="disabled", max_tokens=512)
    assert await service.chat(MESSAGES) == "你好！"
    assert len(requests) == 1
    request = requests[0]
    assert request.method == "POST"
    assert str(request.url) == "https://llm.invalid/v1/chat/completions"
    assert request.headers["Authorization"] == f"Bearer {TEST_KEY}"
    assert json.loads(request.content) == {
        "model": "deepseek-flash",
        "messages": MESSAGES,
        "stream": False,
        "max_tokens": 512,
        "thinking": {"type": "disabled"},
    }


async def test_generic_provider_request_omits_thinking_when_unset(make_llm):
    def handle(request):
        body = json.loads(request.content)
        assert "thinking" not in body
        assert "extra_body" not in body
        assert body["max_tokens"] == 1024
        return httpx.Response(200, json=completion())

    service, _, _ = make_llm(handle)
    assert await service.chat(MESSAGES) == "你好！"


async def test_chat_decisions_request_json_but_assistant_chat_stays_plain(make_llm):
    requests = []

    def handle(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=completion('{"should_reply":true,"reply":"接梗"}'))

    service, _, _ = make_llm(handle, thinking="disabled")
    await service.chat_json([{"role": "user", "content": "Return a JSON decision."}])
    await service.chat(MESSAGES)
    assert requests[0]["response_format"] == {"type": "json_object"}
    assert requests[0]["thinking"] == {"type": "disabled"}
    assert "response_format" not in requests[1]


async def test_json_chat_keeps_safe_errors_and_does_not_retry_as_plain_text(make_llm):
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(400, json={"error": {"message": f"provider detail {TEST_KEY}"}})

    service, _, _ = make_llm(handle)
    with pytest.raises(LLMError) as error:
        await service.chat_json(MESSAGES)
    assert error.value.kind == "api"
    assert TEST_KEY not in str(error.value)
    assert len(requests) == 1


async def test_model_list_uses_provider_auth_and_caches_without_chat_request(make_llm):
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {"id": name, "object": "model", "created": 1, "owned_by": "test"}
                    for name in ["model-b", "model-a", "model-b", "bad\nname"]
                ],
            },
        )

    service, _, _ = make_llm(handle)
    first = await service.list_models()
    assert first == ["model-a", "model-b"]
    first.append("must-not-mutate-cache")
    assert await service.list_models() == ["model-a", "model-b"]
    assert len(requests) == 1
    assert requests[0].method == "GET"
    assert requests[0].url.path == "/v1/models"
    assert requests[0].headers["Authorization"] == f"Bearer {TEST_KEY}"


async def test_model_list_sanitizes_provider_error(make_llm):
    service, _, _ = make_llm(
        lambda request: httpx.Response(401, json={"error": {"message": TEST_KEY}})
    )
    with pytest.raises(LLMError) as error:
        await service.list_models()
    assert error.value.kind == "auth"
    assert TEST_KEY not in str(error.value)


async def test_constructs_sdk_with_configured_timeout_and_retries_disabled(monkeypatch):
    from app.services import llm

    observed = {}

    def constructor(**kwargs):
        observed.update(kwargs)
        return AsyncOpenAI(
            **kwargs,
            http_client=httpx.AsyncClient(
                transport=httpx.MockTransport(lambda _: httpx.Response(200, json=completion()))
            ),
        )

    monkeypatch.setattr(llm, "AsyncOpenAI", constructor)
    service = OpenAICompatibleLLM(
        TEST_KEY, "https://llm.invalid/v1", "deepseek-flash", timeout_seconds=12
    )
    try:
        assert observed == {
            "api_key": TEST_KEY,
            "base_url": "https://llm.invalid/v1",
            "timeout": 12,
            "max_retries": 0,
        }
        assert await service.chat(MESSAGES) == "你好！"
    finally:
        await service.close()


@pytest.mark.parametrize(
    ("status", "kind"),
    [
        (400, "api"),
        (401, "auth"),
        (402, "balance"),
        (422, "api"),
        (429, "rate_limit"),
        (500, "api"),
        (503, "api"),
    ],
)
async def test_http_errors_are_safe_and_never_retry_even_injected_sdk(
    make_llm, caplog, status, kind
):
    calls = 0
    secret_body = f"provider-private-message {TEST_KEY} chat-body"

    def handle(_):
        nonlocal calls
        calls += 1
        return httpx.Response(status, json={"error": {"message": secret_body}})

    service, _, _ = make_llm(handle, sdk_retries=2)
    with caplog.at_level(logging.INFO, logger="app.services.llm"):
        with pytest.raises(LLMError) as caught:
            await service.chat(MESSAGES)
    assert caught.value.kind == kind
    assert caught.value.status_code == status
    assert calls == 1
    assert f"kind={kind} status_code={status}" in caplog.text
    assert "elapsed_ms=" in caplog.text
    rendered_error = "".join(traceback.format_exception(caught.value))
    for private in (secret_body, TEST_KEY, "chat-body"):
        assert private not in str(caught.value)
        assert private not in rendered_error
        assert private not in caplog.text


@pytest.mark.parametrize("error_cls", [httpx.ReadTimeout, httpx.ConnectTimeout])
async def test_sdk_timeout_is_mapped_without_retry(make_llm, error_cls):
    calls = 0

    def handle(request):
        nonlocal calls
        calls += 1
        raise error_cls("secret transport details", request=request)

    service, _, _ = make_llm(handle)
    with pytest.raises(LLMError) as caught:
        await service.chat(MESSAGES)
    assert caught.value.kind == "timeout"
    assert caught.value.status_code is None
    assert "secret" not in str(caught.value)
    assert calls == 1


async def test_sdk_connection_failure_is_mapped(make_llm):
    def handle(request):
        raise httpx.ConnectError("secret transport details", request=request)

    service, _, _ = make_llm(handle)
    with pytest.raises(LLMError) as caught:
        await service.chat(MESSAGES)
    assert caught.value.kind == "connection"
    assert caught.value.status_code is None
    assert "secret" not in str(caught.value)


@pytest.mark.parametrize("content", [None, "", " \t\n"])
async def test_empty_content_does_not_fall_back_to_reasoning(make_llm, content):
    service, _, _ = make_llm(
        lambda _: httpx.Response(
            200, json=completion(content, reasoning_content="must never be returned")
        )
    )
    with pytest.raises(LLMError) as caught:
        await service.chat(MESSAGES)
    assert caught.value.kind == "empty"
    assert "must never be returned" not in str(caught.value)


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"choices": None},
        {"choices": []},
        {"choices": [{}]},
        {"choices": [{"message": {}}]},
        completion(123),
        completion(["not", "a string"]),
    ],
)
async def test_malformed_shapes_and_non_string_content_are_rejected(make_llm, payload):
    service, _, _ = make_llm(lambda _: httpx.Response(200, json=payload))
    with pytest.raises(LLMError) as caught:
        await service.chat(MESSAGES)
    assert caught.value.kind == "malformed"


async def test_invalid_json_response_is_malformed(make_llm):
    service, _, _ = make_llm(
        lambda _: httpx.Response(
            200, text="{not JSON", headers={"content-type": "application/json"}
        )
    )
    with pytest.raises(LLMError) as caught:
        await service.chat(MESSAGES)
    assert caught.value.kind == "malformed"


async def test_sdk_response_validation_error_is_malformed(make_llm):
    service, _, _ = make_llm(lambda _: httpx.Response(200, json={}), strict=True)
    with pytest.raises(LLMError) as caught:
        await service.chat(MESSAGES)
    assert caught.value.kind == "malformed"


async def test_unknown_sdk_error_is_sanitized(make_llm, monkeypatch):
    service, _, _ = make_llm(lambda _: httpx.Response(200, json=completion()))

    async def fail(**_):
        raise RuntimeError(f"private SDK failure {TEST_KEY}")

    monkeypatch.setattr(service._client.chat.completions, "create", fail)
    with pytest.raises(LLMError) as caught:
        await service.chat(MESSAGES)
    assert caught.value.kind == "unknown"
    assert "private" not in str(caught.value)
    assert TEST_KEY not in str(caught.value)


async def test_truncated_output_keeps_text_and_appends_notice(make_llm):
    service, _, _ = make_llm(
        lambda _: httpx.Response(200, json=completion("部分回答", finish_reason="length"))
    )
    assert await service.chat(MESSAGES) == "部分回答\n\n（回复达到长度上限，已截断。）"


async def test_deadline_cancels_blocked_request_and_releases_capacity(make_llm):
    cancelled = asyncio.Event()
    calls = 0

    async def handle(_):
        nonlocal calls
        calls += 1
        if calls == 1:
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
        return httpx.Response(200, json=completion())

    service, _, _ = make_llm(handle, timeout_seconds=0.05, max_concurrency=1)
    async with asyncio.timeout(2):
        with pytest.raises(LLMError) as caught:
            await service.chat(MESSAGES)
        assert caught.value.kind == "timeout"
        assert cancelled.is_set()
        assert await service.chat(MESSAGES) == "你好！"


async def test_deadline_includes_waiting_for_capacity(make_llm):
    calls = 0

    def handle(_):
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=completion())

    service, _, _ = make_llm(handle, timeout_seconds=0.02, max_concurrency=1)
    await service._semaphore.acquire()
    try:
        async with asyncio.timeout(2):
            with pytest.raises(LLMError) as caught:
                await service.chat(MESSAGES)
        assert caught.value.kind == "timeout"
        assert calls == 0
    finally:
        service._semaphore.release()
    assert await service.chat(MESSAGES) == "你好！"


async def test_empty_line_keepalives_cannot_extend_total_deadline(make_llm):
    closed = asyncio.Event()

    class KeepAliveStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            while True:
                await asyncio.sleep(0.005)
                yield b"\n"

        async def aclose(self):
            closed.set()

    service, _, _ = make_llm(
        lambda _: httpx.Response(200, stream=KeepAliveStream()), timeout_seconds=0.05
    )
    async with asyncio.timeout(2):
        with pytest.raises(LLMError) as caught:
            await service.chat(MESSAGES)
    assert caught.value.kind == "timeout"
    assert closed.is_set()


async def test_all_chat_calls_share_the_concurrency_limit(make_llm):
    active = maximum = calls = 0
    full = asyncio.Event()
    release = asyncio.Event()

    async def handle(_):
        nonlocal active, maximum, calls
        calls += 1
        active += 1
        maximum = max(maximum, active)
        if active == 2:
            full.set()
        try:
            await release.wait()
            return httpx.Response(200, json=completion())
        finally:
            active -= 1

    service, _, _ = make_llm(handle, max_concurrency=2)
    tasks = [asyncio.create_task(service.chat(MESSAGES)) for _ in range(6)]
    try:
        async with asyncio.timeout(3):
            await full.wait()
            assert calls == 2
            release.set()
            assert await asyncio.gather(*tasks) == ["你好！"] * 6
        assert maximum == 2
        assert calls == 6
        assert active == 0
    finally:
        release.set()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def test_cancellation_propagates_and_releases_active_and_queued_slots(make_llm):
    entered = asyncio.Event()
    calls = 0

    async def handle(_):
        nonlocal calls
        calls += 1
        if calls == 1:
            entered.set()
            await asyncio.Event().wait()
        return httpx.Response(200, json=completion())

    service, _, _ = make_llm(handle, max_concurrency=1)
    active = asyncio.create_task(service.chat(MESSAGES))
    queued = None
    try:
        async with asyncio.timeout(3):
            await entered.wait()
            queued = asyncio.create_task(service.chat(MESSAGES))
            await asyncio.sleep(0)
            queued.cancel()
            with pytest.raises(asyncio.CancelledError):
                await queued
            active.cancel()
            with pytest.raises(asyncio.CancelledError):
                await active
            assert await service.chat(MESSAGES) == "你好！"
            assert calls == 2
    finally:
        tasks = [active] + ([queued] if queued else [])
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def test_close_releases_injected_sdk_http_client_and_is_repeatable(make_llm):
    service, sdk, http_client = make_llm(lambda _: httpx.Response(200, json=completion()))
    await service.close()
    assert sdk.is_closed()
    assert http_client.is_closed
    await service.close()


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan"), True])
def test_invalid_deadline_is_rejected(timeout):
    with pytest.raises(ValueError):
        OpenAICompatibleLLM(TEST_KEY, "https://llm.invalid", "test", timeout_seconds=timeout)


@pytest.mark.parametrize("field", ["max_concurrency", "max_tokens"])
@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_invalid_capacity_is_rejected(field, value):
    with pytest.raises(ValueError):
        OpenAICompatibleLLM(TEST_KEY, "https://llm.invalid", "test", **{field: value})


def test_invalid_thinking_mode_is_rejected():
    with pytest.raises(ValueError):
        OpenAICompatibleLLM(TEST_KEY, "https://llm.invalid", "test", thinking="invalid")
