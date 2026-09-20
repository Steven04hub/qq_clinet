import asyncio
import json
import logging

import httpx
import pytest

from app.services.jev import (
    DEFAULT_MODEL,
    ENDPOINT,
    MAX_RESPONSE_BYTES,
    QUESTION_IDS,
    JevClient,
    JevError,
    classify_scores,
)


def scores(**overrides):
    return dict.fromkeys(QUESTION_IDS, 0.0) | overrides


def payload(values=None):
    return {
        "model": DEFAULT_MODEL,
        "answers": {
            name: {"type": "noul", "noul": value}
            for name, value in (values or scores(reply_expected=0.95)).items()
        },
        "usage": {"input_tokens": 100, "output_tokens": 0},
    }


@pytest.mark.parametrize(
    ("values", "route", "reason"),
    [
        (scores(asked_to_pause=0.85), "silent", "pause_requested"),
        (scores(reply_expected=0.85), "reply", "reply_expected"),
        (
            scores(natural_opening=0.70, topic_exhausted=0.15, would_interrupt=0.15),
            "reply",
            "natural_opening",
        ),
        (scores(topic_exhausted=0.85), "silent", "topic_exhausted"),
        (scores(would_interrupt=0.85), "silent", "would_interrupt"),
        (scores(natural_opening=0.69), "fallback", "uncertain"),
        (scores(asked_to_pause=0.84), "fallback", "uncertain"),
        (scores(reply_expected=0.84), "fallback", "uncertain"),
        (scores(topic_exhausted=0.84), "fallback", "uncertain"),
        (
            scores(reply_expected=0.85, asked_to_pause=0.85),
            "fallback",
            "contradictory",
        ),
        (
            scores(reply_expected=0.85, topic_exhausted=0.85),
            "fallback",
            "contradictory",
        ),
        (
            scores(reply_expected=0.85, would_interrupt=0.85),
            "fallback",
            "contradictory",
        ),
        (
            scores(natural_opening=0.7, topic_exhausted=0.85),
            "fallback",
            "contradictory",
        ),
        (
            scores(natural_opening=0.7, would_interrupt=0.85),
            "fallback",
            "contradictory",
        ),
    ],
)
def test_classify_routes(values, route, reason):
    assert classify_scores(values) == (route, reason)


@pytest.mark.parametrize(
    "values", [scores(reply_expected=1), scores(asked_to_pause=1), scores(topic_exhausted=1)]
)
def test_unknown_media_always_uses_existing_chat_decision(values):
    assert classify_scores(values, unknown_media_present=True) == ("fallback", "unknown_media")


@pytest.mark.parametrize(
    "values", [scores(reply_expected=1), scores(asked_to_pause=1), scores(topic_exhausted=1)]
)
def test_incomplete_context_always_uses_existing_chat_decision(values):
    assert classify_scores(values, context_incomplete=True) == ("fallback", "context_incomplete")


@pytest.mark.parametrize(
    "value", [True, False, None, "0.1", -0.1, 1.1, float("nan"), float("inf"), 10**400]
)
def test_invalid_score_types_and_ranges(value):
    with pytest.raises(JevError, match="invalid response") as exc:
        classify_scores(scores(reply_expected=value))
    assert exc.value.kind == "malformed"


@pytest.mark.parametrize("values", [{}, {"reply_expected": 1}, scores(extra=0)])
def test_all_and_only_expected_questions_are_required(values):
    with pytest.raises(JevError, match="invalid response"):
        classify_scores(values)


async def test_success_request_and_structured_decision(caplog):
    requests = []

    async def respond(request):
        requests.append(request)
        return httpx.Response(200, json=payload())

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    client = JevClient("test-secret", client=http_client)
    state = {"bot_id": "bot", "new_messages": [{"text": "private-test-message"}]}
    with caplog.at_level(logging.INFO):
        decision = await client.evaluate(state)
    await client.close()
    assert http_client.is_closed
    assert decision.route == "reply"
    assert decision.reason == "reply_expected"
    assert decision.scores == scores(reply_expected=0.95)
    assert decision.input_tokens == 100
    assert decision.elapsed_ms >= 0
    assert len(requests) == 1
    request = requests[0]
    assert str(request.url) == ENDPOINT
    assert request.method == "POST"
    assert request.headers["Authorization"] == "Bearer test-secret"
    sent = json.loads(request.content)
    assert sent["state"] == state
    assert sent["model"] == DEFAULT_MODEL
    assert set(sent["questions"]) == set(QUESTION_IDS)
    assert all(question["type"] == "noul" for question in sent["questions"].values())
    assert "test-secret" not in caplog.text
    assert "private-test-message" not in caplog.text


@pytest.mark.parametrize(
    ("status", "kind"),
    [(401, "auth"), (402, "balance"), (429, "rate_limit"), (529, "rate_limit"), (500, "api")],
)
async def test_http_errors_are_sanitized_and_never_retried(status, kind, caplog):
    calls = 0

    async def respond(request):
        nonlocal calls
        calls += 1
        return httpx.Response(status, text="sensitive-provider-body test-secret private-message")

    client = JevClient(
        "test-secret", client=httpx.AsyncClient(transport=httpx.MockTransport(respond))
    )
    with caplog.at_level(logging.INFO), pytest.raises(JevError) as exc:
        await client.evaluate({"new_messages": [{"text": "private-message"}]})
    await client.close()
    assert calls == 1
    assert exc.value.kind == kind
    assert exc.value.status_code == status
    assert exc.value.__suppress_context__
    visible = str(exc.value) + repr(exc.value) + caplog.text
    assert "test-secret" not in visible
    assert "private-message" not in visible
    assert "sensitive-provider-body" not in visible


async def test_redirect_is_not_followed_even_if_injected_client_enables_it():
    requests = []

    async def respond(request):
        requests.append(request)
        return httpx.Response(302, headers={"location": "https://untrusted.invalid/collect"})

    client = JevClient(
        "test-secret",
        client=httpx.AsyncClient(transport=httpx.MockTransport(respond), follow_redirects=True),
    )
    with pytest.raises(JevError) as exc:
        await client.evaluate({})
    await client.close()
    assert exc.value.status_code == 302
    assert len(requests) == 1


@pytest.mark.parametrize(
    "response",
    [
        [],
        {},
        {"answers": []},
        {"answers": {}},
        {"answers": {name: {"type": "choice", "noul": 0} for name in QUESTION_IDS}},
        {"answers": {name: {"type": "noul"} for name in QUESTION_IDS}},
        {"answers": {name: {"type": "noul", "noul": True} for name in QUESTION_IDS}},
        payload() | {"usage": {"input_tokens": True}},
        payload() | {"usage": {"input_tokens": -1}},
        payload() | {"usage": "secret-malformed-usage"},
    ],
)
async def test_malformed_response(response):
    client = JevClient(
        "test-secret",
        client=httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json=response))
        ),
    )
    with pytest.raises(JevError) as exc:
        await client.evaluate({})
    await client.close()
    assert exc.value.kind == "malformed"


@pytest.mark.parametrize("body", [b"not json", b"\xff", b" " * (MAX_RESPONSE_BYTES + 1)])
async def test_invalid_or_oversized_body(body):
    client = JevClient(
        "test-secret",
        client=httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, content=body))
        ),
    )
    with pytest.raises(JevError) as exc:
        await client.evaluate({})
    await client.close()
    assert exc.value.kind == "malformed"


@pytest.mark.parametrize("flag", ["unknown_media_present", "context_incomplete"])
async def test_optional_usage_and_incomplete_state(flag):
    response = payload()
    response.pop("usage")
    client = JevClient(
        "test-secret",
        client=httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json=response))
        ),
    )
    decision = await client.evaluate({flag: True})
    await client.close()
    assert decision.input_tokens is None
    assert decision.route == "fallback"
    assert decision.reason == ("unknown_media" if flag == "unknown_media_present" else flag)


async def test_hard_timeout_includes_request():
    cancelled = asyncio.Event()

    async def respond(request):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    client = JevClient(
        "test-secret",
        timeout_seconds=0.01,
        client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    )
    with pytest.raises(JevError) as exc:
        await client.evaluate({})
    await client.close()
    assert exc.value.kind == "timeout"
    assert cancelled.is_set()


async def test_hard_timeout_includes_semaphore_queue():
    calls = 0

    async def respond(request):
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=payload())

    client = JevClient(
        "test-secret",
        timeout_seconds=0.01,
        client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    )
    await client._semaphore.acquire()
    await client._semaphore.acquire()
    with pytest.raises(JevError) as exc:
        await client.evaluate({})
    client._semaphore.release()
    client._semaphore.release()
    assert exc.value.kind == "timeout"
    assert calls == 0
    assert (await client.evaluate({})).route == "reply"
    await client.close()


async def test_cancellation_propagates_and_releases_capacity():
    started = asyncio.Event()
    calls = 0

    async def respond(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            started.set()
            await asyncio.Event().wait()
        return httpx.Response(200, json=payload())

    client = JevClient(
        "test-secret", client=httpx.AsyncClient(transport=httpx.MockTransport(respond))
    )
    task = asyncio.create_task(client.evaluate({}))
    await asyncio.wait_for(started.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert client._semaphore._value == 2
    assert (await client.evaluate({})).route == "reply"
    await client.close()


async def test_transport_error_is_sanitized():
    async def respond(request):
        raise httpx.ConnectError("private-request test-secret", request=request)

    client = JevClient(
        "test-secret", client=httpx.AsyncClient(transport=httpx.MockTransport(respond))
    )
    with pytest.raises(JevError) as exc:
        await client.evaluate({})
    await client.close()
    assert exc.value.kind == "connection"
    assert "test-secret" not in str(exc.value)
    assert "private-request" not in str(exc.value)


@pytest.mark.parametrize("timeout", [True, 0, -1, float("inf"), float("nan"), "3"])
def test_timeout_validation(timeout):
    with pytest.raises(ValueError, match="timeout_seconds"):
        JevClient("test-secret", timeout_seconds=timeout)
