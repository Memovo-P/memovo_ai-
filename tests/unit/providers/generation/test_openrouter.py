"""The OpenRouter adapter against a stub transport. No socket is opened."""

import json
from collections.abc import Mapping
from typing import Any

import pytest
from pydantic import SecretStr

from memovo_ai.core.config import DEFAULT_GENERATION_MODEL, OpenRouterSettings
from memovo_ai.generation import (
    GenerationMessage,
    GenerationProvider,
    GenerationRateLimitedError,
    GenerationRequest,
    GenerationTimeoutError,
    GenerationUnavailableError,
    InvalidGenerationOutputError,
)
from memovo_ai.providers.generation import (
    HttpResponse,
    OpenRouterGenerationProvider,
    TransportError,
    TransportTimeoutError,
)
from memovo_ai.providers.generation.openrouter import build_payload

pytestmark = pytest.mark.unit

KEY = "sk-or-v1-ZZQ-test-key-never-logged"
PRIVATE_PROMPT = "ZZQ-what-did-the-cardiologist-say"


def settings(**overrides: object) -> OpenRouterSettings:
    base: dict[str, object] = {"api_key": SecretStr(KEY), "base_url": "https://router.example/v1"}
    return OpenRouterSettings(_env_file=None, **{**base, **overrides})  # type: ignore[arg-type]


def ok_body(text: str = '{"ok": true}', **extra: object) -> bytes:
    document: dict[str, Any] = {
        "id": "gen-1",
        "model": DEFAULT_GENERATION_MODEL,
        "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 4},
    }
    document.update(extra)
    return json.dumps(document).encode()


class StubTransport:
    """Records the one call it receives and answers as scripted."""

    def __init__(self, response: HttpResponse | BaseException | None = None) -> None:
        self.response = response if response is not None else HttpResponse(200, {}, ok_body())
        self.calls: list[dict[str, object]] = []
        self.closed = False

    async def post_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_seconds: float,
    ) -> HttpResponse:
        self.calls.append(
            {
                "url": url,
                "headers": dict(headers),
                "payload": dict(payload),
                "timeout": timeout_seconds,
            }
        )
        if isinstance(self.response, BaseException):
            raise self.response
        return self.response

    async def aclose(self) -> None:
        self.closed = True


def provider(
    transport: StubTransport, *, model: str = DEFAULT_GENERATION_MODEL
) -> OpenRouterGenerationProvider:
    return OpenRouterGenerationProvider(
        transport=transport,
        settings=settings(),
        model=model,
        timeout_seconds=12.5,
        retry_after_max_seconds=30,
    )


def request(text: str = "hello") -> GenerationRequest:
    return GenerationRequest(
        messages=(
            GenerationMessage(role="system", content="Answer in JSON."),
            GenerationMessage(role="user", content=text),
        ),
        max_output_tokens=256,
    )


# --------------------------------------------------------------------------
# The request that goes out
# --------------------------------------------------------------------------
def test_the_adapter_is_a_provider() -> None:
    assert isinstance(provider(StubTransport()), GenerationProvider)


async def test_the_configured_model_is_requested_exactly() -> None:
    transport = StubTransport()

    await provider(transport).generate(request())

    assert transport.calls[0]["payload"]["model"] == "nvidia/nemotron-3-super-120b-a12b:free"  # type: ignore[index]


async def test_no_fallback_or_routing_parameters_are_sent() -> None:
    """No paid/alternate model, no ``models`` fallback list, no routing hints."""
    transport = StubTransport()

    await provider(transport).generate(request())

    payload = transport.calls[0]["payload"]
    assert set(payload) == {"model", "messages", "max_tokens", "temperature", "stream"}  # type: ignore[arg-type]
    assert payload["stream"] is False  # type: ignore[index]


def test_the_payload_carries_the_messages_and_budget() -> None:
    payload = build_payload(request("q"), model="m")

    assert payload["messages"] == [
        {"role": "system", "content": "Answer in JSON."},
        {"role": "user", "content": "q"},
    ]
    assert payload["max_tokens"] == 256
    assert payload["temperature"] == 0.0


async def test_the_endpoint_is_chat_completions_under_the_base_url() -> None:
    transport = StubTransport()

    await provider(transport).generate(request())

    assert transport.calls[0]["url"] == "https://router.example/v1/chat/completions"


async def test_a_trailing_slash_on_the_base_url_is_tolerated() -> None:
    transport = StubTransport()
    adapter = OpenRouterGenerationProvider(
        transport=transport,
        settings=settings(base_url="https://router.example/v1/"),
        model="m",
        timeout_seconds=1,
        retry_after_max_seconds=1,
    )

    await adapter.generate(request())

    assert transport.calls[0]["url"] == "https://router.example/v1/chat/completions"


async def test_the_key_travels_only_in_the_authorization_header() -> None:
    transport = StubTransport()

    await provider(transport).generate(request())

    call = transport.calls[0]
    assert call["headers"]["Authorization"] == f"Bearer {KEY}"  # type: ignore[index]
    assert KEY not in json.dumps(call["payload"])
    assert KEY not in call["url"]  # type: ignore[operator]


async def test_the_timeout_is_forwarded_to_the_transport() -> None:
    transport = StubTransport()

    await provider(transport).generate(request())

    assert transport.calls[0]["timeout"] == 12.5


async def test_exactly_one_attempt_is_made_on_failure() -> None:
    transport = StubTransport(HttpResponse(503, {}, b"down"))

    with pytest.raises(GenerationUnavailableError):
        await provider(transport).generate(request())

    assert len(transport.calls) == 1


def test_an_empty_model_is_refused() -> None:
    with pytest.raises(ValueError, match="model"):
        provider(StubTransport(), model="  ")


# --------------------------------------------------------------------------
# Successful responses
# --------------------------------------------------------------------------
async def test_a_success_yields_text_finish_reason_and_usage() -> None:
    result = await provider(StubTransport()).generate(request())

    assert result.text == '{"ok": true}'
    assert result.finish_reason == "stop"
    assert result.model == DEFAULT_GENERATION_MODEL
    assert result.prompt_tokens == 12
    assert result.completion_tokens == 4


async def test_a_separate_reasoning_field_is_dropped() -> None:
    body = json.loads(ok_body())
    body["choices"][0]["message"]["reasoning"] = "ZZQ-hidden-chain-of-thought"

    result = await provider(
        StubTransport(HttpResponse(200, {}, json.dumps(body).encode()))
    ).generate(request())

    assert "ZZQ-hidden" not in result.text
    assert not hasattr(result, "reasoning")


async def test_missing_usage_yields_none_counts() -> None:
    body = json.loads(ok_body())
    del body["usage"]

    result = await provider(
        StubTransport(HttpResponse(200, {}, json.dumps(body).encode()))
    ).generate(request())

    assert result.prompt_tokens is None
    assert result.completion_tokens is None


async def test_a_length_finish_reason_is_surfaced_for_validation() -> None:
    body = json.loads(ok_body())
    body["choices"][0]["finish_reason"] = "length"

    result = await provider(
        StubTransport(HttpResponse(200, {}, json.dumps(body).encode()))
    ).generate(request())

    assert result.finish_reason == "length"


# --------------------------------------------------------------------------
# Failure mapping
# --------------------------------------------------------------------------
async def test_429_is_rate_limited_with_a_bounded_retry_after() -> None:
    transport = StubTransport(HttpResponse(429, {"retry-after": "120"}, b""))

    with pytest.raises(GenerationRateLimitedError) as raised:
        await provider(transport).generate(request())

    assert raised.value.retry_after_seconds == 30


async def test_429_without_retry_after_carries_none() -> None:
    with pytest.raises(GenerationRateLimitedError) as raised:
        await provider(StubTransport(HttpResponse(429, {}, b""))).generate(request())

    assert raised.value.retry_after_seconds is None


@pytest.mark.parametrize("value", ["7", "7.9", " 3 "])
def test_a_small_retry_after_is_kept(value: str) -> None:
    from memovo_ai.providers.generation.openrouter import _retry_after_seconds

    assert _retry_after_seconds({"retry-after": value}, cap=30) == int(float(value))


def test_an_http_date_retry_after_is_ignored_not_parsed() -> None:
    from memovo_ai.providers.generation.openrouter import _retry_after_seconds

    assert _retry_after_seconds({"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"}, cap=30) is None


@pytest.mark.parametrize("status", [500, 502, 503, 504, 529])
async def test_upstream_5xx_is_unavailable(status: int) -> None:
    with pytest.raises(GenerationUnavailableError):
        await provider(StubTransport(HttpResponse(status, {}, b"x"))).generate(request())


@pytest.mark.parametrize("status", [401, 402, 403, 404])
async def test_credentials_credits_and_unknown_model_are_unavailable(status: int) -> None:
    """None of these is the caller's fault and none is malformed output."""
    with pytest.raises(GenerationUnavailableError):
        await provider(StubTransport(HttpResponse(status, {}, b"x"))).generate(request())


@pytest.mark.parametrize("status", [400, 422])
async def test_a_rejected_request_is_unavailable_not_invalid_response(status: int) -> None:
    with pytest.raises(GenerationUnavailableError):
        await provider(StubTransport(HttpResponse(status, {}, b"x"))).generate(request())


async def test_a_transport_timeout_is_a_generation_timeout() -> None:
    with pytest.raises(GenerationTimeoutError):
        await provider(StubTransport(TransportTimeoutError("slow"))).generate(request())


async def test_a_connection_failure_is_unavailable() -> None:
    with pytest.raises(GenerationUnavailableError):
        await provider(StubTransport(TransportError("refused"))).generate(request())


async def test_an_error_object_in_a_200_is_unavailable() -> None:
    body = json.dumps({"error": {"code": 502, "message": "upstream"}}).encode()

    with pytest.raises(GenerationUnavailableError):
        await provider(StubTransport(HttpResponse(200, {}, body))).generate(request())


async def test_an_error_object_reporting_429_is_rate_limited() -> None:
    body = json.dumps({"error": {"code": 429, "message": "slow"}}).encode()

    with pytest.raises(GenerationRateLimitedError):
        await provider(StubTransport(HttpResponse(200, {}, body))).generate(request())


@pytest.mark.parametrize(
    "body",
    [
        b"not json",
        b"[]",
        json.dumps({"choices": []}).encode(),
        json.dumps({"choices": [{}]}).encode(),
        json.dumps({"choices": [{"message": {"content": ""}}]}).encode(),
        json.dumps({"choices": [{"message": {"content": None}}]}).encode(),
        b"\xff\xfe",
    ],
)
async def test_an_unusable_200_body_is_invalid_output(body: bytes) -> None:
    with pytest.raises(InvalidGenerationOutputError):
        await provider(StubTransport(HttpResponse(200, {}, body))).generate(request())


# --------------------------------------------------------------------------
# Nothing sensitive escapes
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "response",
    [
        HttpResponse(500, {}, f"error while handling {PRIVATE_PROMPT}".encode()),
        HttpResponse(400, {}, f"bad prompt: {PRIVATE_PROMPT}".encode()),
        HttpResponse(200, {}, f"{{oops {PRIVATE_PROMPT}".encode()),
        TransportError(f"failed to connect while sending {PRIVATE_PROMPT} with {KEY}"),
    ],
)
async def test_no_prompt_body_or_key_reaches_an_exception_message(
    response: HttpResponse | BaseException,
) -> None:
    with pytest.raises(Exception) as raised:
        await provider(StubTransport(response)).generate(request(PRIVATE_PROMPT))

    assert PRIVATE_PROMPT not in str(raised.value)
    assert KEY not in str(raised.value)


def test_the_settings_object_never_renders_the_key() -> None:
    rendered = f"{settings()!r} {settings()!s} {settings().model_dump()}"

    assert KEY not in rendered


def test_the_adapter_exposes_no_key_attribute() -> None:
    adapter = provider(StubTransport())

    assert KEY not in repr(adapter)
    assert not any(
        name for name in dir(adapter) if not name.startswith("_") and name in {"api_key", "key"}
    )


# --------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------
def test_connect_refuses_to_start_without_a_key() -> None:
    with pytest.raises(GenerationUnavailableError, match="MEMOVO_OPENROUTER_API_KEY"):
        OpenRouterGenerationProvider.connect(
            settings=settings(api_key=SecretStr("")),
            model="m",
            timeout_seconds=1,
            retry_after_max_seconds=1,
        )


async def test_closing_the_adapter_closes_the_transport() -> None:
    transport = StubTransport()

    await provider(transport).aclose()

    assert transport.closed is True
