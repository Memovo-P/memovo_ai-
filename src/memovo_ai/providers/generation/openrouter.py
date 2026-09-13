"""OpenRouter chat-completions adapter for the approved generation model.

Exactly one model, the one configured -- ``nvidia/nemotron-3-super-120b-a12b:free`` by
approval -- and exactly one attempt per call. No fallback model, no
``:free``-to-paid routing, no SDK retries: the Backend owns retries (contract
section 21.1), and a silent model switch would invalidate every evaluation
made against the approved one.

The HTTP client is behind :class:`HttpTransport`, a one-method protocol, so
the adapter's error mapping is unit-tested against a stub and the real client
(``httpx2``) is imported lazily only when a real transport is built. Ordinary
CI therefore never opens a socket.

Privacy
-------
The API key travels in the ``Authorization`` header of the outbound request
and nowhere else: not in exception messages, not in logs, not in any
attribute a caller can read back. Provider error bodies are never surfaced --
they can echo the prompt, which is user content -- only the status class is.
A thinking model's reasoning trace, whether delivered as a separate
``reasoning`` field or inline ``<think>`` tags, is discarded here or by output
validation and never reaches an answer or a log.
"""

import inspect
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from memovo_ai.core.config import OpenRouterSettings
from memovo_ai.generation.models import (
    GenerationRateLimitedError,
    GenerationRequest,
    GenerationResult,
    GenerationTimeoutError,
    GenerationUnavailableError,
    InvalidGenerationOutputError,
)

__all__ = [
    "CHAT_COMPLETIONS_PATH",
    "HttpResponse",
    "HttpTransport",
    "OpenRouterGenerationProvider",
    "TransportError",
    "TransportTimeoutError",
    "build_payload",
]

CHAT_COMPLETIONS_PATH = "/chat/completions"

_RATE_LIMITED = 429
_SERVER_ERROR_FLOOR = 500
_CLIENT_ERROR_FLOOR = 400
#: Client statuses that mean the *deployment* cannot use the provider right
#: now -- credentials, credits, a model that is not served. None of them is
#: the caller's fault, and none can be fixed by a different request.
_UNAVAILABLE_CLIENT_STATUSES = frozenset({401, 402, 403, 404})


class TransportError(Exception):
    """The request never produced an HTTP response (connection, DNS, TLS).

    Raised by transports; the message must be generic.
    """


class TransportTimeoutError(TransportError):
    """The transport gave up waiting."""


@dataclass(frozen=True, slots=True)
class HttpResponse:
    status: int
    headers: Mapping[str, str]
    body: bytes


class HttpTransport(Protocol):
    """The one operation this adapter needs from an HTTP client."""

    async def post_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_seconds: float,
    ) -> HttpResponse:
        """POST ``payload`` as JSON; never retry; raise :class:`TransportError`
        when no response arrives."""
        ...


class Httpx2Transport:
    """The real transport, built on ``httpx2``.

    Imported lazily so nothing in the default test path touches it. One
    client is held for connection reuse and closed with the adapter.
    """

    __slots__ = ("_client",)

    def __init__(self) -> None:
        import httpx2

        # No retries anywhere: the default transport makes one attempt.
        self._client = httpx2.AsyncClient()

    async def post_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_seconds: float,
    ) -> HttpResponse:
        import httpx2

        try:
            response = await self._client.post(
                url, headers=dict(headers), json=dict(payload), timeout=timeout_seconds
            )
        except httpx2.TimeoutException as error:
            message = "generation provider request timed out"
            raise TransportTimeoutError(message) from error
        except httpx2.HTTPError as error:
            # The library's message can include the URL; ours does not.
            message = "generation provider request failed before a response"
            raise TransportError(message) from error

        return HttpResponse(
            status=response.status_code,
            headers={key.lower(): value for key, value in response.headers.items()},
            body=response.content,
        )

    async def aclose(self) -> None:
        await self._client.aclose()


def build_payload(request: GenerationRequest, *, model: str) -> dict[str, Any]:
    """The chat-completions body, as a plain value for tests to assert on.

    Only widely supported parameters are sent. Structured-output parameters
    are deliberately absent: the plan requires verifying what the approved
    model supports before relying on them, and output validation does not
    depend on the provider enforcing a schema.
    """
    return {
        "model": model,
        "messages": [{"role": m.role, "content": m.content} for m in request.messages],
        "max_tokens": request.max_output_tokens,
        "temperature": request.temperature,
        "stream": False,
    }


def _retry_after_seconds(headers: Mapping[str, str], *, cap: int) -> int | None:
    """The provider's ``Retry-After`` in whole seconds, bounded by ``cap``.

    Only the delay-seconds form is honoured; an HTTP-date form is ignored
    rather than parsed, since the Backend falls back to its own backoff when
    the header is absent (contract section 21.2.1).
    """
    raw = headers.get("retry-after")
    if raw is None:
        return None

    try:
        seconds = int(float(raw.strip()))
    except ValueError:
        return None

    return max(0, min(seconds, cap))


class OpenRouterGenerationProvider:
    """Completes requests against one approved model through OpenRouter."""

    __slots__ = ("_api_key", "_endpoint", "_model", "_retry_after_cap", "_timeout", "_transport")

    def __init__(
        self,
        *,
        transport: HttpTransport,
        settings: OpenRouterSettings,
        model: str,
        timeout_seconds: float,
        retry_after_max_seconds: int,
    ) -> None:
        if not model.strip():
            message = "a generation model identifier is required"
            raise ValueError(message)

        self._transport = transport
        self._api_key = settings.api_key.get_secret_value()
        self._endpoint = settings.base_url.rstrip("/") + CHAT_COMPLETIONS_PATH
        self._model = model
        self._timeout = timeout_seconds
        self._retry_after_cap = retry_after_max_seconds

    @classmethod
    def connect(
        cls,
        *,
        settings: OpenRouterSettings,
        model: str,
        timeout_seconds: float,
        retry_after_max_seconds: int,
    ) -> "OpenRouterGenerationProvider":
        """Build the adapter with the real transport.

        Raises:
            GenerationUnavailableError: if no API key is configured.
        """
        if not settings.is_configured:
            message = (
                "OpenRouter API key is not configured; "
                "set MEMOVO_OPENROUTER_API_KEY to enable generation"
            )
            raise GenerationUnavailableError(message)

        return cls(
            transport=Httpx2Transport(),
            settings=settings,
            model=model,
            timeout_seconds=timeout_seconds,
            retry_after_max_seconds=retry_after_max_seconds,
        )

    @property
    def model(self) -> str:
        return self._model

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """One attempt. Every failure is a :class:`GenerationError` subclass."""
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        try:
            response = await self._transport.post_json(
                self._endpoint,
                headers=headers,
                payload=build_payload(request, model=self._model),
                timeout_seconds=self._timeout,
            )
        except TransportTimeoutError as error:
            message = "generation provider timed out"
            raise GenerationTimeoutError(message) from error
        except TransportError as error:
            message = "generation provider could not be reached"
            raise GenerationUnavailableError(message) from error

        self._raise_for_status(response)

        return self._parse(response.body)

    def _raise_for_status(self, response: HttpResponse) -> None:
        status = response.status

        if status == _RATE_LIMITED:
            message = "generation provider is rate limited"
            raise GenerationRateLimitedError(
                message,
                retry_after_seconds=_retry_after_seconds(
                    response.headers, cap=self._retry_after_cap
                ),
            )

        if status >= _SERVER_ERROR_FLOOR or status in _UNAVAILABLE_CLIENT_STATUSES:
            # The body is not read: a provider error can quote the request.
            message = f"generation provider answered with status class {status // 100}xx"
            raise GenerationUnavailableError(message)

        if status >= _CLIENT_ERROR_FLOOR:
            # 400/422 and the like: the provider rejected *our* request. That
            # is a defect in how the request was built, not a transient
            # condition and not the caller's input -- so it is not
            # AI_INVALID_RESPONSE either. Surfaced as unavailable so the
            # Backend's bounded retries stay harmless and the status is logged.
            message = f"generation provider rejected the request with status {status}"
            raise GenerationUnavailableError(message)

    def _parse(self, body: bytes) -> GenerationResult:
        """Read a successful response, refusing anything structurally off.

        A 200 with an unusable body is malformed output (section 21.9):
        the provider answered, and what it said cannot be used.
        """
        try:
            document = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            message = "generation provider returned a non-JSON success body"
            raise InvalidGenerationOutputError(message) from error

        if not isinstance(document, dict):
            message = "generation provider returned a non-object success body"
            raise InvalidGenerationOutputError(message)

        # OpenRouter reports some upstream failures as a 200 carrying an
        # ``error`` object. That is an unavailable provider, not bad output.
        if isinstance(document.get("error"), dict):
            upstream = document["error"].get("code")
            status = upstream if isinstance(upstream, int) else 0
            if status == _RATE_LIMITED:
                message = "generation provider is rate limited"
                raise GenerationRateLimitedError(message)
            message = "generation provider reported an upstream error"
            raise GenerationUnavailableError(message)

        choices = document.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            message = "generation provider response carries no choices"
            raise InvalidGenerationOutputError(message)

        first = choices[0]
        message_object = first.get("message")
        content = message_object.get("content") if isinstance(message_object, dict) else None
        if not isinstance(content, str) or not content.strip():
            message = "generation provider response carries no content"
            raise InvalidGenerationOutputError(message)

        finish_reason = first.get("finish_reason")
        usage_raw = document.get("usage")
        usage: dict[str, Any] = usage_raw if isinstance(usage_raw, dict) else {}
        model = document.get("model")

        return GenerationResult(
            # ``message.reasoning`` (thinking models) is deliberately dropped.
            text=content,
            finish_reason=str(finish_reason) if finish_reason is not None else "unknown",
            model=str(model) if isinstance(model, str) else self._model,
            prompt_tokens=_count(usage.get("prompt_tokens")),
            completion_tokens=_count(usage.get("completion_tokens")),
        )

    async def aclose(self) -> None:
        close = getattr(self._transport, "aclose", None)
        if close is None:
            return

        result = close()
        if inspect.isawaitable(result):
            await result


def _count(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None
