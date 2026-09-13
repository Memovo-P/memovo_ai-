"""Public error definitions.

Every public failure serializes to the standardized envelope::

    {"error": {"code": ..., "message": ..., "retryable": ...}}

Messages are **fixed per code**, not derived from the underlying exception.
That is deliberate: stack traces, model paths, hostnames, secrets, connection
strings and raw SDK errors must never leak, and the only way to guarantee that
for an exception this service did not raise itself is never to put its text in
the response. Detail belongs in logs, keyed by correlation id.

Classification follows contract v1.9 section 21 and its approved addendum
(section 21.9): the Backend keys retries on **HTTP status + code**, and
``retryable`` must agree with that classification rather than contradict it.

- Every 5xx is retryable **except** ``502 AI_INVALID_RESPONSE``, which is
  always non-retryable: malformed model output is a contract failure, not a
  transient one.
- ``429 RATE_LIMITED`` is retryable, with a bounded ``Retry-After`` when the
  upstream provider supplied one.
- 4xx is non-retryable.

This module imports nothing from the domain layers, so it stays usable from
anywhere. The translation from a domain exception to a code lives at the
transport boundary, in :mod:`memovo_ai.api.errors`.
"""

from types import MappingProxyType

from memovo_ai.schemas.errors import ErrorCode

__all__ = [
    "ERROR_HTTP_STATUS",
    "ERROR_MESSAGE",
    "ERROR_RETRYABLE",
    "AiServiceError",
]

#: HTTP status per code. The JSON structure is what is locked; the statuses
#: follow contract section 21: 502 is reserved for ``AI_INVALID_RESPONSE``.
ERROR_HTTP_STATUS: MappingProxyType[ErrorCode, int] = MappingProxyType(
    {
        ErrorCode.INVALID_INPUT: 422,
        ErrorCode.INVALID_REQUEST: 400,
        ErrorCode.EMBEDDING_FAILED: 503,
        ErrorCode.CHUNKING_FAILED: 500,
        ErrorCode.VECTOR_SEARCH_FAILED: 503,
        ErrorCode.MODEL_UNAVAILABLE: 503,
        ErrorCode.TIMEOUT: 504,
        ErrorCode.INTERNAL_ERROR: 500,
        ErrorCode.GENERATION_UNAVAILABLE: 503,
        ErrorCode.RATE_LIMITED: 429,
        ErrorCode.AI_INVALID_RESPONSE: 502,
    }
)

#: Whether the Backend will retry. Consistent with the status, never an
#: override of it (contract section 21.9).
#:
#: ``INTERNAL_ERROR`` and ``CHUNKING_FAILED`` are 500s and therefore
#: retryable under the canonical 5xx rule, bounded by the Backend's three
#: total attempts. The earlier "deterministic, so never retry" metadata
#: contradicted that classification and was reviewed away with the addendum.
#: A defect still fails every attempt -- the retry is cheap and finite, and
#: a leaking user pre-filter raises rather than returning rows, so a retry
#: cannot expose anything either.
ERROR_RETRYABLE: MappingProxyType[ErrorCode, bool] = MappingProxyType(
    {
        ErrorCode.INVALID_INPUT: False,
        ErrorCode.INVALID_REQUEST: False,
        ErrorCode.EMBEDDING_FAILED: True,
        ErrorCode.CHUNKING_FAILED: True,
        ErrorCode.VECTOR_SEARCH_FAILED: True,
        ErrorCode.MODEL_UNAVAILABLE: True,
        ErrorCode.TIMEOUT: True,
        ErrorCode.INTERNAL_ERROR: True,
        ErrorCode.GENERATION_UNAVAILABLE: True,
        ErrorCode.RATE_LIMITED: True,
        ErrorCode.AI_INVALID_RESPONSE: False,
    }
)

#: The public message per code. Safe by construction: no identifiers, no user
#: content, no infrastructure detail, no provider detail.
ERROR_MESSAGE: MappingProxyType[ErrorCode, str] = MappingProxyType(
    {
        ErrorCode.INVALID_INPUT: "Request input failed validation",
        ErrorCode.INVALID_REQUEST: "The request could not be processed",
        ErrorCode.EMBEDDING_FAILED: "Embedding generation failed",
        ErrorCode.CHUNKING_FAILED: "Content could not be chunked",
        ErrorCode.VECTOR_SEARCH_FAILED: "Vector search is temporarily unavailable",
        ErrorCode.MODEL_UNAVAILABLE: "Embedding model is temporarily unavailable",
        ErrorCode.TIMEOUT: "The request timed out",
        ErrorCode.INTERNAL_ERROR: "An internal error occurred",
        ErrorCode.GENERATION_UNAVAILABLE: "Answer generation is temporarily unavailable",
        ErrorCode.RATE_LIMITED: "The generation provider is rate limited",
        ErrorCode.AI_INVALID_RESPONSE: "The generated output was invalid",
    }
)


class AiServiceError(Exception):
    """An error already classified against the public contract.

    Raise this where the correct public code is known at the raise site.
    Anything else is classified at the transport boundary.

    ``retry_after_seconds`` is only meaningful for ``RATE_LIMITED``; the
    transport layer emits it as a ``Retry-After`` header, already bounded by
    the caller.
    """

    __slots__ = ("code", "public_message", "retry_after_seconds", "retryable")

    def __init__(
        self,
        code: ErrorCode,
        *,
        message: str | None = None,
        retryable: bool | None = None,
        retry_after_seconds: int | None = None,
    ) -> None:
        self.code = code
        self.public_message = message if message is not None else ERROR_MESSAGE[code]
        self.retryable = retryable if retryable is not None else ERROR_RETRYABLE[code]
        self.retry_after_seconds = retry_after_seconds
        super().__init__(self.public_message)

    @property
    def http_status(self) -> int:
        return ERROR_HTTP_STATUS[self.code]
