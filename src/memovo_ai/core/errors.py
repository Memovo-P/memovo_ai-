"""Public error definitions.

Every public failure serializes to the standardized envelope (doc 03,
Phase 15)::

    {"error": {"code": ..., "message": ..., "retryable": ...}}

Messages are **fixed per code**, not derived from the underlying exception.
That is deliberate: doc 06 section 6 forbids leaking stack traces, model
paths, hostnames, secrets, connection strings and raw SDK errors, and the only
way to guarantee that for an exception this service did not raise itself is
never to put its text in the response. Detail belongs in logs (Phase 18),
keyed by correlation id.

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

#: HTTP status per code, following doc 03 Phase 15. That table is an
#: implementation recommendation rather than a locked contract; the JSON
#: structure is what is locked.
#:
#: ``CHUNKING_FAILED`` has no row in the source table. It is mapped to 500:
#: chunking is deterministic local work, so a failure is a service defect
#: rather than a transient dependency problem.
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
    }
)

#: Whether the Backend should retry. A code is retryable only when repeating
#: the identical request could plausibly succeed.
#:
#: ``CHUNKING_FAILED`` is not retryable: chunking is deterministic, so the
#: same input fails the same way. ``INTERNAL_ERROR`` is not retryable either
#: -- it covers defects, including a misconfigured user pre-filter, and
#: retrying that would return the same wrong rows again.
ERROR_RETRYABLE: MappingProxyType[ErrorCode, bool] = MappingProxyType(
    {
        ErrorCode.INVALID_INPUT: False,
        ErrorCode.INVALID_REQUEST: False,
        ErrorCode.EMBEDDING_FAILED: True,
        ErrorCode.CHUNKING_FAILED: False,
        ErrorCode.VECTOR_SEARCH_FAILED: True,
        ErrorCode.MODEL_UNAVAILABLE: True,
        ErrorCode.TIMEOUT: True,
        ErrorCode.INTERNAL_ERROR: False,
    }
)

#: The public message per code. Safe by construction: no identifiers, no user
#: content, no infrastructure detail.
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
    }
)


class AiServiceError(Exception):
    """An error already classified against the public contract.

    Raise this where the correct public code is known at the raise site.
    Anything else is classified at the transport boundary.
    """

    __slots__ = ("code", "public_message", "retryable")

    def __init__(
        self,
        code: ErrorCode,
        *,
        message: str | None = None,
        retryable: bool | None = None,
    ) -> None:
        self.code = code
        self.public_message = message if message is not None else ERROR_MESSAGE[code]
        self.retryable = retryable if retryable is not None else ERROR_RETRYABLE[code]
        super().__init__(self.public_message)

    @property
    def http_status(self) -> int:
        return ERROR_HTTP_STATUS[self.code]
