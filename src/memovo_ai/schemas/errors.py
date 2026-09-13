"""The standardized public error envelope.

Every public failure serializes to exactly::

    {"error": {"code": ..., "message": ..., "retryable": ...}}

Nothing else is exposed. Exception types, stack traces, model paths, provider
names, connection strings and raw SDK errors must never reach these fields.
Mapping internal exceptions onto these codes, and onto HTTP status codes, is
the transport layer's job (:mod:`memovo_ai.api.errors`).

The Backend classifies a failure by **HTTP status + code**; ``retryable`` is
consistent metadata, never an override (contract section 21 and its approved
addendum, section 21.9).
"""

from enum import StrEnum
from typing import Annotated

from pydantic import Field

from memovo_ai.schemas.common import MemovoBaseModel

__all__ = ["ErrorCode", "ErrorDetail", "ErrorResponse"]


class ErrorCode(StrEnum):
    """The supported public error codes. Any other value is rejected."""

    INVALID_INPUT = "INVALID_INPUT"
    INVALID_REQUEST = "INVALID_REQUEST"
    EMBEDDING_FAILED = "EMBEDDING_FAILED"
    CHUNKING_FAILED = "CHUNKING_FAILED"
    VECTOR_SEARCH_FAILED = "VECTOR_SEARCH_FAILED"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    TIMEOUT = "TIMEOUT"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    # Generation (contract section 21 and the approved addendum, 21.9).
    GENERATION_UNAVAILABLE = "GENERATION_UNAVAILABLE"
    RATE_LIMITED = "RATE_LIMITED"
    AI_INVALID_RESPONSE = "AI_INVALID_RESPONSE"


class ErrorDetail(MemovoBaseModel):
    """The error body. Exactly three fields, all required.

    ``code`` opts out of strict mode so the plain contract string is accepted
    as well as the enum member; values outside :class:`ErrorCode` are still
    rejected. ``retryable`` has no default -- every raise site must state it
    explicitly rather than inherit a guess.
    """

    code: Annotated[ErrorCode, Field(strict=False)]
    message: str
    retryable: bool


class ErrorResponse(MemovoBaseModel):
    """The top-level error envelope."""

    error: ErrorDetail
