"""The standardized public error envelope.

Every public failure serializes to exactly::

    {"error": {"code": ..., "message": ..., "retryable": ...}}

Nothing else is exposed. Exception types, stack traces, model paths, provider
names, connection strings and raw SDK errors must never reach these fields
(doc 06, section 6). Mapping internal exceptions onto these codes, and onto
HTTP status codes, is Phase 15.
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
