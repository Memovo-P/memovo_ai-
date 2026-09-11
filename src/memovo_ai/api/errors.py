"""Exception mapping at the transport boundary.

Doc 04 puts exception mapping in ``api/``, and this is the only layer allowed
to know about every domain package at once. Domain code raises its own
exceptions; nothing below this module decides an HTTP status.

Every response produced here is built through
:class:`~memovo_ai.schemas.errors.ErrorResponse`, so the envelope shape is
guaranteed by the same model the contract tests check.
"""

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from memovo_ai.core.errors import (
    ERROR_HTTP_STATUS,
    ERROR_MESSAGE,
    ERROR_RETRYABLE,
    AiServiceError,
)
from memovo_ai.core.logging import log_event, safe_text
from memovo_ai.embeddings.models import (
    EmbeddingError,
    EmbeddingInferenceError,
    InvalidEmbeddingError,
    ModelUnavailableError,
)
from memovo_ai.providers.vector_search.base import (
    InvalidUserScopeError,
    UserIsolationError,
    VectorSearchUnavailableError,
)
from memovo_ai.schemas.errors import ErrorCode, ErrorDetail, ErrorResponse

__all__ = ["classify", "error_response", "register_exception_handlers"]

_logger = logging.getLogger(__name__)

_CLIENT_ERROR_FLOOR = 400
_SERVER_ERROR_FLOOR = 500


def classify(exception: BaseException) -> ErrorCode:
    """Map a raised exception onto a public error code.

    Order matters: the most specific type wins. Anything unrecognised becomes
    ``INTERNAL_ERROR`` rather than being guessed at.
    """
    match exception:
        case AiServiceError():
            return exception.code
        case InvalidUserScopeError():
            return ErrorCode.INVALID_INPUT
        case ModelUnavailableError():
            return ErrorCode.MODEL_UNAVAILABLE
        case InvalidEmbeddingError() | EmbeddingInferenceError() | EmbeddingError():
            return ErrorCode.EMBEDDING_FAILED
        case VectorSearchUnavailableError():
            return ErrorCode.VECTOR_SEARCH_FAILED
        case UserIsolationError():
            # A broken user pre-filter is a service defect, not a bad request,
            # and must never be presented as retryable.
            return ErrorCode.INTERNAL_ERROR
        case TimeoutError():
            return ErrorCode.TIMEOUT
        case _:
            return ErrorCode.INTERNAL_ERROR


def error_response(
    code: ErrorCode,
    *,
    message: str | None = None,
    retryable: bool | None = None,
    status_code: int | None = None,
) -> JSONResponse:
    """Render the standardized envelope."""
    payload = ErrorResponse(
        error=ErrorDetail(
            code=code,
            message=message if message is not None else ERROR_MESSAGE[code],
            retryable=retryable if retryable is not None else ERROR_RETRYABLE[code],
        )
    )

    return JSONResponse(
        status_code=status_code if status_code is not None else ERROR_HTTP_STATUS[code],
        content=payload.model_dump(mode="json"),
    )


def _validation_message(exception: RequestValidationError) -> str:
    """Name the offending fields without echoing their values.

    Pydantic's own error detail carries an ``input`` key holding the rejected
    value, which for this service can be Memory content or a user's query.
    Only the field location and the failure type are safe to return.
    """
    fields = sorted(
        {
            ".".join(str(part) for part in error.get("loc", ()) if part != "body") or "body"
            for error in exception.errors()
        }
    )

    return f"Request input failed validation for: {', '.join(fields)}"


def register_exception_handlers(app: FastAPI) -> None:
    """Install handlers so every failure leaves as the standard envelope."""

    @app.exception_handler(RequestValidationError)
    async def _handle_validation(
        request: Request, exception: RequestValidationError
    ) -> JSONResponse:
        return error_response(ErrorCode.INVALID_INPUT, message=_validation_message(exception))

    @app.exception_handler(AiServiceError)
    async def _handle_service_error(request: Request, exception: AiServiceError) -> JSONResponse:
        return error_response(
            exception.code,
            message=exception.public_message,
            retryable=exception.retryable,
            status_code=exception.http_status,
        )

    @app.exception_handler(StarletteHTTPException)
    async def _handle_http_exception(
        request: Request, exception: StarletteHTTPException
    ) -> JSONResponse:
        # Transport-level failures such as 404 and 405. The status is
        # preserved -- rewriting a 405 to 400 would be less accurate than the
        # recommended table is prescriptive.
        code = (
            ErrorCode.INVALID_REQUEST
            if _CLIENT_ERROR_FLOOR <= exception.status_code < _SERVER_ERROR_FLOOR
            else ErrorCode.INTERNAL_ERROR
        )
        return error_response(code, status_code=exception.status_code)

    @app.exception_handler(Exception)
    async def _handle_unexpected(request: Request, exception: Exception) -> JSONResponse:
        code = classify(exception)
        unclassified = code is ErrorCode.INTERNAL_ERROR

        # Only unclassified failures are defects worth a traceback. Expected
        # conditions -- an unavailable model, a failed embedding -- record
        # the exception type and nothing more, so operational noise stays
        # low. The exception *message* is never logged either: a driver
        # error can carry a connection string (doc 06, section 6).
        log_event(
            _logger,
            "request.failed",
            level=logging.ERROR if unclassified else logging.WARNING,
            exc_info=unclassified,
            route=safe_text(request.url.path, fallback="unmatched"),
            error_type=type(exception).__name__,
            error_code=code.value,
            status=ERROR_HTTP_STATUS[code],
        )

        return error_response(code)
