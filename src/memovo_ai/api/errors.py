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
from memovo_ai.core.logging import (
    CORRELATION_ID_HEADER,
    CORRELATION_SCOPE_KEY,
    bind_correlation_id,
    log_event,
    reset_correlation_id,
    safe_text,
)
from memovo_ai.embeddings.models import (
    EmbeddingError,
    EmbeddingInferenceError,
    InvalidEmbeddingError,
    ModelUnavailableError,
)
from memovo_ai.generation.models import (
    GenerationRateLimitedError,
    GenerationTimeoutError,
    GenerationUnavailableError,
    InvalidGenerationOutputError,
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
        case InvalidGenerationOutputError():
            # Malformed model output: 502, never retryable (section 21.9).
            return ErrorCode.AI_INVALID_RESPONSE
        case GenerationRateLimitedError():
            return ErrorCode.RATE_LIMITED
        case GenerationTimeoutError():
            return ErrorCode.TIMEOUT
        case GenerationUnavailableError():
            return ErrorCode.GENERATION_UNAVAILABLE
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
    retry_after_seconds: int | None = None,
) -> JSONResponse:
    """Render the standardized envelope.

    ``retry_after_seconds`` becomes a ``Retry-After`` header. It is only
    ever set for ``RATE_LIMITED`` and is already bounded by the adapter, so
    the Backend can honour it within its own limits (section 21.2.1).
    """
    payload = ErrorResponse(
        error=ErrorDetail(
            code=code,
            message=message if message is not None else ERROR_MESSAGE[code],
            retryable=retryable if retryable is not None else ERROR_RETRYABLE[code],
        )
    )
    headers = {"Retry-After": str(retry_after_seconds)} if retry_after_seconds is not None else None

    return JSONResponse(
        status_code=status_code if status_code is not None else ERROR_HTTP_STATUS[code],
        content=payload.model_dump(mode="json"),
        headers=headers,
    )


def _correlation_id_of(request: Request) -> str:
    """The id this request was given, read from the ASGI scope.

    Not from the ContextVar: the ``Exception`` handler runs inside Starlette's
    ``ServerErrorMiddleware``, which wraps this service's middleware, so by
    then the middleware has re-raised and reset the ContextVar. The scope is
    the same dict at every layer, so the id survives there.
    """
    value = request.scope.get(CORRELATION_SCOPE_KEY)

    return value if isinstance(value, str) else ""


def _correlated(request: Request, response: JSONResponse) -> JSONResponse:
    """Echo the correlation id, so an error is traceable like a success.

    Applied in the handlers rather than only in the middleware because the
    responses built for an unhandled exception never pass back through it.
    """
    correlation_id = _correlation_id_of(request)
    if correlation_id:
        response.headers[CORRELATION_ID_HEADER] = correlation_id

    return response


def _record_failure(
    request: Request,
    exception: BaseException,
    code: ErrorCode,
    *,
    status_code: int,
    unclassified: bool,
) -> None:
    """Emit the one ``request.failed`` record for a failed request.

    Operational fields only: the route, the exception *type*, the public code
    and the status. Never the exception message -- a driver error can carry
    a connection string, and a service error's message can name a count or
    a field that is better left in the response (doc 06, section 6).

    Only an unclassified failure is a defect worth a traceback. A classified
    one -- an unavailable model, a failed embedding, a chunking error already
    mapped to its public code -- records the type and nothing more, so
    operational noise stays low.

    The correlation id is rebound from the ASGI scope for the duration of
    the call. The ``Exception`` handler runs outside the middleware, where
    the ContextVar has already been reset; without this the one record that
    explains the failure would read ``correlation_id: unset`` and could not
    be joined to the access record for the same request. For a handler that
    runs inside the middleware the rebinding is a no-op.
    """
    token = bind_correlation_id(_correlation_id_of(request) or None)
    try:
        log_event(
            _logger,
            "request.failed",
            level=logging.ERROR if unclassified else logging.WARNING,
            exc_info=unclassified,
            route=safe_text(request.url.path, fallback="unmatched"),
            error_type=type(exception).__name__,
            error_code=code.value,
            status=status_code,
        )
    finally:
        reset_correlation_id(token)


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
        return _correlated(
            request,
            error_response(ErrorCode.INVALID_INPUT, message=_validation_message(exception)),
        )

    @app.exception_handler(AiServiceError)
    async def _handle_service_error(request: Request, exception: AiServiceError) -> JSONResponse:
        # Classified at the raise site, so there is no unknown to diagnose.
        # It is still recorded: without this, a CHUNKING_FAILED or
        # EMBEDDING_FAILED left only an access record -- a 500 or 503 that
        # was visible without being explained.
        _record_failure(
            request,
            exception,
            exception.code,
            status_code=exception.http_status,
            unclassified=False,
        )

        return _correlated(
            request,
            error_response(
                exception.code,
                message=exception.public_message,
                retryable=exception.retryable,
                status_code=exception.http_status,
                retry_after_seconds=exception.retry_after_seconds,
            ),
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
        return _correlated(request, error_response(code, status_code=exception.status_code))

    @app.exception_handler(Exception)
    async def _handle_unexpected(request: Request, exception: Exception) -> JSONResponse:
        code = classify(exception)

        _record_failure(
            request,
            exception,
            code,
            status_code=ERROR_HTTP_STATUS[code],
            unclassified=code is ErrorCode.INTERNAL_ERROR,
        )

        retry_after = (
            exception.retry_after_seconds
            if isinstance(exception, GenerationRateLimitedError)
            else None
        )

        return _correlated(request, error_response(code, retry_after_seconds=retry_after))
