"""Per-request correlation and access logging.

One structured record per request, carrying only what doc 06 section 5 allows:
correlation id, method, route, status and duration. Never the body, never the
query string, never a header value other than the correlation id itself.

The correlation id comes from the Backend's ``X-Request-Id`` header when
present (contract, section 5) and is generated otherwise, so every record is
attributable even if the caller sends nothing. It is echoed back on the
response, which is what lets the Backend tie its job log to this service's
log for a request it did not label itself.
"""

import logging
from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route

from memovo_ai.core.logging import (
    CORRELATION_ID_HEADER,
    Timer,
    bind_correlation_id,
    current_correlation_id,
    log_event,
    reset_correlation_id,
    safe_text,
    timed,
)

__all__ = ["RequestContextMiddleware"]

_logger = logging.getLogger(__name__)

_SERVER_ERROR_STATUS = 500


def _route_of(request: Request) -> str:
    """The matched route template, or a sanitized path.

    The template is preferred because it is a fixed string chosen by this
    service. A path only reaches the fallback when nothing matched -- a 404 --
    and that path is caller-controlled text, so it is sanitized before it can
    reach a log line.
    """
    route = request.scope.get("route")
    if isinstance(route, Route):
        return route.path

    return safe_text(request.url.path, fallback="unmatched")


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Binds a correlation id and logs the outcome of every request."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        token = bind_correlation_id(request.headers.get(CORRELATION_ID_HEADER))
        correlation_id = current_correlation_id()
        elapsed = Timer()

        try:
            try:
                with timed() as elapsed:
                    response = await call_next(request)
            except Exception:
                # The application's handlers normally convert a failure into
                # the standard envelope, so reaching here means the failure
                # escaped them. ``timed`` has already recorded the duration by
                # the time this runs. Record it, then let it propagate:
                # swallowing it would turn a crash into a silent hang.
                log_event(
                    _logger,
                    "http.request",
                    level=logging.ERROR,
                    method=request.method,
                    route=_route_of(request),
                    status=_SERVER_ERROR_STATUS,
                    duration_ms=elapsed.ms,
                )
                raise

            log_event(
                _logger,
                "http.request",
                level=(
                    logging.WARNING
                    if response.status_code >= _SERVER_ERROR_STATUS
                    else logging.INFO
                ),
                method=request.method,
                route=_route_of(request),
                status=response.status_code,
                duration_ms=elapsed.ms,
            )

            response.headers[CORRELATION_ID_HEADER] = correlation_id

            return response
        finally:
            reset_correlation_id(token)
