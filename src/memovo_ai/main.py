"""FastAPI application.

The embedding model is loaded once during startup and shared across requests
(doc 02, section 7). A failed load does not crash the process: it leaves the
service unset, and requests then fail with a clear unavailable state rather
than the application refusing to start (doc 06, section 9). That also lets the
API be exercised without model weights present.

This service is internal. It must not be exposed publicly (doc 06, section 7);
the Backend is the only caller.
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from memovo_ai.api.dependencies import Services, build_services
from memovo_ai.api.errors import register_exception_handlers
from memovo_ai.api.middleware import RequestContextMiddleware
from memovo_ai.api.router import api_router
from memovo_ai.core.logging import configure_logging, log_event
from memovo_ai.core.readiness import ReadinessState, set_readiness
from memovo_ai.embeddings.models import EmbeddingError
from memovo_ai.providers.vector_search.base import VectorSearchUnavailableError

__all__ = ["app", "create_app"]

_logger = logging.getLogger(__name__)


#: Set on the app when it must not build services at startup. Tests supply
#: their own through ``dependency_overrides``; loading a real model just to
#: replace it is slow and makes the suite depend on whether the optional
#: ``embeddings`` extra happens to be installed.
_SKIP_STARTUP_SERVICES = "memovo_skip_startup_services"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build the services once, at startup, sharing one loaded model.

    Importing the composition root here is safe: ``qwen3`` defers the model
    runtime import to ``load()``, so nothing heavy is pulled in until a model
    is actually requested.
    """
    skip = getattr(app.state, _SKIP_STARTUP_SERVICES, False)
    set_readiness(app.state, ReadinessState.LOADING)

    services: Services | None
    try:
        services = None if skip else build_services()
    except (EmbeddingError, VectorSearchUnavailableError, ValueError) as error:
        # Startup continues in an unavailable state. The dependencies raise
        # ModelUnavailableError per request, which Phase 15 maps to a
        # standardized 503 rather than an opaque crash loop.
        #
        # VectorSearchUnavailableError belongs here too: an unset or wrong
        # Atlas connection string is a configuration problem, and a crash
        # loop tells an operator far less than a service that starts and
        # reports itself unavailable (doc 06, section 9).
        services = None

        # The exception type, never its message: a driver error can carry a
        # connection string (doc 06, section 6). The full traceback goes to
        # the operator's log because this one is worth diagnosing.
        log_event(
            _logger,
            "service.startup_failed",
            level=logging.ERROR,
            error_type=type(error).__name__,
            exc_info=True,
        )

    app.state.memory_search_service = services.memory_search if services else None
    app.state.memory_processor_service = services.memory_processor if services else None

    # Readiness follows the services, not the process. Without usable
    # services the process stays alive and answers probes, but must be kept
    # out of the load balancer (doc 06, section 9).
    set_readiness(
        app.state,
        ReadinessState.READY if services is not None else ReadinessState.UNAVAILABLE,
    )

    log_event(_logger, "service.started", services_available=services is not None)

    yield

    # Unready before teardown, so an instance draining connections stops
    # attracting new traffic.
    set_readiness(app.state, ReadinessState.UNAVAILABLE)
    app.state.memory_search_service = None
    app.state.memory_processor_service = None
    if services is not None:
        await services.aclose()

    log_event(_logger, "service.stopped")


def create_app(*, load_services: bool = True) -> FastAPI:
    """Build the application.

    Args:
        load_services: Build the services at startup. Tests pass ``False`` and
            supply their own through ``dependency_overrides``, which keeps the
            suite fast and, more importantly, makes it behave identically
            whether or not the optional ``embeddings`` extra is installed.
            Production always uses the default.
    """
    configure_logging()

    application = FastAPI(
        title="Memovo AI Service",
        version="0.1.0",
        summary="Memory processing and retrieval for Memovo.",
        lifespan=lifespan,
    )
    setattr(application.state, _SKIP_STARTUP_SERVICES, not load_services)
    set_readiness(application.state, ReadinessState.STARTING)
    application.add_middleware(RequestContextMiddleware)
    register_exception_handlers(application)
    application.include_router(api_router)

    return application


app = create_app()
