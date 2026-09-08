"""FastAPI application.

The embedding model is loaded once during startup and shared across requests
(doc 02, section 7). A failed load does not crash the process: it leaves the
service unset, and requests then fail with a clear unavailable state rather
than the application refusing to start (doc 06, section 9). That also lets the
API be exercised without model weights present.

This service is internal. It must not be exposed publicly (doc 06, section 7);
the Backend is the only caller.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from memovo_ai.api.dependencies import Services, build_services
from memovo_ai.api.errors import register_exception_handlers
from memovo_ai.api.router import api_router
from memovo_ai.embeddings.models import EmbeddingError

__all__ = ["app", "create_app"]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build the services once, at startup, sharing one loaded model.

    Importing the composition root here is safe: ``qwen3`` defers the model
    runtime import to ``load()``, so nothing heavy is pulled in until a model
    is actually requested.
    """
    services: Services | None
    try:
        services = build_services()
    except (EmbeddingError, ValueError):
        # Startup continues in an unavailable state. The dependencies raise
        # ModelUnavailableError per request, which Phase 15 maps to a
        # standardized 503 rather than an opaque crash loop.
        services = None

    app.state.memory_search_service = services.memory_search if services else None
    app.state.memory_processor_service = services.memory_processor if services else None
    yield
    app.state.memory_search_service = None
    app.state.memory_processor_service = None


def create_app() -> FastAPI:
    """Build the application.

    Tests override ``get_memory_search_service`` rather than starting the real
    model, so nothing here downloads weights.
    """
    application = FastAPI(
        title="Memovo AI Service",
        version="0.1.0",
        summary="Memory processing and retrieval for Memovo.",
        lifespan=lifespan,
    )
    register_exception_handlers(application)
    application.include_router(api_router)

    return application


app = create_app()
