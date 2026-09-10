"""Composition root and request dependencies.

This is where concrete providers are chosen and wired together. Routes and
services never construct their own infrastructure; they receive it.

Wrapping happens here, once, so the guarantees are not optional:

* the embedding provider is wrapped in
  :class:`~memovo_ai.embeddings.base.ValidatedEmbeddingProvider`, so a
  wrong-dimension or non-finite vector cannot reach the vector engine
* the search service wraps the vector provider in
  :class:`~memovo_ai.providers.vector_search.base.UserScopedVectorSearchProvider`
  itself, so isolation holds regardless of what is passed in
"""

from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Request

from memovo_ai.core.config import (
    FAKE_VECTOR_PROVIDER,
    EmbeddingSettings,
    ProviderSettings,
    SearchSettings,
)
from memovo_ai.embeddings.base import (
    EmbeddingProvider,
    NormalizingEmbeddingProvider,
    ValidatedEmbeddingProvider,
)
from memovo_ai.embeddings.models import ModelUnavailableError
from memovo_ai.embeddings.qwen3 import Qwen3EmbeddingProvider
from memovo_ai.providers.vector_search.base import VectorSearchProvider
from memovo_ai.providers.vector_search.fake import FakeVectorSearchProvider
from memovo_ai.services.memory_processor import MemoryProcessorService
from memovo_ai.services.memory_search import MemorySearchService

__all__ = [
    "ProcessorService",
    "SearchService",
    "Services",
    "build_embedding_provider",
    "build_services",
    "build_vector_search_provider",
    "get_memory_processor_service",
    "get_memory_search_service",
]


@dataclass(frozen=True, slots=True)
class Services:
    """The application services, sharing one loaded embedding model."""

    memory_search: MemorySearchService
    memory_processor: MemoryProcessorService


def build_embedding_provider(settings: EmbeddingSettings | None = None) -> EmbeddingProvider:
    """Load the embedding model once, normalize its output, then validate it.

    Order matters. Qwen3 runs in bfloat16, so its own normalization is only
    accurate to about 1e-3; rescaling in float64 first is what lets the
    validator enforce ``expect_unit_norm`` at its existing tolerance instead
    of the tolerance being widened to accommodate model precision.

    True unit vectors also make cosine and inner product agree, so the vector
    index cannot be configured to score differently from this service.

    Raises:
        ModelUnavailableError: if the runtime is missing or the model fails
            to load.
    """
    resolved = settings if settings is not None else EmbeddingSettings()

    return ValidatedEmbeddingProvider(
        NormalizingEmbeddingProvider(Qwen3EmbeddingProvider.load(resolved)),
        dimension=resolved.dimension,
        expect_unit_norm=True,
    )


def build_vector_search_provider(
    settings: ProviderSettings | None = None,
) -> VectorSearchProvider:
    """Select the read-only vector search adapter.

    Only the in-memory fake is registered. The production database has not
    been chosen (doc 01, section 8), so an unknown name fails loudly rather
    than silently falling back to an empty index and reporting "no memories"
    for every query.
    """
    resolved = settings if settings is not None else ProviderSettings()

    if resolved.vector_provider == FAKE_VECTOR_PROVIDER:
        return FakeVectorSearchProvider()

    message = (
        f"unknown vector provider {resolved.vector_provider!r}; "
        f"only {FAKE_VECTOR_PROVIDER!r} is available until the production "
        "vector database is selected"
    )
    raise ValueError(message)


def build_services(
    *,
    embedding_settings: EmbeddingSettings | None = None,
    provider_settings: ProviderSettings | None = None,
    search_settings: SearchSettings | None = None,
) -> Services:
    """Assemble every service from configuration.

    The embedding provider is built once and shared, so the model is loaded a
    single time no matter how many services use it (doc 02, section 7).
    """
    embeddings = build_embedding_provider(embedding_settings)

    return Services(
        memory_search=MemorySearchService(
            embedding_provider=embeddings,
            vector_search=build_vector_search_provider(provider_settings),
            settings=search_settings if search_settings is not None else SearchSettings(),
        ),
        memory_processor=MemoryProcessorService(embedding_provider=embeddings),
    )


def get_memory_search_service(request: Request) -> MemorySearchService:
    """Return the service built at startup.

    The model is loaded once during startup and shared (doc 02, section 7);
    nothing is constructed per request.

    Raises:
        ModelUnavailableError: if startup could not build the service. Phase
            15 maps this to ``MODEL_UNAVAILABLE``.
    """
    service: MemorySearchService | None = getattr(request.app.state, "memory_search_service", None)

    if service is None:
        message = "embedding model is not available"
        raise ModelUnavailableError(message)

    return service


def get_memory_processor_service(request: Request) -> MemoryProcessorService:
    """Return the processor built at startup.

    Raises:
        ModelUnavailableError: if startup could not build it. Phase 15 maps
            this to ``MODEL_UNAVAILABLE``.
    """
    service: MemoryProcessorService | None = getattr(
        request.app.state, "memory_processor_service", None
    )

    if service is None:
        message = "embedding model is not available"
        raise ModelUnavailableError(message)

    return service


SearchService = Annotated[MemorySearchService, Depends(get_memory_search_service)]
ProcessorService = Annotated[MemoryProcessorService, Depends(get_memory_processor_service)]
