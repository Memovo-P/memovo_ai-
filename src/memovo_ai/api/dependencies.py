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

from typing import Annotated

from fastapi import Depends, Request

from memovo_ai.core.config import (
    FAKE_VECTOR_PROVIDER,
    EmbeddingSettings,
    ProviderSettings,
    SearchSettings,
)
from memovo_ai.embeddings.base import EmbeddingProvider, ValidatedEmbeddingProvider
from memovo_ai.embeddings.models import ModelUnavailableError
from memovo_ai.embeddings.qwen3 import Qwen3EmbeddingProvider
from memovo_ai.providers.vector_search.base import VectorSearchProvider
from memovo_ai.providers.vector_search.fake import FakeVectorSearchProvider
from memovo_ai.services.memory_search import MemorySearchService

__all__ = [
    "SearchService",
    "build_embedding_provider",
    "build_memory_search_service",
    "build_vector_search_provider",
    "get_memory_search_service",
]


def build_embedding_provider(settings: EmbeddingSettings | None = None) -> EmbeddingProvider:
    """Load the embedding model once and wrap it in output validation.

    Raises:
        ModelUnavailableError: if the runtime is missing or the model fails
            to load.
    """
    resolved = settings if settings is not None else EmbeddingSettings()

    return ValidatedEmbeddingProvider(
        Qwen3EmbeddingProvider.load(resolved),
        dimension=resolved.dimension,
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


def build_memory_search_service(
    *,
    embedding_settings: EmbeddingSettings | None = None,
    provider_settings: ProviderSettings | None = None,
    search_settings: SearchSettings | None = None,
) -> MemorySearchService:
    """Assemble the search service from configuration."""
    return MemorySearchService(
        embedding_provider=build_embedding_provider(embedding_settings),
        vector_search=build_vector_search_provider(provider_settings),
        settings=search_settings if search_settings is not None else SearchSettings(),
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


SearchService = Annotated[MemorySearchService, Depends(get_memory_search_service)]
