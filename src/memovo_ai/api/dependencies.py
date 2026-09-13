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

import logging
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Request

from memovo_ai.core.config import (
    ATLAS_VECTOR_PROVIDER,
    FAKE_VECTOR_PROVIDER,
    OPENROUTER_GENERATION_PROVIDER,
    PRODUCTION_ENV,
    AtlasSettings,
    EmbeddingSettings,
    GenerationSettings,
    InvalidConfigurationError,
    OpenRouterSettings,
    ProviderSettings,
    RuntimeSettings,
    SearchSettings,
)
from memovo_ai.core.logging import log_event
from memovo_ai.embeddings.base import (
    EmbeddingProvider,
    NormalizingEmbeddingProvider,
    ValidatedEmbeddingProvider,
)
from memovo_ai.embeddings.models import ModelUnavailableError
from memovo_ai.embeddings.qwen3 import Qwen3EmbeddingProvider
from memovo_ai.generation.base import BoundedGenerationProvider, GenerationProvider
from memovo_ai.generation.models import GenerationError
from memovo_ai.providers.generation.openrouter import OpenRouterGenerationProvider
from memovo_ai.providers.vector_search.atlas import MongoAtlasVectorSearchProvider
from memovo_ai.providers.vector_search.base import VectorSearchProvider
from memovo_ai.providers.vector_search.fake import FakeVectorSearchProvider
from memovo_ai.services.memory_chat import MemoryChatService
from memovo_ai.services.memory_processor import MemoryProcessorService
from memovo_ai.services.memory_search import MemorySearchService
from memovo_ai.services.note_preparation import NotePreparationService

_logger = logging.getLogger(__name__)

__all__ = [
    "ChatService",
    "NotePreparation",
    "ProcessorService",
    "SearchService",
    "Services",
    "build_embedding_provider",
    "build_generation_provider",
    "build_services",
    "build_vector_search_provider",
    "get_memory_chat_service",
    "get_memory_processor_service",
    "get_memory_search_service",
    "get_note_preparation_service",
    "validate_configuration",
]


@dataclass(frozen=True, slots=True)
class Services:
    """The application services, sharing one loaded embedding model.

    ``vector_search`` is kept here purely so startup can hand it back at
    shutdown. An adapter that holds a database client needs closing, and the
    search service deliberately does not expose the provider it wraps.
    """

    memory_search: MemorySearchService
    memory_processor: MemoryProcessorService
    vector_search: VectorSearchProvider
    #: ``None`` when generation is disabled or could not be configured. The
    #: retrieval services above never depend on it.
    generation: GenerationProvider | None = None
    #: Built whenever the retrieval services are; they hold ``generation``
    #: and answer GENERATION_UNAVAILABLE themselves while it is ``None``.
    memory_chat: MemoryChatService | None = None
    note_preparation: NotePreparationService | None = None

    async def aclose(self) -> None:
        """Release anything the providers hold open.

        ``aclose`` is optional on the protocols -- the in-memory fakes have
        nothing to release -- so it is called only when present.
        """
        for provider in (self.vector_search, self.generation):
            close = getattr(provider, "aclose", None)
            if close is not None:
                await close()


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


def validate_configuration(
    provider: ProviderSettings | None = None,
    runtime: RuntimeSettings | None = None,
) -> None:
    """Refuse settings the service must not run with.

    Pure and cheap: no model load, no connection, no side effect. That is
    what lets it run *before* anything expensive, so a misconfigured
    deployment is told immediately rather than after a minute of loading a
    model it was never going to use correctly.

    Raises:
        InvalidConfigurationError: if ``fake`` is selected under
            ``MEMOVO_ENV=production``.
    """
    resolved = provider if provider is not None else ProviderSettings()
    environment = runtime if runtime is not None else RuntimeSettings()

    # The trap this closes. `fake` is the development default, so an
    # *unset* MEMOVO_VECTOR_PROVIDER -- including the case where the variable
    # NAME was mistyped, leaving the real one unset -- selects it silently.
    # The service then loads its model, reports itself ready, answers 200,
    # and returns "no memories found" for every query. Nothing about that
    # looks broken from the outside.
    #
    # A mistyped provider *value* is a different and safer failure: it
    # matches no known provider and raises below.
    if resolved.vector_provider == FAKE_VECTOR_PROVIDER and environment.is_production:
        message = (
            f"the {FAKE_VECTOR_PROVIDER!r} vector provider is an empty in-memory index "
            f"and must not be used when MEMOVO_ENV={PRODUCTION_ENV}; "
            f"set MEMOVO_VECTOR_PROVIDER={ATLAS_VECTOR_PROVIDER!r} "
            f"and supply MEMOVO_ATLAS_URI"
        )
        raise InvalidConfigurationError(message)


def build_vector_search_provider(
    settings: ProviderSettings | None = None,
    runtime: RuntimeSettings | None = None,
) -> VectorSearchProvider:
    """Select the read-only vector search adapter.

    An unknown name fails loudly rather than silently falling back to an
    empty index and reporting "no memories" for every query.

    Raises:
        InvalidConfigurationError: if ``fake`` is selected under
            ``MEMOVO_ENV=production``. Checked here as well as in
            :func:`build_services`, so the guard holds for any caller of this
            function rather than only for the startup path.
    """
    resolved = settings if settings is not None else ProviderSettings()
    environment = runtime if runtime is not None else RuntimeSettings()

    validate_configuration(resolved, environment)

    if resolved.vector_provider == FAKE_VECTOR_PROVIDER:
        return FakeVectorSearchProvider()

    if resolved.vector_provider == ATLAS_VECTOR_PROVIDER:
        return MongoAtlasVectorSearchProvider.connect(AtlasSettings())

    known = f"{FAKE_VECTOR_PROVIDER!r}, {ATLAS_VECTOR_PROVIDER!r}"
    message = f"unknown vector provider {resolved.vector_provider!r}; known providers are {known}"
    raise ValueError(message)


def build_generation_provider(
    settings: GenerationSettings | None = None,
    openrouter: OpenRouterSettings | None = None,
) -> GenerationProvider | None:
    """Select the generation adapter, or ``None`` when generation is disabled.

    Disabled is the default and needs no credentials. Enabled means the
    approved OpenRouter model behind the concurrency/deadline wrapper.

    Raises:
        GenerationUnavailableError: enabled without an API key.
        ValueError: an unknown provider name. There is no fake provider name
            on purpose; a fake is constructed in code, never configured.
    """
    resolved = settings if settings is not None else GenerationSettings()
    if not resolved.enabled:
        return None

    if resolved.provider != OPENROUTER_GENERATION_PROVIDER:
        message = (
            f"unknown generation provider {resolved.provider!r}; "
            f"the only known provider is {OPENROUTER_GENERATION_PROVIDER!r}"
        )
        raise ValueError(message)

    adapter = OpenRouterGenerationProvider.connect(
        settings=openrouter if openrouter is not None else OpenRouterSettings(),
        model=resolved.model,
        timeout_seconds=resolved.timeout_seconds,
        retry_after_max_seconds=resolved.retry_after_max_seconds,
    )

    return BoundedGenerationProvider(
        adapter,
        max_concurrency=resolved.max_concurrency,
        timeout_seconds=resolved.timeout_seconds,
    )


def _tolerated_generation_provider(
    settings: GenerationSettings | None,
    openrouter: OpenRouterSettings | None,
) -> GenerationProvider | None:
    """Build generation, degrading to ``None`` rather than failing startup.

    A missing key or an unknown provider name must not take processing and
    search down with it: they never needed generation. The failure is
    logged by type, and the chat/preparation endpoints answer
    ``GENERATION_UNAVAILABLE`` until it is fixed.
    """
    try:
        return build_generation_provider(settings, openrouter)
    except (GenerationError, ValueError) as error:
        log_event(
            _logger,
            "generation.startup_failed",
            level=logging.ERROR,
            error_type=type(error).__name__,
        )
        return None


def build_services(
    *,
    embedding_settings: EmbeddingSettings | None = None,
    provider_settings: ProviderSettings | None = None,
    search_settings: SearchSettings | None = None,
    runtime_settings: RuntimeSettings | None = None,
    generation_settings: GenerationSettings | None = None,
    openrouter_settings: OpenRouterSettings | None = None,
) -> Services:
    """Assemble every service from configuration.

    The embedding provider is built once and shared, so the model is loaded a
    single time no matter how many services use it (doc 02, section 7).

    Configuration is validated first, before the model is loaded. A setting
    the service must refuse should be reported in the first second of startup,
    not after a gigabyte of weights has been read from disk -- and if both are
    wrong, the misconfiguration is the one an operator needs to see.
    """
    provider = provider_settings if provider_settings is not None else ProviderSettings()
    environment = runtime_settings if runtime_settings is not None else RuntimeSettings()
    generation_config = (
        generation_settings if generation_settings is not None else GenerationSettings()
    )

    validate_configuration(provider, environment)

    embeddings = build_embedding_provider(embedding_settings)
    vector_search = build_vector_search_provider(provider, environment)
    generation = _tolerated_generation_provider(generation_config, openrouter_settings)

    memory_search = MemorySearchService(
        embedding_provider=embeddings,
        vector_search=vector_search,
        settings=search_settings if search_settings is not None else SearchSettings(),
    )

    return Services(
        memory_search=memory_search,
        memory_processor=MemoryProcessorService(embedding_provider=embeddings),
        vector_search=vector_search,
        generation=generation,
        memory_chat=MemoryChatService(
            search=memory_search, generation=generation, settings=generation_config
        ),
        note_preparation=NotePreparationService(generation=generation, settings=generation_config),
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


def get_memory_chat_service(request: Request) -> MemoryChatService:
    """Return the chat service built at startup.

    Raises:
        ModelUnavailableError: if startup could not build the services at
            all. A built service whose generation is disabled answers
            ``GENERATION_UNAVAILABLE`` itself.
    """
    service: MemoryChatService | None = getattr(request.app.state, "memory_chat_service", None)

    if service is None:
        message = "services are not available"
        raise ModelUnavailableError(message)

    return service


def get_note_preparation_service(request: Request) -> NotePreparationService:
    """Return the note preparation service built at startup."""
    service: NotePreparationService | None = getattr(
        request.app.state, "note_preparation_service", None
    )

    if service is None:
        message = "services are not available"
        raise ModelUnavailableError(message)

    return service


SearchService = Annotated[MemorySearchService, Depends(get_memory_search_service)]
ProcessorService = Annotated[MemoryProcessorService, Depends(get_memory_processor_service)]
ChatService = Annotated[MemoryChatService, Depends(get_memory_chat_service)]
NotePreparation = Annotated[NotePreparationService, Depends(get_note_preparation_service)]
