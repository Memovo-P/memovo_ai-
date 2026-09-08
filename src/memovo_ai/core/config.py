"""Runtime configuration.

Settings are read from the environment (and a local ``.env``) with the
``MEMOVO_`` prefix, matching doc 06, section 10. Only embedding settings exist
so far; later phases add their own sections rather than one growing object.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = [
    "DEFAULT_EMBEDDING_DIMENSION",
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_SEARCH_SIMILARITY_THRESHOLD",
    "DEFAULT_SEARCH_TOP_K",
    "FAKE_VECTOR_PROVIDER",
    "EmbeddingSettings",
    "ProviderSettings",
    "SearchSettings",
]

#: Locked by the Sprint 1 model decision.
DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"

#: Locked by doc 01, decision 8. Deliberately a literal rather than an import
#: of ``embeddings.models.EMBEDDING_DIMENSION``: ``core`` sits below
#: ``embeddings`` in the dependency order, and the embedding provider reads
#: this module. A test asserts the two stay in agreement.
DEFAULT_EMBEDDING_DIMENSION = 1024


class EmbeddingSettings(BaseSettings):
    """Configuration for the Qwen3 embedding provider.

    Environment variables use the ``MEMOVO_EMBEDDING_`` prefix, so ``model``
    reads from ``MEMOVO_EMBEDDING_MODEL`` and so on.
    """

    model_config = SettingsConfigDict(
        env_prefix="MEMOVO_EMBEDDING_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    #: Hugging Face repository id of the embedding model.
    model: str = DEFAULT_EMBEDDING_MODEL

    #: Expected output dimension. Changing this means a different model and a
    #: different vector index, not merely a different runtime setting; it is
    #: exposed because doc 06 lists it, and the provider validates against it.
    dimension: int = Field(default=DEFAULT_EMBEDDING_DIMENSION, ge=1)

    #: ``auto`` lets the runtime choose (CUDA when present, otherwise CPU).
    #: Any other value is passed through verbatim, e.g. ``cpu`` or ``cuda:0``.
    device: str = "auto"

    #: Chunks embedded per forward pass.
    batch_size: int = Field(default=32, ge=1)

    #: Name of the model's query instruction prompt, applied to queries only.
    #: Empty disables it. Qwen3-Embedding's card recommends a query
    #: instruction, but enabling it by default would ship behaviour this
    #: repository cannot verify without downloading weights -- the embedding
    #: integration test reports whether the prompt exists, so the default can
    #: be flipped on evidence.
    query_prompt_name: str = ""

    @property
    def resolved_device(self) -> str | None:
        """``None`` when the device should be auto-selected by the runtime."""
        return None if self.device == "auto" else self.device


#: Locked by doc 01, decisions 10 and 9. Literals rather than imports of
#: ``retrieval.filtering``: ``core`` sits below ``retrieval``, and Phase 13's
#: search service will read this module. Tests assert the values agree.
DEFAULT_SEARCH_TOP_K = 5
DEFAULT_SEARCH_SIMILARITY_THRESHOLD = 0.75


class SearchSettings(BaseSettings):
    """Configuration for memory retrieval.

    Environment variables use the ``MEMOVO_SEARCH_`` prefix, so ``top_k``
    reads from ``MEMOVO_SEARCH_TOP_K``.

    The defaults are the locked MVP values. Doc 05 section 13 evaluates
    thresholds from 0.60 to 0.85, and these settings exist so that evaluation
    can sweep them -- not so production can drift away from 0.75 without a
    product decision.
    """

    model_config = SettingsConfigDict(
        env_prefix="MEMOVO_SEARCH_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    #: Chunks requested from the vector engine per query.
    top_k: int = Field(default=DEFAULT_SEARCH_TOP_K, ge=1)

    #: Minimum score for a chunk to count as relevant. Compared inclusively.
    #: No range is imposed: the similarity metric is recommended but not
    #: locked (doc 01, section 7), and an unbounded metric such as inner
    #: product would not fit ``[0, 1]``.
    similarity_threshold: float = Field(
        default=DEFAULT_SEARCH_SIMILARITY_THRESHOLD,
        allow_inf_nan=False,
    )


#: Only the in-memory fake exists. The production vector database has not been
#: selected (doc 01, section 8), so no adapter is registered for it yet.
FAKE_VECTOR_PROVIDER = "fake"


class ProviderSettings(BaseSettings):
    """Which infrastructure adapters to build at startup.

    Uses the bare ``MEMOVO_`` prefix because doc 06 section 10 names the
    variable ``MEMOVO_VECTOR_PROVIDER``, not ``MEMOVO_PROVIDER_*``.
    """

    model_config = SettingsConfigDict(
        env_prefix="MEMOVO_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    #: Read-only vector search adapter. ``fake`` is the development default.
    vector_provider: str = FAKE_VECTOR_PROVIDER
