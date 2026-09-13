"""Runtime configuration.

Settings are read from the environment (and a local ``.env``) with the
``MEMOVO_`` prefix, matching doc 06, section 10. Each concern owns a small
settings class with its own prefix, rather than one object that grows with
every phase.
"""

import logging
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = [
    "ATLAS_VECTOR_PROVIDER",
    "DEFAULT_EMBEDDING_DIMENSION",
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_ENV",
    "DEFAULT_GENERATION_MODEL",
    "DEFAULT_OPENROUTER_BASE_URL",
    "DEFAULT_SEARCH_SIMILARITY_THRESHOLD",
    "DEFAULT_SEARCH_TOP_K",
    "FAKE_VECTOR_PROVIDER",
    "JSON_LOG_FORMAT",
    "OPENROUTER_GENERATION_PROVIDER",
    "PRODUCTION_ENV",
    "TEXT_LOG_FORMAT",
    "AtlasSettings",
    "EmbeddingSettings",
    "GenerationSettings",
    "InvalidConfigurationError",
    "LoggingSettings",
    "OpenRouterSettings",
    "ProviderSettings",
    "RuntimeSettings",
    "SearchSettings",
]


class InvalidConfigurationError(Exception):
    """The service is configured in a way it must refuse to run with.

    Distinct from the failures ``main`` tolerates. A missing model or an
    unreachable Atlas cluster can resolve on its own, so the service starts
    unready and says so; a wrong *setting* cannot, and a process that keeps
    serving with one is the more dangerous outcome. Deliberately not a
    ``ValueError``, because startup catches that one and degrades instead of
    failing.
    """


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
    #: Empty disables it.
    #:
    #: Enabled after the Phase 06 benchmark confirmed the model defines it
    #: ("Instruct: Given a web search query, retrieve relevant passages...")
    #: and that it improves separation: mean relevant-to-best-unrelated margin
    #: 0.19 -> 0.24, worst unrelated score 0.657 -> 0.621, worst no-match
    #: score 0.319 -> 0.268, Arabic relevant mean 0.630 -> 0.678.
    query_prompt_name: str = "query"

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


#: Deployment environment names. ``production`` is the only one that carries
#: behaviour: it refuses configurations that are fine to develop against but
#: would silently serve nothing in front of real users.
PRODUCTION_ENV = "production"
DEFAULT_ENV = "development"


class RuntimeSettings(BaseSettings):
    """Which deployment environment this process is running in.

    Reads ``MEMOVO_ENV``, which doc 06 section 10 already lists. Until now
    nothing consumed it; the production guard on the vector provider is its
    first use.
    """

    model_config = SettingsConfigDict(
        env_prefix="MEMOVO_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    env: str = DEFAULT_ENV

    @property
    def is_production(self) -> bool:
        """Case- and whitespace-insensitive, because this gates a guard.

        ``MEMOVO_ENV=Production`` must not quietly fall through to the
        development path -- the whole point of the check is that a small
        configuration mistake cannot disable it.
        """
        return self.env.strip().lower() == PRODUCTION_ENV


#: In-memory adapter for development and tests.
FAKE_VECTOR_PROVIDER = "fake"
#: MongoDB Atlas Vector Search.
ATLAS_VECTOR_PROVIDER = "atlas"


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


class AtlasSettings(BaseSettings):
    """MongoDB Atlas Vector Search connection and index configuration.

    Environment variables use the ``MEMOVO_ATLAS_`` prefix. The connection
    string carries credentials, so it is a :class:`~pydantic.SecretStr`: it
    never appears in a repr, a log line or a traceback. Nothing here has a
    credential-bearing default -- an unset ``uri`` fails loudly at startup
    rather than silently connecting somewhere unintended.
    """

    model_config = SettingsConfigDict(
        env_prefix="MEMOVO_ATLAS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    #: Full connection string, e.g. ``mongodb+srv://user:pass@cluster/``.
    uri: SecretStr = SecretStr("")

    database: str = "memovo"

    #: Collection the Backend writes chunk documents into (contract
    #: section 12). Overridable, because the Backend provisions Atlas.
    collection: str = "memory_vectors"

    #: Name of the Atlas Vector Search index on that collection (contract
    #: section 12).
    index: str = "vector_index"

    #: Document field holding the 1024-dimension vector.
    path: str = "embedding"

    #: Atlas explores ``numCandidates`` nodes before returning ``limit``
    #: results. Too low costs recall; too high costs latency. Atlas requires
    #: it to be at least the limit, and recommends a healthy multiple.
    num_candidates_multiplier: int = Field(default=10, ge=1)

    #: Floor on numCandidates, so a Top-K of 5 still explores enough of the
    #: graph to find good neighbours.
    min_num_candidates: int = Field(default=100, ge=1)

    #: Applied to server selection, connection and the query itself.
    timeout_ms: int = Field(default=5_000, ge=1)

    @property
    def is_configured(self) -> bool:
        return bool(self.uri.get_secret_value().strip())

    def num_candidates(self, top_k: int) -> int:
        return max(top_k * self.num_candidates_multiplier, self.min_num_candidates)


#: One JSON object per line, for log aggregation. The production default.
JSON_LOG_FORMAT: Literal["json"] = "json"
#: Flat ``key=value`` lines, easier to read while developing.
TEXT_LOG_FORMAT: Literal["text"] = "text"


class LoggingSettings(BaseSettings):
    """Operational logging configuration (doc 03, Phase 18).

    Environment variables use the ``MEMOVO_LOG_`` prefix, so ``level`` reads
    from ``MEMOVO_LOG_LEVEL``.

    There is deliberately no switch to log request or Memory content. Doc 06
    section 5 forbids it outright, and a setting that could turn it on would
    make the guarantee a configuration accident away from failing.
    """

    model_config = SettingsConfigDict(
        env_prefix="MEMOVO_LOG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    #: Standard ``logging`` level name.
    level: str = "INFO"

    #: ``json`` or ``text``.
    format: Literal["json", "text"] = JSON_LOG_FORMAT

    #: Mixed into the ``userId`` hash. A hash of a low-entropy identifier can
    #: be confirmed by guessing candidates; a per-deployment salt removes
    #: that. Empty means unsalted, which still keeps raw identifiers out of
    #: logs. Secret because it is only useful while it stays unknown.
    user_salt: SecretStr = SecretStr("")

    @property
    def level_number(self) -> int:
        """The configured level, falling back to ``INFO`` if unrecognised.

        An unknown level must not stop the service from starting: losing log
        detail is recoverable, refusing to boot over a typo is not.
        """
        resolved = logging.getLevelName(self.level.strip().upper())

        return resolved if isinstance(resolved, int) else logging.INFO


#: The only generation provider (plan section 5). There is no "fake" name on
#: purpose: fake generation is constructed in code, never selected by config,
#: so a production deployment cannot serve fabricated answers by mistake.
OPENROUTER_GENERATION_PROVIDER = "openrouter"
#: The approved model, exactly. No paid, alternate or ``openrouter/free``
#: fallback without new owner approval.
DEFAULT_GENERATION_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class GenerationSettings(BaseSettings):
    """Answer generation configuration (``MEMOVO_GENERATION_`` prefix).

    Disabled by default. Processing and search never depend on any of this;
    only Memory Chat and note preparation do.

    The numeric limits are provisional operating defaults chosen to be safe,
    not measured targets. They are re-baselined against the real endpoint
    (plan P6/P7) and are all overridable.
    """

    model_config = SettingsConfigDict(
        env_prefix="MEMOVO_GENERATION_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    enabled: bool = False
    provider: str = OPENROUTER_GENERATION_PROVIDER
    model: str = DEFAULT_GENERATION_MODEL

    #: Deadline for one generation call, including any wait for a slot.
    timeout_seconds: float = Field(default=60.0, gt=0)
    #: In-flight generation calls per process.
    max_concurrency: int = Field(default=4, ge=1)
    #: Hard caps forwarded to the provider as ``max_tokens``, one per
    #: operation: a chat answer is short, a prepared note can be a whole
    #: cleaned document (up to the 10,000-character Note limit).
    chat_max_output_tokens: int = Field(default=1024, ge=1)
    note_max_output_tokens: int = Field(default=4096, ge=1)
    #: The model context the budget check assumes, and the conservative
    #: characters-per-token estimate used because the generator tokenizer
    #: is not available in-process. Retain the conservative 40,960-token
    #: operating budget during the model migration; this is not a claim
    #: about the provider's full context. Over-budget requests fail, never trim.
    context_window_tokens: int = Field(default=40_960, ge=1)
    chars_per_token: float = Field(default=2.0, gt=0)
    #: Upper bound on a forwarded ``Retry-After`` (contract section 21.2.1).
    retry_after_max_seconds: int = Field(default=30, ge=0)


class OpenRouterSettings(BaseSettings):
    """OpenRouter connection (``MEMOVO_OPENROUTER_`` prefix).

    The key is a :class:`~pydantic.SecretStr`: it never appears in a repr, a
    log line or a traceback. There is no default key; an empty value means
    generation cannot be enabled, and the service says so at startup.
    """

    model_config = SettingsConfigDict(
        env_prefix="MEMOVO_OPENROUTER_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    base_url: str = DEFAULT_OPENROUTER_BASE_URL
    api_key: SecretStr = SecretStr("")

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key.get_secret_value().strip())
