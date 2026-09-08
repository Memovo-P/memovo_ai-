"""Chunk models.

These are internal domain models, not the public contract. The public chunk
shape lives in :mod:`memovo_ai.schemas`. A :class:`Chunk` carries no
``chunkId`` (Phase 04 derives it from ``chunkIndex`` and final content) and no
embedding (Phase 06).
"""

from dataclasses import dataclass

__all__ = ["Chunk", "ChunkingConfig"]

#: Locked by doc 01, decision 5. Soft target, not a hard cut.
DEFAULT_TARGET_TOKENS = 500
#: Locked by doc 01, decision 6.
DEFAULT_OVERLAP_TOKENS = 50


@dataclass(frozen=True, slots=True)
class Chunk:
    """One chunk of a Memory's canonical content.

    ``content`` is final: Phase 04 hashes exactly this string, so any change
    to it changes the chunk ID.
    """

    index: int
    content: str
    token_count: int


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    """Chunking parameters.

    Defaults are the locked Sprint 1 values. These are deliberately not
    exposed as ``MEMOVO_`` environment variables: no source document defines
    chunking env settings, and doc 03 presents them as code-level config.
    """

    target_tokens: int = DEFAULT_TARGET_TOKENS
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS

    def __post_init__(self) -> None:
        if self.target_tokens < 1:
            message = "target_tokens must be at least 1"
            raise ValueError(message)
        if self.overlap_tokens < 0:
            message = "overlap_tokens must not be negative"
            raise ValueError(message)
        if self.overlap_tokens >= self.target_tokens:
            message = "overlap_tokens must be smaller than target_tokens"
            raise ValueError(message)
