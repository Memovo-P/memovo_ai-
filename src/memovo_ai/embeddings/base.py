"""The embedding provider boundary.

Application services depend on :class:`EmbeddingProvider`, never on a concrete
model runtime. ``Qwen3EmbeddingProvider`` implements this interface at Phase 06.

Query and document embedding stay separate methods even where an
implementation shares internal logic (doc 02, section 6): Qwen3-style models
apply an instruction prefix to queries but not to documents, so collapsing them
would bake a wrong assumption into the interface.
"""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from memovo_ai.embeddings.models import (
    EMBEDDING_DIMENSION,
    Embedding,
    validate_embedding,
    validate_embeddings,
)

__all__ = ["EmbeddingProvider", "ValidatedEmbeddingProvider"]


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Produces embeddings for documents and for queries."""

    async def embed_documents(self, texts: Sequence[str]) -> list[Embedding]:
        """Embed chunk texts, returning one embedding per input, in order."""
        ...

    async def embed_query(self, query: str) -> Embedding:
        """Embed a single search query."""
        ...


class ValidatedEmbeddingProvider:
    """Wraps a provider so every output is validated before it escapes.

    Doc 03, Phase 05 requires validating *every* output. Enforcing it at the
    boundary rather than inside each implementation means a new provider --
    or a test double -- cannot forget to do it, and the check lives in one
    place. This is itself an :class:`EmbeddingProvider`, so it is transparent
    to callers.
    """

    __slots__ = ("_dimension", "_expect_unit_norm", "_provider")

    def __init__(
        self,
        provider: EmbeddingProvider,
        *,
        dimension: int = EMBEDDING_DIMENSION,
        expect_unit_norm: bool = False,
    ) -> None:
        self._provider = provider
        self._dimension = dimension
        self._expect_unit_norm = expect_unit_norm

    async def embed_documents(self, texts: Sequence[str]) -> list[Embedding]:
        embeddings = await self._provider.embed_documents(texts)
        return validate_embeddings(
            embeddings,
            expected_count=len(texts),
            dimension=self._dimension,
            expect_unit_norm=self._expect_unit_norm,
        )

    async def embed_query(self, query: str) -> Embedding:
        embedding = await self._provider.embed_query(query)
        return validate_embedding(
            embedding,
            dimension=self._dimension,
            expect_unit_norm=self._expect_unit_norm,
        )
