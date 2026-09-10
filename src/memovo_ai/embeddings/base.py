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
    InvalidEmbeddingError,
    l2_norm,
    validate_embedding,
    validate_embeddings,
)

__all__ = [
    "EmbeddingProvider",
    "NormalizingEmbeddingProvider",
    "ValidatedEmbeddingProvider",
]


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


def _to_unit_length(embedding: Embedding, *, position: int | None = None) -> Embedding:
    """Scale a vector to exactly unit length, in full float precision."""
    norm = l2_norm(embedding)

    if norm == 0.0:
        where = "" if position is None else f" at position {position}"
        message = f"embedding{where} has zero magnitude and cannot be normalized"
        raise InvalidEmbeddingError(message)

    return [value / norm for value in embedding]


class NormalizingEmbeddingProvider:
    """Rescales every embedding to exactly unit length.

    Qwen3-Embedding already ends its module stack with a ``Normalize`` layer,
    so its output is *semantically* normalized. The model runs in bfloat16
    though, so that normalization is only accurate to about three significant
    digits: measured L2 norms land in roughly ``1.0 +/- 0.003``. Passing
    ``normalize_embeddings=True`` to the encoder does not help, because the
    rounding happens after the model's own normalization -- measured, it is a
    no-op.

    Rescaling here in Python float (float64) closes that gap exactly, which is
    what lets :class:`ValidatedEmbeddingProvider` enforce ``expect_unit_norm``
    at its existing tolerance rather than having the tolerance widened to
    accommodate model precision.

    It also makes the similarity contract unambiguous: with true unit vectors,
    cosine similarity and inner product agree, so the vector index cannot be
    misconfigured into scoring differently from the service.

    Direction is untouched -- scaling changes no cosine similarity. Wrap the
    concrete provider with this, then wrap that in validation.
    """

    __slots__ = ("_provider",)

    def __init__(self, provider: EmbeddingProvider) -> None:
        self._provider = provider

    async def embed_documents(self, texts: Sequence[str]) -> list[Embedding]:
        embeddings = await self._provider.embed_documents(texts)

        return [
            _to_unit_length(embedding, position=position)
            for position, embedding in enumerate(embeddings)
        ]

    async def embed_query(self, query: str) -> Embedding:
        return _to_unit_length(await self._provider.embed_query(query))
