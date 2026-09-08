"""In-memory vector search provider.

Used by unit, integration, ranking and security tests, and for development
before the production vector database is chosen (doc 01, section 8).

It is a real implementation of the read-only contract, not a stub: it applies
the ``user_id`` pre-filter *before* scoring, then ranks and truncates to
``top_k``. Ordering the two steps that way is what makes the Phase 10 isolation
tests meaningful -- a fake that scored first and filtered second would pass a
post-filtering implementation just as happily.

Records are supplied at construction and stored immutably. There is
deliberately no ``add`` or ``delete`` method: the fake cannot be written to,
which mirrors the read-only boundary exactly.
"""

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from memovo_ai.retrieval.models import VectorSearchHit

__all__ = ["FakeVectorSearchProvider", "SearchCall", "VectorRecord"]


@dataclass(frozen=True, slots=True)
class VectorRecord:
    """One indexed chunk, as the Backend would have written it.

    Supply either ``embedding`` (cosine similarity is computed against the
    query) or ``score`` (used verbatim). The fixed-score form exists so
    ranking and deduplication tests can state the scores they mean instead of
    hand-crafting vectors that happen to produce them.
    """

    user_id: str
    memory_id: str
    chunk_id: str
    chunk_index: int
    content: str
    title: str = ""
    tags: tuple[str, ...] = ()
    embedding: tuple[float, ...] | None = None
    score: float | None = None

    def __post_init__(self) -> None:
        if self.embedding is None and self.score is None:
            message = f"record {self.chunk_id} needs either an embedding or a fixed score"
            raise ValueError(message)
        if self.chunk_index < 0:
            message = f"chunk_index must not be negative, got {self.chunk_index}"
            raise ValueError(message)


@dataclass(frozen=True, slots=True)
class SearchCall:
    """A recorded invocation, so tests can assert what the engine was asked.

    Phase 10 uses this to prove the pre-filter reached the provider rather
    than being applied afterwards in application code.
    """

    user_id: str
    query_embedding: tuple[float, ...]
    top_k: int


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        message = f"embedding dimension mismatch: query {len(left)}, record {len(right)}"
        raise ValueError(message)

    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))

    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0

    return dot / (left_norm * right_norm)


@dataclass(eq=False)
class FakeVectorSearchProvider:
    """A read-only, in-memory :class:`VectorSearchProvider`."""

    records: tuple[VectorRecord, ...] = ()
    calls: list[SearchCall] = field(default_factory=list)

    def __init__(self, records: Iterable[VectorRecord] = ()) -> None:
        self.records = tuple(records)
        self.calls = []

    async def search(
        self,
        *,
        user_id: str,
        query_embedding: Sequence[float],
        top_k: int,
    ) -> list[VectorSearchHit]:
        if top_k < 1:
            message = f"top_k must be at least 1, got {top_k}"
            raise ValueError(message)

        self.calls.append(
            SearchCall(user_id=user_id, query_embedding=tuple(query_embedding), top_k=top_k)
        )

        # Pre-filter FIRST. Everything below only ever sees this user's chunks.
        owned = [record for record in self.records if record.user_id == user_id]

        scored = [(self._score(record, query_embedding), record) for record in owned]

        # Sort by score descending, then by chunk id, so equal scores order
        # deterministically instead of depending on insertion order.
        scored.sort(key=lambda pair: (-pair[0], pair[1].chunk_id))

        return [
            VectorSearchHit(
                user_id=record.user_id,
                memory_id=record.memory_id,
                chunk_id=record.chunk_id,
                chunk_index=record.chunk_index,
                content=record.content,
                score=score,
                title=record.title,
                tags=record.tags,
            )
            for score, record in scored[:top_k]
        ]

    @staticmethod
    def _score(record: VectorRecord, query_embedding: Sequence[float]) -> float:
        if record.score is not None:
            return record.score

        assert record.embedding is not None  # noqa: S101 - guaranteed by VectorRecord
        return _cosine_similarity(query_embedding, record.embedding)
