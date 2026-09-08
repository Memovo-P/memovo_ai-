"""Retrieval domain models.

Doc 04 assigns vector hit models to ``retrieval/``. Filtering (Phase 11),
ranking and deduplication (Phase 12) and orchestration (Phase 13) build on
this type; none of them exist yet.
"""

import math
from dataclasses import dataclass

__all__ = ["RankedMemory", "VectorSearchHit"]


@dataclass(frozen=True, slots=True)
class VectorSearchHit:
    """One matching chunk returned by a vector search provider.

    Carries everything needed to build a search result without a second
    lookup: the AI Service never reads MongoDB, so the Memory's title and
    tags have to travel with the hit.

    ``user_id`` is the owner recorded in the vector index. It exists so
    isolation can be *verified* after the fact, not so it can be filtered on:
    a hit belonging to anyone but the requesting user means the pre-filter is
    broken, and
    :class:`~memovo_ai.providers.vector_search.base.UserScopedVectorSearchProvider`
    raises rather than quietly discarding it. It is never serialized into an
    API response.

    ``score`` is whatever similarity the vector engine reports. No range is
    enforced -- the similarity metric is recommended but not locked (doc 01,
    section 7), so bounding it here would harden a decision the architecture
    has not made.

    ``tags`` is a tuple so a hit cannot be mutated after the provider
    returns it.
    """

    user_id: str
    memory_id: str
    chunk_id: str
    chunk_index: int
    content: str
    score: float
    title: str
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.chunk_index < 0:
            message = f"chunk_index must not be negative, got {self.chunk_index}"
            raise ValueError(message)
        if not math.isfinite(self.score):
            message = "score must be a finite number"
            raise ValueError(message)


@dataclass(frozen=True, slots=True)
class RankedMemory:
    """One unique Memory in the retrieval result set.

    Produced by :func:`~memovo_ai.retrieval.ranking.rank_memories` after
    grouping the relevant chunks by ``memory_id``. Each Memory appears exactly
    once (doc 01, decision 17).

    ``score`` is the maximum score across the Memory's matching chunks (doc 01,
    decision 18) -- not a sum or a mean, either of which would let a Memory
    with many weak chunks outrank one with a single strong match.

    ``chunks`` holds every matching chunk that survived the threshold, in
    reading order.
    """

    memory_id: str
    score: float
    title: str
    tags: tuple[str, ...]
    chunks: tuple[VectorSearchHit, ...]

    def __post_init__(self) -> None:
        if not self.chunks:
            message = f"ranked memory {self.memory_id} must have at least one chunk"
            raise ValueError(message)
        if not math.isfinite(self.score):
            message = "score must be a finite number"
            raise ValueError(message)
