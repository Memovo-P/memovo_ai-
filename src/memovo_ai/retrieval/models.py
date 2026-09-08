"""Retrieval domain models.

Doc 04 assigns vector hit models to ``retrieval/``. Filtering (Phase 11),
ranking and deduplication (Phase 12) and orchestration (Phase 13) build on
this type; none of them exist yet.
"""

import math
from dataclasses import dataclass

__all__ = ["VectorSearchHit"]


@dataclass(frozen=True, slots=True)
class VectorSearchHit:
    """One matching chunk returned by a vector search provider.

    Carries everything needed to build a search result without a second
    lookup: the AI Service never reads MongoDB, so the Memory's title and
    tags have to travel with the hit.

    ``score`` is whatever similarity the vector engine reports. No range is
    enforced -- the similarity metric is recommended but not locked (doc 01,
    section 7), so bounding it here would harden a decision the architecture
    has not made.

    ``tags`` is a tuple so a hit cannot be mutated after the provider
    returns it.
    """

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
