"""Grouping, deduplication and ranking.

The last step before a search response is assembled (doc 01, Flow B)::

    userId pre-filter -> Top-K -> threshold -> deduplicate -> rank

Relevant chunks are grouped by ``memory_id``, each Memory scores the maximum
of its matching chunks, and the Memories come back in descending score order,
each appearing exactly once (doc 01, decisions 17-19).

Worked example from doc 03, Phase 12::

    Memory A / Chunk 0 -> 0.94      Memory A -> 0.94
    Memory A / Chunk 1 -> 0.91  =>  Memory B -> 0.89
    Memory B / Chunk 0 -> 0.89

Maximum, not sum or mean: summing would let a Memory with several weak chunks
outrank one with a single strong match, and averaging would penalise a long
Memory for the chunks that happen not to match.
"""

from collections.abc import Sequence

from memovo_ai.retrieval.models import RankedMemory, VectorSearchHit

__all__ = ["rank_memories"]


def _chunk_order(hit: VectorSearchHit) -> tuple[int, str]:
    """Reading order within a Memory.

    Chunks are returned in ``chunkIndex`` order rather than by score. The
    public chunk shape carries no score field, so a score ordering would be
    invisible and unexplainable to the Backend, whereas ``chunkIndex`` is
    right there in the payload. ``chunk_id`` breaks any tie so the order is
    total.
    """
    return (hit.chunk_index, hit.chunk_id)


def _relevance_order(hit: VectorSearchHit) -> tuple[float, int, str]:
    """Best chunk first, deterministically even when scores tie."""
    return (-hit.score, hit.chunk_index, hit.chunk_id)


def rank_memories(hits: Sequence[VectorSearchHit]) -> list[RankedMemory]:
    """Group relevant chunks into unique Memories, best Memory first.

    ``hits`` should already be threshold-filtered (Phase 11); this function
    applies no score cutoff of its own.

    Ties on the Memory score break on ``memory_id``, so an equal-scoring pair
    orders identically across runs instead of depending on the order the
    vector engine happened to return.

    The input is not mutated, and no chunk is dropped: every hit given ends up
    inside exactly one returned Memory.
    """
    grouped: dict[str, list[VectorSearchHit]] = {}
    for hit in hits:
        grouped.setdefault(hit.memory_id, []).append(hit)

    ranked = [_build_memory(memory_id, memory_hits) for memory_id, memory_hits in grouped.items()]
    ranked.sort(key=lambda memory: (-memory.score, memory.memory_id))

    return ranked


def _build_memory(memory_id: str, hits: list[VectorSearchHit]) -> RankedMemory:
    # Title and tags are taken from the best-matching chunk. Every chunk of a
    # Memory should carry the same values; picking the representative one
    # deterministically means a stale index cannot make the result depend on
    # which chunk the engine returned first.
    best = min(hits, key=_relevance_order)

    return RankedMemory(
        memory_id=memory_id,
        score=best.score,
        title=best.title,
        tags=best.tags,
        chunks=tuple(sorted(hits, key=_chunk_order)),
    )
