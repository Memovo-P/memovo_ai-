"""Grouping, deduplication and ranking of relevant chunks."""

import random

import pytest

from memovo_ai.retrieval import (
    RankedMemory,
    VectorSearchHit,
    filter_by_threshold,
    rank_memories,
)

pytestmark = pytest.mark.unit


def hit(
    memory_id: str,
    chunk_index: int,
    score: float,
    *,
    title: str = "",
    tags: tuple[str, ...] = (),
    chunk_id: str | None = None,
    content: str = "",
) -> VectorSearchHit:
    return VectorSearchHit(
        user_id="user_123",
        memory_id=memory_id,
        chunk_id=chunk_id or f"{memory_id}_chunk_{chunk_index}",
        chunk_index=chunk_index,
        content=content or f"{memory_id} chunk {chunk_index}",
        score=score,
        title=title or f"Title of {memory_id}",
        tags=tags,
    )


# --------------------------------------------------------------------------
# The worked example from doc 03, Phase 12
# --------------------------------------------------------------------------
DOCUMENTED_INPUT = [
    hit("memory_a", 0, 0.94),
    hit("memory_a", 1, 0.91),
    hit("memory_b", 0, 0.89),
]


def test_the_documented_example_produces_the_documented_output() -> None:
    ranked = rank_memories(DOCUMENTED_INPUT)

    assert [(memory.memory_id, memory.score) for memory in ranked] == [
        ("memory_a", 0.94),
        ("memory_b", 0.89),
    ]


def test_each_memory_appears_exactly_once() -> None:
    """Doc 01, decision 17: deduplicate by memoryId."""
    ranked = rank_memories(DOCUMENTED_INPUT)

    assert len(ranked) == 2
    assert len({memory.memory_id for memory in ranked}) == 2


def test_both_chunks_of_a_memory_are_kept() -> None:
    ranked = rank_memories(DOCUMENTED_INPUT)

    assert len(ranked[0].chunks) == 2
    assert len(ranked[1].chunks) == 1


# --------------------------------------------------------------------------
# Memory score is the maximum
# --------------------------------------------------------------------------
def test_memory_score_is_the_maximum_chunk_score() -> None:
    """Doc 01, decision 18."""
    ranked = rank_memories([hit("m", 0, 0.40), hit("m", 1, 0.95), hit("m", 2, 0.60)])

    assert ranked[0].score == 0.95


def test_memory_score_is_not_the_sum() -> None:
    """Summing would let many weak chunks outrank a single strong match."""
    ranked = rank_memories(
        [
            hit("weak", 0, 0.40),
            hit("weak", 1, 0.40),
            hit("weak", 2, 0.40),
            hit("strong", 0, 0.90),
        ]
    )

    assert [memory.memory_id for memory in ranked] == ["strong", "weak"]
    assert ranked[1].score == 0.40


def test_memory_score_is_not_the_mean() -> None:
    """Averaging would penalise a long Memory for its non-matching chunks."""
    ranked = rank_memories([hit("m", 0, 0.99), hit("m", 1, 0.01)])

    assert ranked[0].score == 0.99


def test_memory_score_is_not_the_first_chunk_seen() -> None:
    ranked = rank_memories([hit("m", 0, 0.10), hit("m", 1, 0.90)])

    assert ranked[0].score == 0.90


# --------------------------------------------------------------------------
# Ordering
# --------------------------------------------------------------------------
def test_memories_are_ordered_by_descending_score() -> None:
    ranked = rank_memories([hit("low", 0, 0.76), hit("high", 0, 0.99), hit("mid", 0, 0.85)])

    assert [memory.memory_id for memory in ranked] == ["high", "mid", "low"]


def test_input_order_does_not_affect_the_result() -> None:
    """The vector engine's return order must not leak into the ranking."""
    hits = [hit(f"memory_{n}", 0, n / 100) for n in range(20)]
    expected = [memory.memory_id for memory in rank_memories(hits)]

    for seed in range(10):
        shuffled = hits[:]
        random.Random(seed).shuffle(shuffled)  # noqa: S311 - seeded test input, not crypto
        assert [memory.memory_id for memory in rank_memories(shuffled)] == expected


def test_equal_memory_scores_break_on_memory_id() -> None:
    forward = rank_memories([hit("memory_b", 0, 0.80), hit("memory_a", 0, 0.80)])
    reverse = rank_memories([hit("memory_a", 0, 0.80), hit("memory_b", 0, 0.80)])

    assert [m.memory_id for m in forward] == ["memory_a", "memory_b"]
    assert [m.memory_id for m in reverse] == ["memory_a", "memory_b"]


def test_chunks_are_ordered_by_chunk_index() -> None:
    """The public chunk shape has no score field, so a score ordering would be
    invisible to the Backend; chunkIndex is right there in the payload."""
    ranked = rank_memories([hit("m", 2, 0.99), hit("m", 0, 0.80), hit("m", 1, 0.90)])

    assert [chunk.chunk_index for chunk in ranked[0].chunks] == [0, 1, 2]


def test_chunk_order_is_reading_order_not_relevance_order() -> None:
    ranked = rank_memories([hit("m", 0, 0.10), hit("m", 1, 0.99)])

    assert [chunk.chunk_index for chunk in ranked[0].chunks] == [0, 1]
    assert ranked[0].score == 0.99


def test_equal_chunk_indexes_break_on_chunk_id() -> None:
    ranked = rank_memories(
        [
            hit("m", 0, 0.9, chunk_id="chunk_z"),
            hit("m", 0, 0.9, chunk_id="chunk_a"),
        ]
    )

    assert [chunk.chunk_id for chunk in ranked[0].chunks] == ["chunk_a", "chunk_z"]


# --------------------------------------------------------------------------
# Title and tags
# --------------------------------------------------------------------------
def test_title_and_tags_come_from_the_memory() -> None:
    ranked = rank_memories([hit("m", 0, 0.9, title="MongoDB", tags=("db", "ai"))])

    assert ranked[0].title == "MongoDB"
    assert ranked[0].tags == ("db", "ai")


def test_title_is_taken_from_the_best_matching_chunk() -> None:
    """A stale index must not make the result depend on engine ordering."""
    ranked = rank_memories(
        [
            hit("m", 0, 0.10, title="stale title"),
            hit("m", 1, 0.99, title="current title"),
        ]
    )

    assert ranked[0].title == "current title"


def test_representative_chunk_is_stable_when_scores_tie() -> None:
    forward = rank_memories([hit("m", 1, 0.9, title="second"), hit("m", 0, 0.9, title="first")])
    reverse = rank_memories([hit("m", 0, 0.9, title="first"), hit("m", 1, 0.9, title="second")])

    assert forward[0].title == reverse[0].title == "first"


# --------------------------------------------------------------------------
# Edges
# --------------------------------------------------------------------------
def test_no_hits_produce_no_memories() -> None:
    """The no-match response is built from this."""
    assert rank_memories([]) == []


def test_a_single_hit_produces_one_memory() -> None:
    ranked = rank_memories([hit("m", 0, 0.9)])

    assert len(ranked) == 1
    assert ranked[0].memory_id == "m"
    assert len(ranked[0].chunks) == 1


def test_no_chunk_is_dropped() -> None:
    hits = [hit(f"memory_{n % 3}", n, 0.8 + n / 100) for n in range(12)]

    ranked = rank_memories(hits)

    assert sum(len(memory.chunks) for memory in ranked) == len(hits)


def test_the_input_is_not_mutated() -> None:
    hits = [hit("m", 1, 0.8), hit("m", 0, 0.9)]

    rank_memories(hits)

    assert [h.chunk_index for h in hits] == [1, 0]


def test_a_tuple_of_hits_is_accepted() -> None:
    assert len(rank_memories((hit("m", 0, 0.9),))) == 1


def test_no_threshold_is_applied_here() -> None:
    """Score filtering is Phase 11; ranking takes what it is given."""
    ranked = rank_memories([hit("m", 0, 0.01)])

    assert [memory.score for memory in ranked] == [0.01]


def test_ranked_memories_are_immutable() -> None:
    ranked = rank_memories([hit("m", 0, 0.9)])

    with pytest.raises(AttributeError):
        ranked[0].score = 1.0  # type: ignore[misc]


def test_a_ranked_memory_cannot_be_built_without_chunks() -> None:
    with pytest.raises(ValueError, match="at least one chunk"):
        RankedMemory(memory_id="m", score=0.9, title="t", tags=(), chunks=())


# --------------------------------------------------------------------------
# Composed with threshold filtering (Phase 11)
# --------------------------------------------------------------------------
def test_filtering_then_ranking_matches_the_documented_flow() -> None:
    """Top-K -> threshold -> deduplicate -> rank (doc 01, Flow B)."""
    hits = [
        hit("memory_a", 0, 0.94),
        hit("memory_a", 1, 0.60),
        hit("memory_b", 0, 0.89),
        hit("memory_c", 0, 0.40),
    ]

    ranked = rank_memories(filter_by_threshold(hits))

    assert [(m.memory_id, m.score) for m in ranked] == [("memory_a", 0.94), ("memory_b", 0.89)]
    assert len(ranked[0].chunks) == 1


def test_a_memory_survives_on_its_best_chunk_alone() -> None:
    """One strong chunk keeps the Memory even when its others fall below."""
    hits = [hit("m", 0, 0.80), hit("m", 1, 0.40), hit("m", 2, 0.30)]

    ranked = rank_memories(filter_by_threshold(hits))

    assert len(ranked) == 1
    assert ranked[0].score == 0.80
    assert [chunk.chunk_index for chunk in ranked[0].chunks] == [0]


def test_everything_below_the_threshold_yields_no_memories() -> None:
    hits = [hit("memory_a", 0, 0.74), hit("memory_b", 0, 0.10)]

    assert rank_memories(filter_by_threshold(hits)) == []
