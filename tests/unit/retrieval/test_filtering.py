"""Similarity threshold filtering."""

import math

import pytest

from memovo_ai.retrieval import (
    DEFAULT_SIMILARITY_THRESHOLD,
    VectorSearchHit,
    filter_by_threshold,
    meets_threshold,
)

pytestmark = pytest.mark.unit


def hit(score: float, *, chunk_id: str = "chunk_001") -> VectorSearchHit:
    return VectorSearchHit(
        user_id="user_123",
        memory_id="memory_456",
        chunk_id=chunk_id,
        chunk_index=0,
        content="MongoDB Vector Search...",
        score=score,
        title="MongoDB Vector Search",
    )


# --------------------------------------------------------------------------
# The locked threshold
# --------------------------------------------------------------------------
def test_the_default_threshold_is_the_locked_value() -> None:
    """Doc 01, decision 9."""
    assert DEFAULT_SIMILARITY_THRESHOLD == 0.75


def test_the_locked_threshold_is_exactly_representable() -> None:
    """0.75 is 3/4, so the boundary comparison is exact, not approximate."""
    assert DEFAULT_SIMILARITY_THRESHOLD == 3 / 4
    assert float("0.75") == DEFAULT_SIMILARITY_THRESHOLD


# --------------------------------------------------------------------------
# Boundary behaviour
# --------------------------------------------------------------------------
def test_a_score_exactly_at_the_threshold_is_kept() -> None:
    """Every source document writes the rule as `score >= 0.75`."""
    assert filter_by_threshold([hit(0.75)]) == [hit(0.75)]


@pytest.mark.parametrize("score", [0.7499999999, 0.74, 0.5, 0.0, -0.3])
def test_scores_below_the_threshold_are_dropped(score: float) -> None:
    assert filter_by_threshold([hit(score)]) == []


@pytest.mark.parametrize("score", [0.75, 0.7500001, 0.9, 1.0])
def test_scores_at_or_above_the_threshold_are_kept(score: float) -> None:
    assert len(filter_by_threshold([hit(score)])) == 1


def test_the_boundary_is_inclusive_not_exclusive() -> None:
    """Stated explicitly: an off-by-one here silently drops valid matches."""
    kept = filter_by_threshold([hit(0.75, chunk_id="at"), hit(0.7499, chunk_id="below")])

    assert [h.chunk_id for h in kept] == ["at"]


def test_meets_threshold_agrees_with_the_filter() -> None:
    assert meets_threshold(hit(0.75)) is True
    assert meets_threshold(hit(0.7499)) is False


# --------------------------------------------------------------------------
# Behaviour over a result set
# --------------------------------------------------------------------------
def test_only_relevant_hits_survive() -> None:
    hits = [hit(0.94, chunk_id="a"), hit(0.60, chunk_id="b"), hit(0.80, chunk_id="c")]

    assert [h.chunk_id for h in filter_by_threshold(hits)] == ["a", "c"]


def test_order_is_preserved() -> None:
    """The provider already ranked these; filtering must not reshuffle them."""
    hits = [hit(0.99, chunk_id="a"), hit(0.90, chunk_id="b"), hit(0.80, chunk_id="c")]

    assert [h.chunk_id for h in filter_by_threshold(hits)] == ["a", "b", "c"]


def test_nothing_relevant_yields_an_empty_result() -> None:
    """The no-match response is built from this (doc 01, decision 20)."""
    assert filter_by_threshold([hit(0.1), hit(0.2), hit(0.74)]) == []


def test_an_empty_input_yields_an_empty_result() -> None:
    assert filter_by_threshold([]) == []


def test_filtering_never_truncates() -> None:
    """Top-K was already applied by the provider; dropping more would change
    the count the ranking phase sees."""
    hits = [hit(0.9, chunk_id=f"chunk_{n}") for n in range(20)]

    assert len(filter_by_threshold(hits)) == 20


def test_the_input_is_not_mutated() -> None:
    hits = [hit(0.9, chunk_id="a"), hit(0.1, chunk_id="b")]

    filter_by_threshold(hits)

    assert [h.chunk_id for h in hits] == ["a", "b"]


def test_a_new_list_is_returned() -> None:
    hits = [hit(0.9)]

    assert filter_by_threshold(hits) is not hits


def test_a_tuple_of_hits_is_accepted() -> None:
    assert len(filter_by_threshold((hit(0.9), hit(0.1)))) == 1


# --------------------------------------------------------------------------
# Custom thresholds
# --------------------------------------------------------------------------
@pytest.mark.parametrize("threshold", [0.60, 0.65, 0.70, 0.75, 0.80, 0.85])
def test_the_evaluation_sweep_thresholds_are_all_usable(threshold: float) -> None:
    """Doc 05, section 13 sweeps exactly these values."""
    hits = [hit(score / 100, chunk_id=f"chunk_{score}") for score in range(50, 100, 5)]

    kept = filter_by_threshold(hits, threshold=threshold)

    assert all(h.score >= threshold for h in kept)
    assert len(kept) == sum(1 for h in hits if h.score >= threshold)


def test_a_higher_threshold_keeps_fewer_hits() -> None:
    hits = [hit(0.70, chunk_id="a"), hit(0.80, chunk_id="b"), hit(0.90, chunk_id="c")]

    assert len(filter_by_threshold(hits, threshold=0.60)) == 3
    assert len(filter_by_threshold(hits, threshold=0.85)) == 1


@pytest.mark.parametrize("threshold", [-1.0, 0.0, 1.0, 2.5])
def test_thresholds_outside_zero_to_one_are_accepted(threshold: float) -> None:
    """The metric is not locked (doc 01, section 7); inner product is unbounded."""
    assert filter_by_threshold([hit(3.0)], threshold=threshold) == [hit(3.0)]


@pytest.mark.parametrize("threshold", [math.nan, math.inf, -math.inf])
def test_a_non_finite_threshold_is_rejected(threshold: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        filter_by_threshold([hit(0.9)], threshold=threshold)


def test_nan_scores_cannot_reach_the_filter() -> None:
    """VectorSearchHit already refuses them, so no NaN comparison happens here."""
    with pytest.raises(ValueError, match="finite"):
        hit(math.nan)
