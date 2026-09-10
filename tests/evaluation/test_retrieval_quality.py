"""The evaluation harness itself.

These run in CI with a deterministic bag-of-words embedder, so they verify the
*measurement* is correct. The real numbers come from
``scripts/evaluate_retrieval.py``, which needs model weights.

A wrong metric would quietly justify the wrong threshold, so the arithmetic is
pinned against hand-computed values.
"""

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from memovo_ai.embeddings import EMBEDDING_DIMENSION, Embedding
from memovo_ai.evaluation import (
    QueryOutcome,
    evaluate_thresholds,
    load_dataset,
    score_query,
    summarize,
)

pytestmark = pytest.mark.evaluation

DATASET_PATH = Path(__file__).resolve().parents[2] / "eval" / "datasets" / "sprint1_retrieval.json"


def outcome(
    query_id: str = "q",
    *,
    kind: str = "direct",
    expected: Sequence[str] = ("a",),
    returned: Sequence[str] = ("a",),
) -> QueryOutcome:
    return QueryOutcome(
        query_id=query_id,
        kind=kind,
        expected=frozenset(expected),
        returned=tuple(returned),
    )


# --------------------------------------------------------------------------
# Per-query arithmetic
# --------------------------------------------------------------------------
def test_a_perfect_single_hit_scores_one_everywhere() -> None:
    assert score_query(outcome()) == {
        "recall": 1.0,
        "precision": 1.0,
        "reciprocal_rank": 1.0,
        "top1": 1.0,
    }


def test_recall_is_the_share_of_expected_memories_found() -> None:
    scored = score_query(outcome(expected=["a", "b", "c", "d"], returned=["a", "b"]))

    assert scored["recall"] == 0.5


def test_precision_is_the_share_of_returned_memories_that_were_wanted() -> None:
    scored = score_query(outcome(expected=["a"], returned=["x", "a", "y", "z"]))

    assert scored["precision"] == 0.25


def test_reciprocal_rank_uses_the_first_correct_position() -> None:
    assert score_query(outcome(expected=["a"], returned=["x", "y", "a"]))["reciprocal_rank"] == (
        pytest.approx(1 / 3)
    )


def test_reciprocal_rank_is_zero_when_nothing_correct_returns() -> None:
    assert score_query(outcome(expected=["a"], returned=["x", "y"]))["reciprocal_rank"] == 0.0


def test_top1_only_counts_the_first_result() -> None:
    assert score_query(outcome(expected=["a"], returned=["x", "a"]))["top1"] == 0.0
    assert score_query(outcome(expected=["a"], returned=["a", "x"]))["top1"] == 1.0


def test_returning_nothing_scores_zero_rather_than_dividing_by_zero() -> None:
    scored = score_query(outcome(expected=["a"], returned=[]))

    assert scored == {"recall": 0.0, "precision": 0.0, "reciprocal_rank": 0.0, "top1": 0.0}


def test_a_no_match_case_has_no_recall_or_precision() -> None:
    """Averaging it in would hide the recall/no-match trade-off."""
    assert score_query(outcome(expected=[], returned=[])) == {}


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------
def test_summary_separates_answerable_from_no_match() -> None:
    metrics = summarize(
        0.75,
        [
            outcome("q1"),
            outcome("q2", expected=["b"], returned=["x"]),
            outcome("q3", expected=[], returned=[]),
        ],
    )

    assert metrics.answerable == 2
    assert metrics.no_match == 1
    assert metrics.recall_at_k == 0.5
    assert metrics.top1_accuracy == 0.5


def test_no_match_accuracy_counts_correct_silence() -> None:
    metrics = summarize(
        0.75,
        [
            outcome("n1", expected=[], returned=[]),
            outcome("n2", expected=[], returned=[]),
            outcome("n3", expected=[], returned=["x"]),
            outcome("q1"),
        ],
    )

    assert metrics.no_match_accuracy == pytest.approx(2 / 3)
    assert metrics.false_positive_rate == pytest.approx(1 / 3)


def test_no_match_accuracy_and_false_positive_rate_are_complements() -> None:
    metrics = summarize(
        0.75,
        [outcome("n1", expected=[], returned=["x"]), outcome("n2", expected=[], returned=[])],
    )

    assert metrics.no_match_accuracy + metrics.false_positive_rate == pytest.approx(1.0)


def test_missed_entirely_counts_answerable_queries_returning_nothing() -> None:
    metrics = summarize(
        0.75,
        [
            outcome("q1", expected=["a"], returned=[]),
            outcome("q2", expected=["b"], returned=["b"]),
            outcome("n1", expected=[], returned=[]),
        ],
    )

    assert metrics.missed_entirely == 1


def test_recall_is_reported_per_query_kind() -> None:
    metrics = summarize(
        0.75,
        [
            outcome("q1", kind="direct", expected=["a"], returned=["a"]),
            outcome("q2", kind="vague", expected=["b"], returned=[]),
            outcome("q3", kind="vague", expected=["c"], returned=["c"]),
        ],
    )

    assert metrics.per_kind_recall == {"direct": 1.0, "vague": 0.5}


def test_a_dataset_of_only_no_match_queries_is_handled() -> None:
    metrics = summarize(0.75, [outcome("n1", expected=[], returned=[])])

    assert metrics.answerable == 0
    assert metrics.recall_at_k == 0.0
    assert metrics.no_match_accuracy == 1.0


# --------------------------------------------------------------------------
# The shipped dataset
# --------------------------------------------------------------------------
def test_the_dataset_loads_and_validates_against_the_real_schema() -> None:
    dataset = load_dataset(DATASET_PATH)

    assert len(dataset.memories) >= 30
    assert len(dataset.cases) >= 60


def test_every_expected_memory_id_exists() -> None:
    """A typo in an id would silently depress recall forever."""
    dataset = load_dataset(DATASET_PATH)
    known = {memory.memory_id for memory in dataset.memories}

    for case in dataset.cases:
        assert case.expected <= known, f"{case.query_id} references unknown ids"


def test_memory_ids_are_unique() -> None:
    dataset = load_dataset(DATASET_PATH)
    ids = [memory.memory_id for memory in dataset.memories]

    assert len(set(ids)) == len(ids)


def test_query_ids_are_unique() -> None:
    dataset = load_dataset(DATASET_PATH)
    ids = [case.query_id for case in dataset.cases]

    assert len(set(ids)) == len(ids)


def test_the_dataset_covers_every_category_doc_05_asks_for() -> None:
    dataset = load_dataset(DATASET_PATH)
    kinds = {case.kind for case in dataset.cases}

    for required in ("direct", "paraphrase", "multi", "arabic", "mixed", "vague", "no-match"):
        assert required in kinds, f"dataset has no {required} queries"


def test_the_dataset_has_a_meaningful_no_match_set() -> None:
    """False-positive rate cannot be estimated from a handful of cases."""
    dataset = load_dataset(DATASET_PATH)

    assert sum(1 for case in dataset.cases if not case.expected) >= 10


def test_the_dataset_contains_arabic_memories() -> None:
    dataset = load_dataset(DATASET_PATH)
    arabic = [m for m in dataset.memories if any("؀" <= c <= "ۿ" for c in m.title)]

    assert len(arabic) >= 3


def test_multi_target_queries_exist() -> None:
    """Recall@5 is meaningless if every query has exactly one answer."""
    dataset = load_dataset(DATASET_PATH)

    assert sum(1 for case in dataset.cases if len(case.expected) > 1) >= 8


def test_the_dataset_file_is_valid_json_with_the_documented_shape() -> None:
    raw = json.loads(DATASET_PATH.read_text(encoding="utf-8"))

    assert {"name", "memories", "queries"} <= set(raw)
    assert all({"id", "kind", "query", "expectedMemoryIds"} <= set(q) for q in raw["queries"])


# --------------------------------------------------------------------------
# End to end through the real pipeline, with a deterministic embedder
# --------------------------------------------------------------------------
def _dimension(token: str) -> int:
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % EMBEDDING_DIMENSION


def _embed(text: str) -> Embedding:
    vector = [0.0] * EMBEDDING_DIMENSION
    for token in text.lower().split():
        cleaned = token.strip(".,:!?()[]\"'")
        if cleaned:
            vector[_dimension(cleaned)] += 1.0

    norm = sum(value * value for value in vector) ** 0.5
    return vector if norm == 0.0 else [value / norm for value in vector]


class BagOfWordsEmbeddings:
    """Deterministic stand-in: shares no weights, but discriminates by wording."""

    async def embed_documents(self, texts: Sequence[str]) -> list[Embedding]:
        return [_embed(text) for text in texts]

    async def embed_query(self, query: str) -> Embedding:
        return _embed(query)


async def test_the_harness_runs_the_whole_pipeline() -> None:
    dataset = load_dataset(DATASET_PATH)

    results, outcomes = await evaluate_thresholds(
        dataset, embeddings=BagOfWordsEmbeddings(), thresholds=[0.0, 0.5]
    )

    assert len(results) == 2
    assert set(outcomes) == {0.0, 0.5}
    assert all(len(per_query) == len(dataset.cases) for per_query in outcomes.values())


async def test_raising_the_threshold_never_increases_recall() -> None:
    """The core trade-off the sweep exists to expose."""
    dataset = load_dataset(DATASET_PATH)

    results, _ = await evaluate_thresholds(
        dataset, embeddings=BagOfWordsEmbeddings(), thresholds=[0.0, 0.3, 0.6, 0.9]
    )
    recalls = [m.recall_at_k for m in results]

    assert recalls == sorted(recalls, reverse=True)


async def test_raising_the_threshold_never_decreases_no_match_accuracy() -> None:
    dataset = load_dataset(DATASET_PATH)

    results, _ = await evaluate_thresholds(
        dataset, embeddings=BagOfWordsEmbeddings(), thresholds=[0.0, 0.3, 0.6, 0.9]
    )
    accuracies = [m.no_match_accuracy for m in results]

    assert accuracies == sorted(accuracies)


async def test_an_impossible_threshold_returns_nothing_at_all() -> None:
    dataset = load_dataset(DATASET_PATH)

    results, _ = await evaluate_thresholds(
        dataset, embeddings=BagOfWordsEmbeddings(), thresholds=[1.01]
    )

    assert results[0].recall_at_k == 0.0
    assert results[0].no_match_accuracy == 1.0
    assert results[0].missed_entirely == results[0].answerable


async def test_results_never_exceed_top_k_memories() -> None:
    dataset = load_dataset(DATASET_PATH)

    _, outcomes = await evaluate_thresholds(
        dataset, embeddings=BagOfWordsEmbeddings(), thresholds=[0.0], top_k=5
    )

    assert all(len(o.returned) <= 5 for o in outcomes[0.0])


async def test_evaluation_is_deterministic() -> None:
    dataset = load_dataset(DATASET_PATH)

    first, _ = await evaluate_thresholds(
        dataset, embeddings=BagOfWordsEmbeddings(), thresholds=[0.3]
    )
    second, _ = await evaluate_thresholds(
        dataset, embeddings=BagOfWordsEmbeddings(), thresholds=[0.3]
    )

    assert first[0].recall_at_k == second[0].recall_at_k
    assert first[0].mrr == second[0].mrr


async def test_the_locked_threshold_is_not_mutated_by_evaluation() -> None:
    """Sweeping must not leave the service configured differently."""
    from memovo_ai.core.config import SearchSettings

    dataset = load_dataset(DATASET_PATH)
    await evaluate_thresholds(dataset, embeddings=BagOfWordsEmbeddings(), thresholds=[0.1, 0.9])

    assert SearchSettings(_env_file=None).similarity_threshold == 0.75  # type: ignore[call-arg]
