"""Retrieval quality metrics (doc 05, section 12).

Every metric is computed from the Memory ids a query returned, in rank order,
against the ids it should have returned. Nothing here knows about embeddings
or vectors -- which is what makes it testable without a model.
"""

import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field

__all__ = ["QueryOutcome", "RetrievalMetrics", "score_query", "summarize"]


@dataclass(frozen=True, slots=True)
class QueryOutcome:
    """What one query asked for and what it got back."""

    query_id: str
    kind: str
    expected: frozenset[str]
    returned: tuple[str, ...]
    #: Score of the best returned Memory, or None when nothing was returned.
    top_score: float | None = None

    @property
    def is_no_match_case(self) -> bool:
        """The correct answer is "I couldn't find a relevant memory"."""
        return not self.expected

    @property
    def found(self) -> bool:
        return bool(self.returned)

    @property
    def hits(self) -> list[str]:
        return [memory_id for memory_id in self.returned if memory_id in self.expected]


def score_query(outcome: QueryOutcome) -> dict[str, float]:
    """Per-query metrics.

    ``recall`` is the share of expected Memories that came back. ``precision``
    is the share of returned Memories that were expected. ``reciprocal_rank``
    is 1/rank of the first correct Memory, or 0 if none appeared.

    No-match cases have no recall, precision or rank to speak of; they are
    scored separately by :func:`summarize`.
    """
    if outcome.is_no_match_case:
        return {}

    hits = outcome.hits
    recall = len(hits) / len(outcome.expected)
    precision = len(hits) / len(outcome.returned) if outcome.returned else 0.0

    reciprocal_rank = 0.0
    for rank, memory_id in enumerate(outcome.returned, start=1):
        if memory_id in outcome.expected:
            reciprocal_rank = 1.0 / rank
            break

    return {
        "recall": recall,
        "precision": precision,
        "reciprocal_rank": reciprocal_rank,
        "top1": 1.0 if outcome.returned and outcome.returned[0] in outcome.expected else 0.0,
    }


@dataclass(frozen=True, slots=True)
class RetrievalMetrics:
    """Aggregate quality at one threshold."""

    threshold: float
    answerable: int
    no_match: int
    recall_at_k: float
    precision_at_k: float
    mrr: float
    top1_accuracy: float
    no_match_accuracy: float
    false_positive_rate: float
    #: Answerable queries that returned nothing at all.
    missed_entirely: int
    per_kind_recall: dict[str, float] = field(default_factory=dict)

    def as_row(self) -> str:
        return (
            f"{self.threshold:>9.2f}"
            f"{self.recall_at_k:>10.3f}"
            f"{self.precision_at_k:>12.3f}"
            f"{self.mrr:>7.3f}"
            f"{self.top1_accuracy:>8.3f}"
            f"{self.no_match_accuracy:>11.3f}"
            f"{self.false_positive_rate:>9.3f}"
            f"{self.missed_entirely:>9d}"
        )


HEADER = (
    f"{'threshold':>9}{'recall@5':>10}{'precision@5':>12}{'mrr':>7}"
    f"{'top-1':>8}{'no-match':>11}{'fp rate':>9}{'missed':>9}"
)


def summarize(threshold: float, outcomes: Sequence[QueryOutcome]) -> RetrievalMetrics:
    """Aggregate per-query outcomes into the doc 05 section 12 metrics.

    Answerable and no-match queries are kept apart deliberately. Averaging a
    no-match query's "recall" into the same number as a real query would hide
    exactly the trade-off a threshold sweep is meant to expose: raising the
    threshold improves no-match accuracy and destroys recall.
    """
    answerable = [o for o in outcomes if not o.is_no_match_case]
    no_match = [o for o in outcomes if o.is_no_match_case]

    scored = [score_query(o) for o in answerable]

    def mean(key: str) -> float:
        return statistics.fmean([s[key] for s in scored]) if scored else 0.0

    correctly_silent = sum(1 for o in no_match if not o.found)
    per_kind: dict[str, list[float]] = {}
    for outcome, score in zip(answerable, scored, strict=True):
        per_kind.setdefault(outcome.kind, []).append(score["recall"])

    return RetrievalMetrics(
        threshold=threshold,
        answerable=len(answerable),
        no_match=len(no_match),
        recall_at_k=mean("recall"),
        precision_at_k=mean("precision"),
        mrr=mean("reciprocal_rank"),
        top1_accuracy=mean("top1"),
        no_match_accuracy=correctly_silent / len(no_match) if no_match else 1.0,
        false_positive_rate=(
            (len(no_match) - correctly_silent) / len(no_match) if no_match else 0.0
        ),
        missed_entirely=sum(1 for o in answerable if not o.found),
        per_kind_recall={
            kind: statistics.fmean(values) for kind, values in sorted(per_kind.items())
        },
    )
