"""Retrieval quality evaluation (doc 03, Phase 20).

Tooling rather than request-path code: it measures the pipeline, it is not
part of it. Kept under ``src`` so it is importable, type-checked and testable
like everything else, rather than living as an untested script.

Evaluation informs a later product decision about the locked 0.75 threshold.
It never changes it (doc 03, Phase 20).
"""

from memovo_ai.evaluation.harness import (
    Dataset,
    EvaluationCase,
    evaluate_thresholds,
    index_memories,
    load_dataset,
)
from memovo_ai.evaluation.metrics import (
    HEADER,
    QueryOutcome,
    RetrievalMetrics,
    score_query,
    summarize,
)

__all__ = [
    "HEADER",
    "Dataset",
    "EvaluationCase",
    "QueryOutcome",
    "RetrievalMetrics",
    "evaluate_thresholds",
    "index_memories",
    "load_dataset",
    "score_query",
    "summarize",
]
