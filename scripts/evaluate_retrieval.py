"""Run the retrieval evaluation against the real Qwen3 model.

    uv sync --extra embeddings
    uv run python scripts/evaluate_retrieval.py

Loads model weights, so it is a manual tool rather than part of the test
suite. Writes a JSON report to ``eval/results/`` and prints the doc 05
section 13 threshold table.

It reports; it decides nothing. Replacing the locked 0.75 is a product
decision (doc 03, Phase 20).
"""

import argparse
import asyncio
import json
import time
from datetime import UTC, datetime
from pathlib import Path

from memovo_ai.api.dependencies import build_embedding_provider
from memovo_ai.core.config import EmbeddingSettings
from memovo_ai.evaluation import HEADER, evaluate_thresholds, load_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = REPO_ROOT / "eval" / "datasets" / "sprint1_retrieval.json"
RESULTS_DIR = REPO_ROOT / "eval" / "results"

#: The sweep doc 05, section 13 asks for.
DEFAULT_THRESHOLDS = (0.60, 0.65, 0.70, 0.75, 0.80, 0.85)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=list(DEFAULT_THRESHOLDS),
    )
    parser.add_argument(
        "--extra-thresholds",
        type=float,
        nargs="*",
        default=[],
        help="Additional thresholds to sweep, e.g. finer steps around a candidate.",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    dataset = load_dataset(args.dataset)
    thresholds = sorted({*args.thresholds, *args.extra_thresholds})

    answerable = sum(1 for case in dataset.cases if case.expected)
    print(
        f"dataset '{dataset.name}': {len(dataset.memories)} memories, "
        f"{len(dataset.cases)} queries ({answerable} answerable, "
        f"{len(dataset.cases) - answerable} no-match), top_k={args.top_k}"
    )

    settings = EmbeddingSettings()
    print(f"model: {settings.model}  query_prompt={settings.query_prompt_name!r}")

    started = time.time()
    embeddings = build_embedding_provider(settings)
    results, outcomes = await evaluate_thresholds(
        dataset, embeddings=embeddings, thresholds=thresholds, top_k=args.top_k
    )
    elapsed = time.time() - started

    print(f"\n{HEADER}")
    print("-" * len(HEADER))
    for metrics in results:
        print(metrics.as_row())

    print("\nrecall by query kind")
    kinds = sorted({k for m in results for k in m.per_kind_recall})
    print(f"{'threshold':>9}" + "".join(f"{k:>22}" for k in kinds))
    for metrics in results:
        row = f"{metrics.threshold:>9.2f}"
        for kind in kinds:
            row += f"{metrics.per_kind_recall.get(kind, 0.0):>22.3f}"
        print(row)

    # Where each answerable query actually lands, so a candidate threshold can
    # be read off the distribution rather than guessed.
    scores = sorted(
        (o.top_score, o.query_id, o.kind)
        for o in outcomes[min(thresholds)]
        if o.top_score is not None and o.expected and o.returned
    )
    print("\nlowest-scoring correct answers (the ones a high threshold loses)")
    for score, query_id, kind in scores[:12]:
        print(f"  {query_id} [{kind:<20}] {score:.4f}")

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    report = RESULTS_DIR / f"{dataset.name}_{stamp}.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        json.dumps(
            {
                "dataset": dataset.name,
                "model": settings.model,
                "query_prompt_name": settings.query_prompt_name,
                "top_k": args.top_k,
                "memories": len(dataset.memories),
                "queries": len(dataset.cases),
                "elapsed_seconds": round(elapsed, 1),
                "thresholds": [
                    {
                        "threshold": m.threshold,
                        "recall_at_k": m.recall_at_k,
                        "precision_at_k": m.precision_at_k,
                        "mrr": m.mrr,
                        "top1_accuracy": m.top1_accuracy,
                        "no_match_accuracy": m.no_match_accuracy,
                        "false_positive_rate": m.false_positive_rate,
                        "missed_entirely": m.missed_entirely,
                        "per_kind_recall": m.per_kind_recall,
                    }
                    for m in results
                ],
                "per_query": {
                    str(threshold): [
                        {
                            "id": o.query_id,
                            "kind": o.kind,
                            "expected": sorted(o.expected),
                            "returned": list(o.returned),
                            "top_score": o.top_score,
                        }
                        for o in per_query
                    ]
                    for threshold, per_query in outcomes.items()
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {report.relative_to(REPO_ROOT)}  ({elapsed:.1f}s)")


if __name__ == "__main__":
    asyncio.run(main())
