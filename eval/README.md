# Retrieval evaluation

Measures retrieval quality through the **real pipeline** — canonical content,
chunking, embedding, user pre-filter, Top-K, threshold, dedupe, max-chunk
ranking. Scoring whole memories instead would measure a system nobody ships.

## Running it

```bash
uv sync --extra embeddings
uv run python scripts/evaluate_retrieval.py
```

Loads Qwen3 weights, so it is a manual tool rather than part of the suite.
Reports land in `eval/results/` (gitignored — they are artifacts).

```bash
# finer steps around a candidate
uv run python scripts/evaluate_retrieval.py --thresholds 0.35 0.40 0.45
```

The harness itself is tested in CI (`tests/evaluation/`) with a deterministic
bag-of-words embedder, so the metric arithmetic is verified without weights.

## Dataset

`datasets/sprint1_retrieval.json` — 30 memories, 65 queries (53 answerable,
12 no-match). Memories use the `/ai/memories/process` request shape, so they
feed the pipeline unchanged. Query kinds: direct, paraphrase, multi-target,
arabic, mixed-script, crosslingual, vague, related-not-relevant, no-match.

`expectedMemoryIds: []` means the correct answer is no match.

## Results — 2026-09-10, Qwen3-Embedding-0.6B, query prompt on, Top-K 5

| threshold | recall@5 | precision@5 | MRR | top-1 | no-match | fp rate | missed |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.30 | 0.898 | 0.585 | 0.934 | 0.906 | 0.917 | 0.083 | 2 |
| **0.35** | **0.800** | 0.636 | 0.858 | 0.849 | 1.000 | 0.000 | 5 |
| **0.40** | **0.770** | **0.680** | 0.802 | 0.792 | 1.000 | 0.000 | 10 |
| **0.45** | 0.720 | **0.691** | 0.745 | 0.736 | 1.000 | 0.000 | 13 |
| 0.50 | 0.648 | 0.612 | 0.660 | 0.660 | 1.000 | 0.000 | 17 |
| 0.55 | 0.566 | 0.568 | 0.585 | 0.585 | 1.000 | 0.000 | 22 |
| 0.60 | 0.434 | 0.462 | 0.472 | 0.472 | 1.000 | 0.000 | 28 |
| 0.65 | 0.346 | 0.349 | 0.358 | 0.358 | 1.000 | 0.000 | 34 |
| 0.70 | 0.270 | 0.283 | 0.283 | 0.283 | 1.000 | 0.000 | 38 |
| *0.75 (locked)* | *0.189* | *0.189* | *0.189* | *0.189* | *1.000* | *0.000* | *43* |
| 0.80 | 0.094 | 0.094 | 0.094 | 0.094 | 1.000 | 0.000 | 48 |
| 0.85 | 0.019 | 0.019 | 0.019 | 0.019 | 1.000 | 0.000 | 52 |

### Reading

- **0.75 loses 43 of 53 answerable queries.** Recall 0.189. Not a marginal
  mis-calibration.
- **No false positives anywhere at 0.35 and above.** The risk a high threshold
  is meant to manage does not materialise until 0.30, where fp rate reaches
  0.083. The safe ceiling for zero false positives is 0.35.
- **Precision peaks at 0.45** (0.691); recall is still 0.720 there.
- **F1 peaks at 0.40** (0.722, versus 0.709 at 0.35 and 0.705 at 0.45).

### By query kind

`direct` holds recall 1.000 down to 0.60 and only collapses above it.
`vague` is effectively unretrievable above 0.35 — no threshold serves it
without also admitting noise. `multi` (multi-target) degrades fastest, which
is expected: Top-K 5 chunks can dedupe to fewer than 5 memories, so a query
expecting 4 answers cannot always get them.

### Recommendation

**0.40**, with 0.35 as the recall-favouring alternative. Not applied —
replacing the locked 0.75 is a product decision (doc 03, Phase 20).

### Caveats

30 memories and 65 queries is a small corpus, and relevance labels for the
`vague` and `related-not-relevant` kinds are judgement calls. The *direction*
is unambiguous; the second decimal place is not. Re-run against real user
memories before treating 0.40 as settled.
