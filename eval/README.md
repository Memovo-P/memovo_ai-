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

## Generation evaluation

Two synthetic datasets hold retrieval constant so that generation is judged on
its own:

- `datasets/generation_chat.json` — Memory Chat cases with pinned evidence:
  grounded (English, Arabic, mixed script, multi-memory), insufficient
  evidence, conflicting evidence, short follow-ups with history, ambiguous
  references, injection in a memory and in history, and no-match.
- `datasets/note_preparation.json` — raw content with the tokens a faithful
  note must keep and the additions it must not make.

```bash
MEMOVO_GENERATION_ENABLED=true MEMOVO_OPENROUTER_API_KEY=... \
  uv run python scripts/evaluate_generation.py
```

Opt-in only: it calls the approved OpenRouter model with synthetic content.
It prints structural metrics (`found` correctness, source membership,
required/forbidden mentions, abstention markers, preservation, additions,
invalid-output counts) and writes every answer to `eval/results/` for human
review against each case's `rubric`. The lexical checks cannot prove a claim
is supported; the review can. Targets are set with the owner from those
reviews.

The harness arithmetic is verified in CI (`tests/evaluation/test_generation_quality.py`)
with scripted fake models. **No real-model run has been recorded yet**: as of
2026-09-13 the local key is rejected by OpenRouter and the approved model id
is absent from its catalog (see `docs/deployment.md`, section 8).

## Dataset

`datasets/sprint1_retrieval.json` — 30 memories, 65 queries (53 answerable,
12 no-match). Memories use the contract v1.9 `/ai/memories/process` Note
shape, so they feed the pipeline unchanged. Query kinds: direct, paraphrase,
multi-target, arabic, mixed-script, crosslingual, vague, related-not-relevant,
no-match.

The dataset was migrated to v1.9 on 2026-09-13: `description` became
`content`, `type: "note"` was added, and the former `whySaved` field was
dropped without mapping (contract section 8.5). The results table below was
produced **before** that migration, with `whySaved` still part of the
embedded text; it has not been re-run since and should be read as historical
until it is.

`expectedMemoryIds: []` means the correct answer is no match.

## Results — 2026-09-10, Qwen3-Embedding-0.6B, query prompt on, Top-K 5

> **Scored with raw cosine similarity, via the in-memory `fake` provider.**
> These numbers do **not** describe MongoDB Atlas. Atlas returns its own
> `vectorSearchScore`, which its documentation describes as normalized for
> cosine similarity, so the same threshold value does not select the same
> chunks on both. Every threshold in this table means "raw cosine >= x" and
> nothing more. See "Score semantics" below.

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

- **At raw cosine 0.75, this corpus loses 43 of its 53 answerable queries.**
  Recall 0.189. Not a marginal mis-calibration -- but read it as a statement
  about raw cosine on this dataset, not as a measurement of Atlas. Nothing
  here was run against a vector database.
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

**Raw cosine 0.40**, with 0.35 as the recall-favouring alternative. Not
applied -- replacing the locked 0.75 is a product decision (doc 03, Phase 20).

**Do not copy 0.40 into a production Atlas deployment.** It is a raw-cosine
figure measured through the fake provider. `MEMOVO_SEARCH_SIMILARITY_THRESHOLD`
is compared against whatever score the *active* provider returns, and the
Atlas adapter forwards Atlas's score unchanged. Transferring the number
requires establishing the mapping between the two scales first -- see below.

### Score semantics -- open, and blocking any threshold decision

| | Score the provider returns |
|---|---|
| `fake` (this evaluation) | raw cosine similarity, computed in-process |
| `atlas` | Atlas's `vectorSearchScore`, forwarded verbatim by the adapter |

MongoDB documents `vectorSearchScore` as normalized for `cosine` and
`dotProduct` similarity, which would place it on a different scale from raw
cosine. **This has not been verified against a live cluster** -- no Atlas
integration run has happened -- so the exact relationship is unconfirmed
here and stated as documentation, not measurement.

Consequences, none of them resolved:

- a single `MEMOVO_SEARCH_SIMILARITY_THRESHOLD` value does not mean the same
  thing under both providers
- this evaluation cannot be used to pick a production threshold until the
  mapping is established
- `0.75` under Atlas is not the `0.75` measured above

**Owner:** AI, with Product for the final value. **Required evidence:** a
threshold sweep run through the Atlas adapter against a seeded index, so the
numbers come from the engine that will serve production. Until then neither
`0.75` nor `0.40` is a defensible production setting, and nothing in the code
has been changed.

### Caveats

30 memories and 65 queries is a small corpus, and relevance labels for the
`vague` and `related-not-relevant` kinds are judgement calls. The *direction*
is unambiguous; the second decimal place is not.

Two things must happen before any number here is treated as settled: re-run
against real user memories, and re-run through the Atlas adapter so the
scores come from the production engine rather than from raw cosine.
