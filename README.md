# Memovo AI Service

AI processing and retrieval service for Memovo. **Sprint 1 is RAG-only.**

The authoritative specification lives in the sibling `memovo-ai-docs/` repository.
Those documents are the source of truth; this README does not restate them.

## Sprint 1 endpoints

| Endpoint | Purpose |
|---|---|
| `POST /ai/memories/process` | Chunk a Memory and return the complete chunk set with embeddings |
| `POST /ai/memories/search` | Embed a query and retrieve the user's matching Memories |

Nothing else is in scope. No chat, agents, tool calling, answer generation,
authentication, or database writes.

## Responsibility boundary

This service owns content preparation, chunking, embeddings, vector search,
similarity filtering, and ranking.

The Backend owns authentication, the trusted `userId`, authorization, business
logic, MongoDB persistence, Vector DB persistence, idempotency, and queues.

**This service never writes to MongoDB or the Vector DB.** Its vector
integration is read-only.

## Requirements

- Python 3.12 (pinned in `.python-version`)
- [`uv`](https://docs.astral.sh/uv/)

## Setup

```bash
uv sync
cp .env.example .env
```

This installs everything needed for development and the full test suite. The
model runtime is an optional extra, kept out of the default install so `uv sync`
and CI stay light:

```bash
uv sync --extra embeddings    # adds sentence-transformers (pulls torch)
```

Unit tests inject a fake encoder, so they never import the runtime and never
download weights. The real model is exercised only by an opt-in integration
test:

```bash
MEMOVO_RUN_EMBEDDING_INTEGRATION=1 uv run pytest tests/integration -s
```

It downloads roughly 1.2 GB on first run and reports the model's output
normalization and query-prompt availability.

## Verification gate

Every phase must pass all five commands before it is considered complete:

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src
```

`uv sync` must succeed first.

## Running the service

```bash
uv run uvicorn memovo_ai.main:app --reload
```

Both Sprint 1 endpoints are live: `POST /ai/memories/process` (Notes and
Links) and `POST /ai/memories/search`. Link page content is extracted by the
Backend and sent in the request; this service never fetches a URL. Without the `embeddings` extra installed the model
cannot load, so the service starts in an unavailable state and requests fail
with `MODEL_UNAVAILABLE` rather than silently reporting no memories.

This service is internal and must not be exposed publicly; the Backend is the
only caller.

## Repository layout

The target layout is defined in `memovo-ai-docs/04_REPOSITORY_STRUCTURE.md`.
Directories are created phase by phase as the code that belongs in them is
written, rather than all at once as empty scaffolding.

## Build status

| Phase | Status |
|---|---|
| 00 - Repository foundation | Complete |
| 01 - API contract schemas | Complete |
| 02 - Canonical content preparation | Complete |
| 03 - Hybrid chunker | Complete |
| 04 - Deterministic chunk IDs | Complete |
| 05 - Embedding provider abstraction | Complete |
| 06 - Qwen3 embedding integration | Complete |
| 07 - Memory processor service | Complete |
| 08 - `/ai/memories/process` route | Complete |
| 09 - Vector search provider | Complete |
| 10 - User isolation enforcement | Complete |
| 11 - Retrieval filtering | Complete |
| 12 - Ranking and deduplication | Complete |
| 13 - Memory search service | Complete |
| 14 - `/ai/memories/search` route | Complete |
| 15 - Error contract | Complete |
| 16 - Reprocessing contract tests | Not started |
| 17 - Link support (`about`, `source`) | Complete |
| 18+ | Not started |
