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

## Verification gate

Every phase must pass all five commands before it is considered complete:

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src
```

`uv sync` must succeed first.

## Repository layout

The target layout is defined in `memovo-ai-docs/04_REPOSITORY_STRUCTURE.md`.
Directories are created phase by phase as the code that belongs in them is
written, rather than all at once as empty scaffolding.

## Build status

| Phase | Status |
|---|---|
| 00 - Repository foundation | Complete |
| 01 - API contract schemas | Not started |
| 02+ | Not started |
