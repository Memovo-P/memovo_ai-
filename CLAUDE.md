# Memovo AI Service - Engineering Rules

The specification in the sibling `memovo-ai-docs/` repository is the **source of
truth**, and `memovo-ai-docs/CLAUDE.md` is the authoritative ruleset. Read it
before making changes. This file is a working summary for the code repository;
where the two ever disagree, the docs win - report the drift rather than
silently following this copy.

Do not redesign locked product or architecture decisions unless explicitly
instructed.

## Sprint 1 scope

Only two endpoints: `POST /ai/memories/process` and `POST /ai/memories/search`.

Out of scope: general chat, create-memory-via-chat, intent routing, agents,
tool calling, Llama answer generation, AI-side DB writes, authentication or
authorization, JWT decoding, partial reprocessing, embedding migrations.

## Hard invariants

- The AI Service never writes to MongoDB or the Vector DB. Vector integration
  is **read-only**.
- `userId` is trusted service-to-service input from the authenticated Backend.
  Never decode or verify the end user's JWT here.
- Vector search must apply `userId == request userId` as a **pre-filter** in the
  vector engine. Never search globally and filter afterward. Cross-user
  retrieval is a critical security failure.
- Reprocessing returns the **complete** current chunk set, never a diff and
  never only changed chunks. The Backend reconciles.

## Locked decisions

**Embeddings:** Qwen3-Embedding family, dimension 1024, chunk-level. Embedding
input is Title + Description/Content + WhySaved + Tags.

**Chunking:** hybrid, ~500 token soft target, ~50 token overlap. Prefer
sections, then paragraphs, then sentences, before any token-based fallback.

**Chunk IDs:** deterministic. `memoryId + chunkIndex + final chunk content`
always yields the same `chunkId`; any change to final content changes it.

**Retrieval:** Top-K 5, similarity threshold 0.75, deduplicate by `memoryId`,
memory score = max matching chunk score, unique Memories ordered by score. No
chunk above threshold returns the contract-defined no-match response.

## API contract

Do not silently modify field names, request shapes, response shapes, error
structures, required fields, or locked default values. When the architecture
documents are ambiguous or contradictory, report it before changing externally
observable behavior.

Error codes: `INVALID_INPUT`, `INVALID_REQUEST`, `EMBEDDING_FAILED`,
`CHUNKING_FAILED`, `VECTOR_SEARCH_FAILED`, `MODEL_UNAVAILABLE`, `TIMEOUT`,
`INTERNAL_ERROR`. Never expose stack traces, model internals, infrastructure
secrets, or raw exception details through public API responses.

## Privacy

Never log raw Memory content, raw user queries, embeddings, retrieved chunk
text, or `WhySaved`. Prefer structured operational metadata: correlation ID,
endpoint, duration, chunk count, result count, safe error code.

## Architecture

Keep business logic out of FastAPI route handlers. Use dependency injection and
provider interfaces. Embedding and Vector Search providers must be abstracted
away from application services. Do not depend directly on a specific Vector DB
until the production one is explicitly selected.

Dependency direction: API -> services -> domain components / provider
interfaces -> infrastructure adapters.

## Quality gate

For every phase: inspect the repository first, preserve existing behavior, add
or update tests, then run all of:

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src
```

Report every failure. **Do not claim success unless the commands actually
passed.** Do not implement future phases unless explicitly requested. Prefer
small, reviewable changes over large rewrites.

At the end of every phase report: files created, files modified, behavior
implemented, design decisions, tests added, commands executed, exact test
results, unresolved issues, and the recommended next phase.
