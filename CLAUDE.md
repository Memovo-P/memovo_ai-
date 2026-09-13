# Memovo AI Service - Engineering Rules

The wire contract is `docs/AI_CONTRACT_FINAL.md` (v1.9 plus its section 21.9
addendum) and the implementation plan is `docs/SPRINT2_GENERATION_PLAN.md`.
Read both before making changes. The sibling `memovo-ai-docs/` repository is
implementation history: its engineering and security rules still apply, but
where its prose requires legacy process fields (`whySaved`, `description`) or
excludes the generation endpoints, the adopted contract wins.

Do not redesign locked product or architecture decisions unless explicitly
instructed.

## Scope

Four contract endpoints: `POST /ai/memories/process`, `POST /ai/memories/search`,
`POST /ai/chat/memories` (RAG-only, read-only) and
`POST /ai/memories/prepare-note`.

Out of scope: general-purpose chat, intent routing, agents, tool calling,
memory mutation, AI-side DB writes, authentication or authorization, JWT
decoding, URL fetching, AI-side conversation state, embedding migrations,
any generation model or provider other than the approved one.

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
input is Title + Content + Tags for a Note, plus Source metadata and Extracted
Content for a Link. `whySaved` is removed and never mapped; the URL is not
embedded.

**Generation:** OpenRouter, exactly `nvidia/nemotron-3-super-120b-a12b:free`, disabled by default.
No paid or alternate model fallback, no SDK retries. Malformed model output is
`502 AI_INVALID_RESPONSE`, never retryable. History is passed to the model as
the Backend bounded it and is never trimmed here.

**Chunking:** hybrid, ~500 token soft target, ~50 token overlap. Prefer
sections, then paragraphs, then sentences, before any token-based fallback.

**Chunk IDs:** deterministic. `memoryId + chunkIndex + final chunk content`
always yields the same `chunkId`; any change to final content changes it.

**Retrieval:** Top-K 5, similarity threshold 0.75, deduplicate by `memoryId`,
memory score = max matching chunk score, unique Memories ordered by score. No
chunk above threshold returns exactly `{"found": false, "results": []}`.

## API contract

Do not silently modify field names, request shapes, response shapes, error
structures, required fields, or locked default values. When the architecture
documents are ambiguous or contradictory, report it before changing externally
observable behavior.

Error codes: `INVALID_INPUT`, `INVALID_REQUEST`, `EMBEDDING_FAILED`,
`CHUNKING_FAILED`, `VECTOR_SEARCH_FAILED`, `MODEL_UNAVAILABLE`, `TIMEOUT`,
`INTERNAL_ERROR`, `GENERATION_UNAVAILABLE`, `RATE_LIMITED`,
`AI_INVALID_RESPONSE`. Classification is HTTP status + code; `retryable` is
consistent metadata (every 5xx retryable except `502 AI_INVALID_RESPONSE`).
Never expose stack traces, model internals, provider error bodies,
infrastructure secrets, or raw exception details through public API responses.

## Privacy

Never log raw Memory content, raw user queries, chat messages or history,
prompts, generated answers, prepared notes, embeddings, retrieved chunk text,
provider error bodies or API keys. Prefer structured operational metadata:
correlation ID, endpoint, duration, chunk/evidence/source counts, token
counts, safe error code, safe provider and model identifiers.

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
