# Memovo AI — Contract Migration and Generation Plan

Updated: 2026-09-12. Status: implementation in progress; this document is not a completion report.

## 1. Authority and purpose

The owner has adopted [AI_CONTRACT_FINAL.md](AI_CONTRACT_FINAL.md), currently v1.9, and the final Backend clarification in section 4 below. This plan is the single implementation plan. It replaces D0_DECISIONS.md and all previous proposals in this file. There are no archived alternatives to choose from.

The contract calls Memory Chat Sprint 1; “Sprint 2” here names the AI implementation phase. Implement the agreed features rather than reopening sprint naming. Earlier sibling specifications or repository prose that prohibit generation or require legacy process fields are superseded for this authorized work. Keep their unrelated engineering/security rules.

Deliver four synchronous HTTP operations:

| Operation | Endpoint | Contract |
|---|---|---|
| Process persisted memory | POST /ai/memories/process | Sections 7–11 |
| Retrieve memories | POST /ai/memories/search | Sections 14–16 |
| Generate a grounded memory answer | POST /ai/chat/memories | Section 17 |
| Prepare an explicitly requested note | POST /ai/memories/prepare-note | Section 6 |

Backend owns authentication, authorization, conversation persistence, Atlas provisioning/indexes/writes, reconciliation/cleanup, link extraction, queues, retries and public-response mapping. AI owns computation and read-only user-scoped retrieval. A synchronous HTTP contract does not require blocking the event loop.

No general assistant, intent classifier, agents/tools, automatic memory mutation, AI sessions, JWT verification, URL fetching, OCR/multimodal, streaming, or embedding-model changes are included.

## 2. Resume from the actual working tree

Observed on 2026-09-12, without running the suite:

| Area | Observed state | Next action |
|---|---|---|
| P0 operational work | Production guard, request correlation and AiServiceError logging changes exist | Review and preserve; rerun relevant regressions, do not rebuild from scratch |
| P1 process | Note/Link discriminated schemas, four-field source, new process response and content composition exist locally | Audit against section 3; finish callers, fixtures and tests |
| P2 search/config | Search response changes and memory_vectors/vector_index defaults exist locally | Verify service mapping and migrate consumers/tests |
| Evaluation | Harness/dataset have local edits | Preserve them; check compatibility, do not claim new model evaluation |
| Tests | Numerous contract/integration tests still reference description/whySaved or search NO_MATCH_MESSAGE | Finish intentional test migration before declaring P1/P2 complete |
| Generation | No generation/chat/preparation implementation found in inspected source paths | Recheck before starting P3; another working session may have progressed |
| Error addendum | AI_INVALID_RESPONSE was not found in inspected source or canonical file | Implement section 4 and synchronize the canonical documentation |
| Deployment docs | docs/deployment.md was already deleted before this cleanup | Do not restore outdated instructions; produce a current guide at handoff |

This is a snapshot, not permission to overwrite newer work. Start with git status/diff and relevant source reads. No prior test count proves the current tree passes. Do not reset, checkout away changes, or run a blanket rewrite of work already done.

## 3. Finalized interface decisions — do not reopen

### Processing

| Field | Note | Link |
|---|---|---|
| type | Required, non-null, note | Required, non-null, link |
| memoryId | Required, non-null, non-empty identifier | Same |
| title | Required, non-null, 0–200 characters | Required, non-null, 1–500 characters |
| content | Required, non-null, 1–10,000 characters | Optional/nullable, empty allowed, maximum 1000 characters |
| tags | Optional/nullable, maximum 20 strings, maximum 50 characters each | Same |
| url | Not a Note field | Required, HTTP/HTTPS, maximum 2048 characters |
| source | Not a Note field | Required, non-null; exact shape below |
| extractedContent | Not a Note field | Optional/nullable, empty allowed; no added AI-specific character limit |

Every source key is required; each value is string or null:

```json
{"sourceTitle":null,"sourceDescription":null,"authorName":null,"publicationDate":null}
```

Do not accept platform/contentType/thumbnailUrl/canonicalUrl or legacy siteName/favicon/ogImage/publishedAt. Do not impose ISO-only parsing on publicationDate. source={} is invalid. Do not add the old proposed ID, extraction, or source-text length caps.

whySaved is removed: do not expect, accept, embed, or automatically map it. Link content is user context; extractedContent is Backend-extracted page text; source is selected metadata. AI never fetches the URL.

Return memoryId matching the request and the complete current chunks array. Each chunk has chunkId, chunkIndex, content and a 1024-dimensional finite numeric embedding. Remove legacy intent and top-level canonical content from this response. Keep chunkIndex in processing/internal storage.

### Search

Input is userId/query only. No AI memory-type filter or public limit; Backend handles both. Keep top-5 chunks, max matching chunk score, deduplication and userId pre-filter behavior.

Success: found/results, each result with memoryId, finite score, title, tags, chunks. Search chunks expose chunkId/content only, not chunkIndex. Backend supplies public fields through mapping/enrichment.

No-match is exactly:

```json
{"found":false,"results":[]}
```

No message field. Backend type filtering after AI top-k can yield fewer public results; do not compensate by widening retrieval or adding a type pre-filter.

### Chat/history

Required: userId and message. Optional: history and conversationId, with types from contract section 17. Optional must not be interpreted as nullable without evidence.

History allows user/assistant roles only: at most 30 messages, 6000 characters per message content, 30000 total content characters. Backend validates and retains the newest COMPLETE messages within the limits. AI must not trim history, partially truncate messages, or silently drop it to satisfy its token budget.

conversationId is an identifier only. Backend owns/persists/loads/authorizes conversation state and sends bounded history for each request. AI neither stores sessions nor retrieves history by ID. The ID does not change retrieval behavior. Do not inherit old speculative 128-character ID or 4000-character current-message limits.

Return found/answer/sources. found=false only if no relevant memories exist. If memories exist but cannot support the requested answer, found=true and clearly explain the insufficiency without unsupported facts. sources contains only memories actually used/referenced. No sufficient/status fields, inline [1]/[2] citations, public source IDs/indexes, or alternate citations schema.

### Prepare-note

Input content; output title/content only, following section 6. Preserve meaning and substantive facts. No generated tags, IDs, timestamps, retrieval dependency or persistence. Backend validates, creates a Note with tags=[], then queues normal processing. Do not invent endpoint-specific limits by copying Chat history limits.

## 4. Final Backend error clarification — approved addendum

When AI detects malformed model/generated output or output-schema violation BEFORE responding:

```text
HTTP 502 Bad Gateway
error.code = AI_INVALID_RESPONSE
error.retryable = false
```

This is ALWAYS non-retryable. It is an explicit exception to retryable service/transient 5xx. Backend also does not retry an HTTP-200 body that fails canonical schema validation. Never intentionally emit invalid HTTP 200 to signal failure.

Canonical HTTP status + error.code classification takes precedence over error.retryable. The latter is consistent metadata, not an override.

| Failure | Retry |
|---|---|
| 400 validation/bad request; 401/403; deterministic 4xx | No |
| Transient 429 | Yes; bounded Retry-After when supplied |
| Service/transient 5xx; timeout; network/connection failure | Yes |
| 502 + AI_INVALID_RESPONSE; malformed output/schema violation | No |

Backend owns at most 3 TOTAL attempts with exponential backoff and bounded waits. AI and its SDK must not add automatic retries. This clarification does not introduce authentication inside AI or change the public Flutter API.

Review older INTERNAL_ERROR/CHUNKING_FAILED metadata for consistency with canonical classification; do not blindly preserve contradictory old tables. Distinguish malformed output from upstream outage, invalid caller input, and valid insufficient-evidence answers. Add AI_INVALID_RESPONSE to canonical documentation if absent, using this approved addendum; do not ask this question again.

## 5. Approved model and prepared configuration

- Generation model: NVIDIA Nemotron 3 Super (free), selected with owner authorization on 2026-09-13 after the original Qwen free ID became unavailable.
- API provider: OpenRouter; hosting: Hosted API.
- Exact requested model ID: nvidia/nemotron-3-super-120b-a12b:free.
- Base URL: https://openrouter.ai/api/v1; chat/completions operation.
- No paid-model fallback, alternate model, or openrouter/free routing without new owner approval.
- Embeddings remain Qwen/Qwen3-Embedding-0.6B at 1024 dimensions.

Prepared environment names:

```dotenv
MEMOVO_GENERATION_ENABLED=false
MEMOVO_GENERATION_PROVIDER=openrouter
MEMOVO_GENERATION_MODEL=nvidia/nemotron-3-super-120b-a12b:free
MEMOVO_OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
MEMOVO_OPENROUTER_API_KEY=
```

The owner fills the key locally. These names do not prove settings/adapter implementation exists. Preserve existing .env values, never print secrets, and never put a real key in .env.example. Processing/search must not require generation credentials.

Verify current availability and supported parameters against OpenRouter documentation and an opt-in synthetic smoke test. A catalog listing is not successful inference evidence. Free-tier capacity is not a production SLA. Do not install generation weights/GPU dependencies in the embedding image. Runtime needs outbound Atlas/OpenRouter access. Review provider retention/training-use settings before real personal memories are sent.

References: [model listing](https://openrouter.ai/nvidia/nemotron-3-super-120b-a12b:free), [OpenRouter documentation](https://openrouter.ai/docs/quickstart). Recheck at implementation time.

## 6. Work sequence and phase acceptance

### P0 — preserve and verify operational baseline

Inspect existing guard/correlation/error-logging work. Check generated and supplied request IDs across success, validation, service errors, provider failures, timeout and unexpected exceptions. Verify safe logging and startup rejection before model loading. Keep all-or-nothing startup limitations documented; do not redesign it merely to resume this plan.

Gate: relevant regressions pass. If schema migration currently prevents test collection, record the cause and resolve it as part of P1/P2 rather than reverting source.

### P1 — finish processing migration already underway

Audit the new union/schemas, route/service mapping, normalization, source selection, response, exports, evaluation fixtures and all callers. Preserve valid current work. Finish outdated contract/integration/unit tests to reflect section 3 while preserving determinism, privacy and read-only invariants.

Define deterministic canonical section order and test it; keep user content/source/extracted text distinct. Empty sections do not render literal null/None. Validate URL format as required without dereferencing it. Do not redesign chunking or hash framing without evidence.

Canonical text changes can change IDs and embeddings for existing records. Backend owns reprocessing and complete-set reconciliation. A new ID replaces an absent old ID; do not mutate content under an old ID solely because the chunk position matches. There is no guarantee that one paragraph edit changes exactly two chunks or that embeddings are byte-identical across machines.

Gate: finalized Note/Link boundary/null/omitted/extra-field tests, response memoryId, complete chunk sets, finite 1024 vectors, stable IDs, no fetch/write, and migrated consumers pass.

### P2 — finish search/config migration already underway

Verify memory_vectors/vector_index defaults and env overrides; confirm database with Backend. Remove wire-only search chunkIndex/message while retaining internal order/index metadata. Do not add type/limit input. Migrate service tests and consumers still importing NO_MATCH_MESSAGE.

Separate score calibration from schema changes. Preserve 0.75 and retrieval logic during migration. The fake evaluator uses raw cosine while Atlas forwards engine scores; compare known vectors and run an Atlas-backed evaluation before approving a score/threshold change. Do not reuse the old raw-cosine 0.40 recommendation as production configuration.

Gate: Backend-compatible JSON and local regression suite pass. Live positive retrieval and two-user isolation require Backend fixtures; skipped/empty tests do not certify Atlas.

### P3 — generation abstraction and OpenRouter adapter

Build an injectable async provider interface, deterministic fake and one real adapter for the approved model. Implement the prepared settings with a secret type for the key. Separate Chat and Prepare Note prompts/output schemas despite sharing a provider.

Enforce bounded concurrency, timeout/cancellation, output budgets and complete-response validation. Disable SDK retries and model fallbacks. Verify supported structured-output parameters; do not assume every OpenAI-compatible feature is supported. Keep reasoning/provider internals out of answer/logs.

Use AI_INVALID_RESPONSE for malformed/truncated output that fails the expected output contract; use canonical transient errors for unavailable/rate-limited/network paths. Establish safe fixed public messages and consistent metadata.

Gate: mocked success/429/timeout/network/unavailable/malformed/cancel paths, exact model selection, no fallback/retry, secret hygiene, and optional synthetic live smoke evidence.

### P4 — grounded memory chat

Pipeline: validate -> contextualize current query if necessary -> existing user-scoped retrieval in-process -> no-match short circuit -> bounded evidence/source map -> generation -> output/reference validation -> found/answer/sources.

History is context for interpreting follow-ups, not factual memory evidence. Retrieve afresh each turn; old answers may cite deleted or changed memories. For ambiguous references, use a bounded, evaluated query-contextualization strategy or a clear clarification/abstention; do not invent facts or introduce intent routing. No-match skips ANSWER generation; if contextualization uses a model before retrieval, measure that call separately.

Budget instructions, current message, ALL accepted history, metadata, evidence and output reserve against actual model capacity. Select evidence deterministically; omitted evidence is not a used source. Do not secretly summarize/trim accepted history. If the request cannot fit, return a documented valid failure rather than pretending to answer from complete context. Choose budgets from measured provider capabilities, not old D0 numbers.

Use internal request-local source handles to validate references; reconstruct source IDs/scores/titles/chunks from actual retrieval, not generated metadata. Return only used evidence. The public answer has no inline citation markers. Source membership checks do not prove semantic entailment; evaluate grounded claims separately.

Treat memories/history/metadata as untrusted data, separate from system instructions. No tools, writes or background-knowledge fallback. Handle no-match, insufficient evidence and conflicts distinctly from provider failures without adding fields.

Gate: direct/multi-memory/follow-up/ambiguous/no-match/insufficient/conflicting cases; history limits/roles; cross-user isolation; unknown-source rejection; injection and invalid-output errors; consumer response validation.

### P5 — explicit note preparation

Implement content -> title/content using its own prompt. Preserve numbers, names, dates and meaning; no substantive additions/tags. No retrieval, session or database dependency. Backend creates the Note only after a valid result, then queues process.

Gate: Arabic/English/mixed-language fidelity, boundaries from the contract, injection resistance, no additional fields/writes, malformed-output classification and provider failure behavior. JSON validity alone is not sufficient acceptance.

### P6 — runtime and observability integration

Keep generation disabled until configured/verified. Enabling fake generation in production must not serve fabricated answers. Preserve process/search when generation is disabled/unavailable according to explicit capability readiness policy; do not silently redefine /ready for all traffic.

/health remains liveness. Startup loading may prevent HTTP acceptance; readiness does not establish Atlas/remote-provider availability. Test event-loop responsiveness, cancellation/slot release and mixed processing/search/chat/preparation load.

Log only safe request IDs, codes/outcomes, durations, token/evidence/source counts and safe model/provider identifiers. Never log raw query/history/evidence/prompts/answers/prepared notes/embeddings/keys/provider error messages. Choose timeouts/concurrency/output budgets from actual capacity; speculative old p95/GPU figures are not requirements.

Gate: consistent errors/correlation, no secret leakage, proper resource release, disabled/misconfigured provider behavior and healthy existing capabilities.

### P7 — quality and real integration

Maintain versioned synthetic or approved sanitized data with separate development and held-out sets. Record code/model/prompt/config/dataset versions. Evaluate retrieval, fixed-evidence generation, end-to-end multi-turn chat, and preparation fidelity separately.

Cover Arabic/English/mixed input, paraphrases, multiple memories, partial/conflicting evidence, dates/numbers, long/duplicate context, injection, two users with similar text, and edits/deletions between turns.

Measure correct/supported/complete claims, false answers and unnecessary abstentions, actual source usage, invalid references, preparation additions/omissions, cold/warm p50/p95 latency, concurrency/error rate, tokens/cost. Human review is required for grounding/fidelity; model judges are supplementary.

Set numerical quality/performance targets with the owner from representative tests. Require zero cross-user exposure and invalid source references escaping validation in the acceptance suite, without claiming universal hallucination prevention. Real Atlas needs seeded positive controls for both users. Fake tests do not close external acceptance.

### P8 — clean handoff and coordinated cutover

Update current README/runbook/examples to the approved contract, add a current deployment/integration guide when the behavior is verified, and remove stale links. Do not restore outdated process payloads just to satisfy artifact tests.

Coordinate breaking process/search changes with Backend. Backend owns data reprocessing/reconciliation and stale-job prevention. Rollback must account for consumer schema compatibility, not only image version.

Build and smoke-test the new image; record a pinned release and hosted CI results. Configuration changes require container recreation to reload env files. Do not guarantee model-layer caching or reproducible independent builds without evidence. Registry publishing is optional for initial integration.

Gate: four-endpoint examples/outcomes/errors, configuration, migration/rollback instructions, evaluation evidence, known limitations and owners, tested release reference, no committed secrets.

## 7. Execution rules and definition of done

Finish P1/P2 rather than restarting P0. P3 fake tests can proceed without a key/Atlas; P5 depends on P3 and can proceed independently of P4. External outages block live verification, not useful local implementation.

For each phase run focused tests then the existing gate:

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src
```

Default CI must not download models or contact paid/external services. Real dependencies are opt-in. Preserve existing work, update obsolete schema tests intentionally, and never delete privacy/isolation/domain checks just to make the gate pass.

Report actual changed files, wire behavior, commands/results, remaining dependencies and the next phase. Do not claim completion from documentation or historical green totals. No commits, pushes, deployments, Atlas writes/provisioning or public API redesign without separate user authorization.

Done means all four endpoints conform; Points 1–10 are implemented; real-provider grounding/fidelity and agreed budgets pass; Atlas isolation and score/threshold acceptance are evidenced; CI and coordinated handoff are complete. Until then distinguish local implementation, Backend integration readiness and production sign-off.
