# Memovo AI Service — deployment and integration guide

Status date: 2026-09-13. Contract: [AI_CONTRACT_FINAL.md](AI_CONTRACT_FINAL.md) v1.9 plus the
approved addendum in its section 21.9. Plan: [SPRINT2_GENERATION_PLAN.md](SPRINT2_GENERATION_PLAN.md).

This guide describes what the current tree does and how it was verified. It separates three
levels explicitly: **implemented and locally tested** (deterministic fakes, no network),
**Backend-integration ready** (needs the Backend's Atlas data and validators), and **externally
verified** (a real provider or cluster answered). Nothing in this document claims the third
level unless it says so.

## 1. What the service exposes

| Endpoint | Contract | Needs | Status |
|---|---|---|---|
| `POST /ai/memories/process` | §7–13 | embedding model | implemented, locally tested; Backend acceptance pending |
| `POST /ai/memories/search` | §14–16 | embedding model, Atlas (read-only) | implemented, locally tested; live Atlas pending |
| `POST /ai/chat/memories` | §17 | the above + generation | implemented, locally tested with a fake model; real model **not** verified |
| `POST /ai/memories/prepare-note` | §6 | generation | implemented, locally tested with a fake model; real model **not** verified |
| `GET /health` | operational | — | liveness: the process serves HTTP. Never depends on the model, Atlas or generation |
| `GET /ready` | operational | — | readiness: the retrieval services are built and usable |

Readiness certifies the embedding model and vector adapter were built. It does **not** probe
Atlas or the generation provider, and it does not change when generation is disabled: an
instance with generation off stays in rotation and the two generation endpoints answer
`503 GENERATION_UNAVAILABLE` on their own.

**Process startup and service readiness are two different events.** The HTTP process is up
within seconds; the retrieval services follow once the embedding model is loaded, which on a
cold cache includes downloading it. The lifecycle every deployment goes through:

| Step | `/health` | `/ready` | AI endpoints |
|---|---|---|---|
| container starts; configuration validated | — | — | — |
| invalid production configuration (`MEMOVO_ENV=production` with `MEMOVO_VECTOR_PROVIDER=fake`) | process exits, `service.startup_rejected` logged | | |
| HTTP bound; model downloading / loading in the background | `200 {"status":"alive"}` | `503 {"status":"loading"}` | `503 MODEL_UNAVAILABLE` (retryable) |
| build finished | `200` | `200 {"status":"ready"}` | serving |
| build failed (weights missing, Atlas URI unset) | `200` | `503 {"status":"unavailable"}` | `503 MODEL_UNAVAILABLE`; `service.startup_failed` logged |
| shutdown began | `200` until exit | `503 {"status":"unavailable"}` | draining |

A platform healthcheck or liveness probe must therefore target `/health`. Pointing it at
`/ready` turns a normal cold load into a killed deployment. Traffic routing is the Backend's
concern: it owns retries, and `MODEL_UNAVAILABLE` is retryable, so requests that arrive during
the load are retried rather than lost.

## 2. Build and run

```bash
docker build -t memovo-ai:<release> .   # no weights in the image; downloaded on first start
```

The image holds the embedding runtime but not the weights: `Qwen/Qwen3-Embedding-0.6B` is
downloaded into `HF_HOME` (`/home/memovo/.cache/huggingface`) on first start and loaded from
that cache afterwards, so the runtime needs outbound access to `https://huggingface.co` on a
cold cache. The download and load happen in the background after the HTTP process is up:
`/health` answers immediately and `/ready` reports `loading` until they finish (see §1). Set
`HF_HUB_OFFLINE=1` to forbid the download and require a pre-populated cache. No generation
weights are ever downloaded: generation is a hosted API call. The runtime also needs outbound
access to Atlas and, when generation is enabled, to `https://openrouter.ai`.

Two optional pieces of runtime environment make cold starts faster and more reliable. Neither
is a `MEMOVO_*` setting and neither is parsed by the service:

- `HF_TOKEN`: a Hugging Face **read** token, which is all a public model needs. The
  `huggingface_hub` library picks it up on its own and gets higher rate limits and more reliable
  downloads. It is a secret: keep it in `.env` locally (ignored by git) or in the platform's
  secret store, never in the image, the compose file or the repository. A missing token is not
  an application error; the download runs unauthenticated and logs a one-line warning.
- A persistent volume mounted at `/home/memovo/.cache/huggingface`, so a replacement container
  finds the weights already downloaded and reaches `ready` in the time of a load from disk, not
  a download. Operational configuration, not a secret. Without it every new container downloads
  the model again.

```bash
cp .env.example .env            # then edit; never commit .env
docker run -d --name memovo-ai --env-file .env --read-only --tmpfs /tmp \
  -v memovo-hf-cache:/home/memovo/.cache/huggingface \
  --security-opt no-new-privileges:true -p 127.0.0.1:8000:8000 memovo-ai:<release>
```

Changing `.env` requires recreating the container; `docker restart` does not reread it.
`docker-compose.yml` sets `MEMOVO_ENV` and `MEMOVO_VECTOR_PROVIDER` in its `environment:` block,
which wins over `.env`. The service is internal: never publish it beyond the Backend network.

### Railway

The image is Railway-compatible as built: the `CMD` listens on the `PORT` Railway injects
(falling back to 8000 elsewhere) with one Uvicorn worker, and the weights are not baked in. The
settings below are configured in the Railway service, not in the repository.

| Setting | Value | Why |
|---|---|---|
| Healthcheck path | `/health` | Liveness. It answers as soon as the process is up. It is **not** `/ready`: readiness is 503 for the whole cold load, and a healthcheck on it kills a healthy deployment |
| Healthcheck timeout | default (300 s) is enough | `/health` answers within seconds of container start; the model load no longer gates it |
| `PORT` | leave to Railway | Railway injects it and sends healthchecks to it; the container honours it. Do not hardcode a different value |
| `HF_TOKEN` | secret variable, a Hugging Face read token | Optional. Better rate limits and reliability while downloading a cold cache; never committed |
| Volume | mount at `/home/memovo/.cache/huggingface` | Optional but recommended. The first healthy deployment downloads the model; every later container replacement reuses the cache and becomes ready in seconds instead of minutes. Not a secret |
| `MEMOVO_ENV` | `production` | Enables the production guards |
| `MEMOVO_VECTOR_PROVIDER` | `atlas` | `fake` is refused in production and the process exits |
| `MEMOVO_ATLAS_URI` | secret variable | Required for `atlas`; unset leaves the instance alive but unready (`service.startup_failed`, `VectorSearchUnavailableError`) |
| `MEMOVO_ATLAS_DATABASE`, `MEMOVO_ATLAS_COLLECTION`, `MEMOVO_ATLAS_INDEX` | as provisioned by the Backend | Defaults in §3 |
| `MEMOVO_GENERATION_ENABLED`, `MEMOVO_OPENROUTER_API_KEY` | only if generation is wanted | Retrieval readiness never depends on them |
| Memory | at least 1 GB | The loaded model is roughly 0.8 GB resident on CPU |

What the deploy log shows on a cold cache, in order: `service.initialization_started`, the
Hugging Face download lines, `service.started` with `services_available: true`, and `/ready`
turning 200. `Uvicorn running on http://0.0.0.0:<PORT>` appears before the download begins;
if it never appears, the problem is process start, not the model.

## 3. Configuration

All settings are `MEMOVO_*` environment variables (see `.env.example` for every one).

| Variable | Default | Notes |
|---|---|---|
| `MEMOVO_ENV` | `development` | `production` refuses `MEMOVO_VECTOR_PROVIDER=fake` at startup |
| `MEMOVO_VECTOR_PROVIDER` | `fake` | `atlas` for a real index |
| `MEMOVO_ATLAS_URI` | empty | secret; required for `atlas` |
| `MEMOVO_ATLAS_DATABASE` | `memovo` | confirm with Backend |
| `MEMOVO_ATLAS_COLLECTION` | `memory_vectors` | contract §12 |
| `MEMOVO_ATLAS_INDEX` | `vector_index` | contract §12 |
| `MEMOVO_SEARCH_TOP_K` / `MEMOVO_SEARCH_SIMILARITY_THRESHOLD` | `5` / `0.75` | locked; see §7 |
| `MEMOVO_GENERATION_ENABLED` | `false` | processing/search never need generation |
| `MEMOVO_GENERATION_PROVIDER` | `openrouter` | the only provider; no fake by configuration |
| `MEMOVO_GENERATION_MODEL` | `nvidia/nemotron-3-super-120b-a12b:free` | the approved free replacement; see §8 |
| `MEMOVO_OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | |
| `MEMOVO_OPENROUTER_API_KEY` | empty | secret; enabled without it logs `generation.startup_failed` and generation answers 503 |
| `MEMOVO_GENERATION_TIMEOUT_SECONDS` | `60` | per call, including the wait for a slot |
| `MEMOVO_GENERATION_MAX_CONCURRENCY` | `4` | in-flight generation calls per process |
| `MEMOVO_GENERATION_CHAT_MAX_OUTPUT_TOKENS` | `1024` | sent as `max_tokens` |
| `MEMOVO_GENERATION_NOTE_MAX_OUTPUT_TOKENS` | `4096` | a cleaned note can approach the 10,000-character Note limit |
| `MEMOVO_GENERATION_CONTEXT_WINDOW_TOKENS` | `40960` | budget assumption; history is never trimmed to fit |
| `MEMOVO_GENERATION_CHARS_PER_TOKEN` | `2.0` | conservative estimate used by the budget check |
| `MEMOVO_GENERATION_RETRY_AFTER_MAX_SECONDS` | `30` | cap on a forwarded `Retry-After` |

The generation limits are provisional operating defaults. They were **not** measured against the
real endpoint (see §8) and must be re-baselined before production.

### Atlas index the Backend must provision

```json
{
  "fields": [
    {"type": "vector", "path": "embedding", "numDimensions": 1024, "similarity": "cosine"},
    {"type": "filter", "path": "userId"}
  ]
}
```

Each document in `memory_vectors` carries `userId`, `memoryId`, `chunkId`, `chunkIndex`,
`content`, `title`, `tags`, `embedding`. The AI Service only reads; the Backend writes,
reconciles by `chunkId`, and deletes.

## 4. Requests and responses

### Process — Note

```json
{"type":"note","memoryId":"67890abcdef1234567890abc","title":"MongoDB Performance Tips",
 "content":"Use indexes for frequently queried fields...","tags":["database","performance"]}
```

### Process — Link

```json
{"type":"link","memoryId":"12345abcdef6789012345abc","url":"https://example.com/article",
 "title":"Understanding Vector Embeddings","content":"Good reference for understanding embeddings",
 "tags":["ai","embeddings"],
 "source":{"sourceTitle":"Tech Blog","sourceDescription":"Articles about AI and ML","authorName":"Jane Doe","publicationDate":"2026-09-01"},
 "extractedContent":"Vector embeddings are numerical representations..."}
```

Limits are exactly the contract's (§8.1–8.4). `source` must carry all four keys (values may be
`null`); `{}` is rejected. `whySaved`, `description`, `about`, `platform`, `contentType`,
`siteName` and other legacy keys are rejected with 422. The URL is never fetched.

Response: `{"memoryId": "...", "chunks": [{"chunkId","chunkIndex","content","embedding"}]}` —
the complete current set, 1024 finite floats per chunk. Embedded text for a Note is
`Title / Content / Tags`; for a Link it is `Title / Content / Source / Extracted Content / Tags`.
The URL is not embedded.

### Search

Request `{"userId":"...","query":"..."}` and nothing else (`type`, `limit` are rejected).
Response `{"found":true,"results":[{"memoryId","score","title","tags","chunks":[{"chunkId","content"}]}]}`
or exactly `{"found":false,"results":[]}`.

### Chat

Request `{"userId","message","conversationId"?,"history"?}`; `history` items are
`{"role":"user"|"assistant","content"}`. History arrives as the Backend bounded it and is passed
to the model whole; the service never trims it and does not enforce the 30/6,000/30,000 limits.
`conversationId` is an identifier only. Response `{"found","answer","sources":[{"memoryId","score","title","chunks":[{"chunkId","content"}]}]}`.
`found:false` only when retrieval returns nothing (then no model call is made). Insufficient
evidence is `found:true` with an explanatory answer. `sources` lists only memories the model
reported using; no inline citation markers reach the answer.

### Prepare note

Request `{"content":"..."}`; response `{"title":"...","content":"..."}`. The prepared title is
bounded to 200 characters and the content to 10,000 (the Note limits the Backend will apply);
anything else from the model is reported as invalid output, not handed on.

## 5. Errors (contract §21 and §21.9)

| HTTP | code | retryable | when |
|---|---|:---:|---|
| 422 | `INVALID_INPUT` | false | schema/limit violation; also a request over the generation context budget |
| 400 | `INVALID_REQUEST` | false | transport-level client error (404/405 keep their status) |
| 429 | `RATE_LIMITED` | true | generation provider rate limited; bounded `Retry-After` when supplied |
| 500 | `INTERNAL_ERROR` | true | defect |
| 500 | `CHUNKING_FAILED` | true | chunking defect |
| 502 | `AI_INVALID_RESPONSE` | **false** | malformed/truncated model output or schema violation; never retry |
| 503 | `EMBEDDING_FAILED` / `MODEL_UNAVAILABLE` / `VECTOR_SEARCH_FAILED` | true | retrieval dependencies |
| 503 | `GENERATION_UNAVAILABLE` | true | generation disabled, unconfigured, unreachable, or upstream failure |
| 504 | `TIMEOUT` | true | vector search or generation deadline |

Classification is HTTP status + code; `retryable` matches it. The Backend owns retries
(3 total attempts, exponential backoff, bounded `Retry-After`); the service and its HTTP client
never retry. `X-Request-Id` is echoed on every response, including errors, and appears in every
log record for the request.

## 6. Migration from the pre-v1.9 wire format

Breaking changes the Backend must deploy together with this release:

| Area | Before | Now |
|---|---|---|
| Process request | `memoryId,title,description,whySaved,tags(+about,source{siteName,favicon,ogImage,publishedAt})` | `type` discriminator; Note `title,content,tags?`; Link `url,title,content?,tags?,source{4 required nullable},extractedContent?` |
| Process response | `intent, content, chunks` | `memoryId, chunks` |
| Search response chunks | `chunkId, chunkIndex, content` | `chunkId, content` |
| Search no-match | `found,results,message` | `{"found":false,"results":[]}` |
| Atlas defaults | `memory_chunks` / `memory_chunks_vector_index` | `memory_vectors` / `vector_index` |
| `INTERNAL_ERROR`, `CHUNKING_FAILED` | retryable `false` | retryable `true` (5xx rule) |
| New codes | — | `GENERATION_UNAVAILABLE`, `RATE_LIMITED`, `AI_INVALID_RESPONSE` |

Canonical embedded text changed (no `Why Saved`/`About` sections; new `Source` fields;
`Extracted Content`), so chunk IDs and embeddings of existing records change on reprocessing.
The Backend reprocesses every memory through `/process` and reconciles by `chunkId`; a Note
that never had a `whySaved` composes identically to before and keeps its IDs.

Rollback means rolling back **both** sides: the previous image will reject the new request
shape (422) and the Backend validator will reject the old response shape.

## 7. Retrieval score and threshold

Unchanged during this migration: Top-K 5, threshold 0.75, max chunk score, dedupe by
`memoryId`, `userId` pre-filter inside `$vectorSearch`. The fake evaluator scores raw cosine;
Atlas forwards `vectorSearchScore`. The two scales are not the same, and no Atlas-backed
evaluation has run. Do not copy the fake-evaluation figures (see `eval/README.md`) into an
Atlas deployment; an Atlas threshold sweep with seeded data for two users is a precondition
for any threshold change.

## 8. Generation: verified and unverified

Verified locally (no network): provider interface, deterministic fake, bounded concurrency and
deadlines, cancellation releasing slots, one attempt per call, error mapping for 429/5xx/
401/402/403/404/timeouts/network/malformed bodies, reasoning-trace stripping, output
validation, source-handle validation, history passed untrimmed, injection-in-data placement,
privacy of logs, mixed load with probes answering during generation.

**Provider update, 2026-09-13:** the replacement local key passed authentication (HTTP 200).
The owner authorized a free replacement for the unavailable Qwen free ID.
`nvidia/nemotron-3-super-120b-a12b:free` is listed with zero input/output token pricing
and passed a synthetic JSON completion through the project adapter. It is now the
configured default; there is no automatic model or paid fallback.
The conservative application context budget remains unchanged pending evaluation.

The [free endpoint policy](https://openrouter.ai/nvidia/nemotron-3-super-120b-a12b:free)
states that sessions are logged and confidential/personal data must not be uploaded.
Use synthetic data for this development configuration. This is not production approval
for sending real user memories. Live Atlas, quality acceptance and release checks remain separate.

Run the opt-in synthetic checks:

```bash
MEMOVO_RUN_OPENROUTER_SMOKE=1 uv run pytest tests/integration/test_openrouter_live.py -s
MEMOVO_GENERATION_ENABLED=true uv run python scripts/evaluate_generation.py
```

The smoke test sends one synthetic request; the evaluation sends the synthetic cases in
`eval/datasets/generation_chat.json` and `eval/datasets/note_preparation.json` and writes every
answer to `eval/results/` for human review against each case's rubric. Numerical targets are
set with the owner from those reviews. Thinking-mode behaviour (reasoning tokens counting
against `max_tokens`) and `response_format` support must be confirmed there before the output
budgets are finalized.

Provider data policy (retention, training use, region) must be reviewed before real memories
are sent to any hosted model.

## 9. Verification status of this tree

| Check | Result |
|---|---|
| `uv run pytest` | 1746 passed, 16 skipped (opt-in Atlas, embedding-model and OpenRouter tests) |
| `uv run ruff check .` / `ruff format --check .` / `mypy src` | clean |
| Hosted CI | never executed: no git remote |
| Live Atlas | not run |
| Real generation model | Free Nemotron smoke passed; 11 chat and 6 note cases completed (2026-09-13); see results below |
| Docker image | built 2026-09-15 without weights (433 MB compressed); a full cold start, download included, reached `/ready` 200 locally. The background-load lifecycle is covered by `tests/integration/test_startup_lifecycle.py` with a blocking fake build, not by a real download |

### Free-model evaluation, 2026-09-13

Local synthetic report: `eval/results/generation_20260913T084855Z.json`.
The live smoke passed. All 11 fixed-evidence chat cases completed with zero errors
or invalid outputs and 100% automated found/source/mention checks. All 6 note cases
completed with zero errors or invalid outputs, 100% lexical preservation and valid titles.
One lexical addition flag appeared: `injection_content` retained the original `PWNED`
instruction-like sentence as content, while returning a bill-related title. Manual
inspection confirms this is allowed by the case rubric; the lexical flag conflicts with
that rubric and does not demonstrate an added fact or executed instruction. Preserve
the raw result; align the evaluation criterion separately.
These synthetic checks do not close human grounding review, live Atlas, production
privacy approval or performance acceptance. No real user memories were sent.

## 10. Owners

| Item | Owner |
|---|---|
| Atlas provisioning, index, data writes, reconciliation, cleanup | Backend |
| OpenRouter credentials, free-provider limits and production data-policy suitability | AI owner |
| Backend validator updates for the new shapes and codes | Backend |
| Reprocessing of existing memories after cutover | Backend |
| Score/threshold decision after an Atlas-backed sweep | AI + Product |
| Human review of generation evaluation outputs; quality targets | AI owner + Product |
| Publishing the repository, running CI, tagging the release | repository owner |
