# Deployment

Doc 06 sections 7–9 and 12 are the source of truth; this describes how this
repository implements them.

## Topology

```
Internet
   │
   ▼
Backend ──────────► MongoDB
   │
   ├──────────────► Vector DB   (WRITE — the Backend owns every write)
   │
   └──────────────► AI Service
                       │
                       ├── Qwen3 embeddings
                       │
                       └────► Vector DB   (READ only)
```

**The AI Service must not be publicly routable.** It performs no end-user
authentication by design (doc 06 §1–2, and `tests/integration/test_trusted_boundary.py`
pins that absence), so the network boundary is the only thing keeping it
private. Give it no ingress, no public load-balancer rule and no public DNS
name; the Backend reaches it over an internal network only.

`docker-compose.yml` publishes the port on `127.0.0.1` for local work. That
is a convenience for a developer machine, not a deployment pattern.

## What the image contains

Exactly the list in doc 06 §8: the FastAPI application, the Qwen embedding
runtime and tokenizer, the read-only vector adapter, configuration, and the
health/readiness endpoints.

It deliberately does **not** contain queue orchestration, a database writer,
or any credential. Secrets arrive through the environment at run time.

## Building

```bash
# Production: weights baked in, startup is offline and deterministic.
docker build -t memovo-ai:0.1.0 .

# Fast local build: no weights, downloads on first start.
docker build --build-arg BAKE_MODEL=false -t memovo-ai:thin .
```

Baking is the default on purpose. A thin image makes a Hugging Face outage
into a failed rollout, and makes two replicas of the same tag capable of
running different weights.

## Running

```bash
docker run --rm -p 127.0.0.1:8000:8000 \
  -e MEMOVO_VECTOR_PROVIDER=atlas \
  -e MEMOVO_ATLAS_URI="$MEMOVO_ATLAS_URI" \
  memovo-ai:0.1.0
```

The container runs as an unprivileged user (`memovo`, uid 1001) with a
read-only root filesystem and `no-new-privileges`.

## Probes

| Endpoint | Question | Wire it to |
|---|---|---|
| `GET /health` | Is the process alive? | liveness probe / `HEALTHCHECK` |
| `GET /ready` | Should it receive traffic? | readiness probe, load-balancer membership |

Never point a liveness probe at `/ready`. A failed model load leaves the
service alive but unready; restarting it produces an identical replacement,
so a liveness probe on readiness turns a degraded service into a crash loop.

`/ready` returns `503` with `{"status": "..."}` — `starting`, `loading` or
`unavailable` — until the services are built, then `200 {"status": "ready"}`.
Allow a generous start period: a cold CPU model load is slow.

## Scaling

**One worker per container.** The embedding model is loaded once per process
(doc 02 §7) and is over a gigabyte resident; a second worker doubles memory
for nothing. Scale with replicas.

The service is stateless — no database, no session, no local writes — so
replicas need no coordination.

## Configuration

Every setting is an environment variable; see `.env.example` for the full
list with defaults. The ones that matter in production:

| Variable | Notes |
|---|---|
| `MEMOVO_VECTOR_PROVIDER` | `atlas` in production. `fake` is an empty in-memory index and will answer "no memories found" for every query. |
| `MEMOVO_ATLAS_URI` | Carries credentials. Secret store only — never an image layer, never a committed file. |
| `MEMOVO_SEARCH_SIMILARITY_THRESHOLD` | Currently `0.75`, pending the product decision in contract §2.4. |
| `MEMOVO_LOG_FORMAT` | `json` for aggregation. |
| `MEMOVO_LOG_USER_SALT` | Optional. Makes the hashed `userId` in logs resistant to confirmation by guessing. |

`.env` is excluded from the build context (`.dockerignore`), so a developer's
credentials cannot be baked into an image by accident.

## Before production

Doc 06 §12's checklist, with current state:

| Item | State |
|---|---|
| AI endpoint is private | deployment-time; nothing in the service exposes it |
| User pre-filter verified against the production adapter | **pending a live cluster** |
| Vector provider has no write path | done — enforced and tested three ways |
| Public errors are sanitized | done |
| Model readiness works | done |
| Logging is privacy-safe | done |
| Configuration is externalized | done |
| Test / lint / type gates pass | done, in CI |
| Retrieval evaluation baseline exists | done — `eval/` |
