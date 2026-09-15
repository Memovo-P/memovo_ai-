# Memovo AI Service

Memory processing, retrieval, grounded Memory Chat and explicit note preparation for Memovo.

The canonical wire contract is [docs/AI_CONTRACT_FINAL.md](docs/AI_CONTRACT_FINAL.md) (v1.9 plus the
addendum in its section 21.9). The implementation plan and current status are in
[docs/SPRINT2_GENERATION_PLAN.md](docs/SPRINT2_GENERATION_PLAN.md); the deployment and integration
guide, including what is and is not verified, is [docs/deployment.md](docs/deployment.md).

## Endpoints

| Endpoint | Purpose |
|---|---|
| `POST /ai/memories/process` | Chunk a Note or Link and return the complete chunk set with 1024-d embeddings |
| `POST /ai/memories/search` | Embed a query and retrieve the user's matching memories |
| `POST /ai/chat/memories` | Answer a question grounded only in the user's retrieved memories |
| `POST /ai/memories/prepare-note` | Turn raw content into a title and cleaned note content |
| `GET /health`, `GET /ready` | Liveness and readiness probes (outside the contract) |

Out of scope by contract: general chat, intent detection, agents and tools, memory mutation,
authentication, URL fetching, and any database write.

## Responsibility boundary

This service owns canonical content, chunking, embeddings, read-only user-scoped vector
retrieval, answer generation and note preparation. The Backend owns authentication and the
trusted `userId`, authorization, persistence, vector writes/reconciliation/cleanup, link
extraction, conversation history, queues and retries.

**This service never writes to MongoDB or the vector index, never fetches a URL, and never
retries an upstream call.**

## Requirements

- Python 3.12 (pinned in `.python-version`)
- [`uv`](https://docs.astral.sh/uv/)

## Setup

```bash
uv sync
cp .env.example .env
```

The default install runs the whole suite with deterministic fakes: no model weights, no
cluster, no network. Optional extras:

```bash
uv sync --extra embeddings    # sentence-transformers (pulls torch) for the real Qwen3 embedder
uv sync --extra atlas         # pymongo for MongoDB Atlas Vector Search
```

Opt-in real dependencies, never run by default:

```bash
MEMOVO_RUN_EMBEDDING_INTEGRATION=1 uv run pytest tests/integration/test_embedding_model.py -s
MEMOVO_RUN_OPENROUTER_SMOKE=1     uv run pytest tests/integration/test_openrouter_live.py -s
```

## Quality gate

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src
```

## Running the service

```bash
uv run uvicorn memovo_ai.main:app --reload
```

Generation is disabled by default (`MEMOVO_GENERATION_ENABLED=false`); processing and search never
need it. When enabled, the only provider is OpenRouter and the model is exactly the approved
`nvidia/nemotron-3-super-120b-a12b:free`; the key lives in the local `.env` or the deployment secret store. See
[docs/deployment.md](docs/deployment.md) for every setting, the error table, the migration notes
and the current verification status.

## Evaluation

`eval/` holds the retrieval quality dataset and threshold sweep, and the generation datasets
(fixed-evidence chat, note preparation) with their harness. See [eval/README.md](eval/README.md).

## Docker

```bash
docker build -t memovo-ai:<release> .   # embedding weights are downloaded on first start
docker compose up --build
```

## Status

Gates pass locally. The CI workflow in `.github/workflows/ci.yml` has never executed because
there is no git remote. Live Atlas and the real generation model have not been exercised; the
reasons and the exact commands to do so are in [docs/deployment.md](docs/deployment.md).
