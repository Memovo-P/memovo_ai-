# Memovo AI Service (doc 03, Phase 23; doc 06, sections 7-8).
#
# The image holds exactly what doc 06 section 8 lists: the FastAPI
# application, the Qwen embedding runtime and its tokenizer, the read-only
# vector adapter, configuration, and the health/readiness endpoints. No queue
# orchestration -- that is the Backend's, and bundling it here would put a
# Backend responsibility inside the AI boundary.
#
# Two stages so the runtime image carries no build tooling, no compiler and
# no uv binary. Only the virtualenv crosses the boundary.

# ---------------------------------------------------------------------------
# Stage 1 - build the virtualenv
# ---------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS builder

COPY --from=ghcr.io/astral-sh/uv:0.11.28 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_NO_CACHE=1

# Same path as the runtime stage, deliberately. A virtualenv is not
# relocatable: every console script in bin/ carries an absolute shebang, so
# building at /build and copying to /app would leave `uvicorn` pointing at a
# python that does not exist in the final image.
WORKDIR /app

# Dependencies resolve from the lockfile before the source is copied, so a
# code change does not re-resolve or re-download torch.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-editable --no-install-project \
        --extra embeddings --extra atlas

COPY src ./src
# --no-editable installs the package into site-packages instead of leaving a
# .pth file that points back at ./src. Without it the runtime image would
# have to carry the source too, and would import it through a path that only
# exists during the build.
RUN uv sync --frozen --no-dev --no-editable --extra embeddings --extra atlas

# ---------------------------------------------------------------------------
# Stage 2 - runtime
# ---------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS runtime

# Unprivileged. The service reads a vector index and answers HTTP; it needs
# write access to nothing but its own model cache.
RUN groupadd --system --gid 1001 memovo \
 && useradd --system --uid 1001 --gid memovo --create-home memovo

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Model weights live under a path the service user owns. The default
    # cache is /root/.cache, which an unprivileged process cannot write, so
    # the failure would appear on first load rather than at build time.
    HF_HOME=/home/memovo/.cache/huggingface \
    # Offline by default: every weight the service needs is baked in below.
    # A silent download on a cache miss would make startup depend on an
    # external service and hide a missing bake until production.
    HF_HUB_OFFLINE=1 \
    # Otherwise every start logs a symlink warning.
    HF_HUB_DISABLE_SYMLINKS_WARNING=1

WORKDIR /app

# The virtualenv is the whole application: --no-editable above put the
# package inside it, so no source directory is needed here.
COPY --from=builder --chown=memovo:memovo /app/.venv /app/.venv

USER memovo

# Bake the weights. Startup then loads from disk: deterministic, offline, and
# identical on every replica. Set --build-arg BAKE_MODEL=false for a thin
# image that downloads on first start -- acceptable for local work, not for
# production, where a Hugging Face outage would become a failed rollout.
ARG BAKE_MODEL=true
ARG MEMOVO_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B
RUN if [ "$BAKE_MODEL" = "true" ]; then \
        HF_HUB_OFFLINE=0 python -c "\
import sys; \
from sentence_transformers import SentenceTransformer; \
SentenceTransformer(sys.argv[1]); \
print('baked', sys.argv[1])" "$MEMOVO_EMBEDDING_MODEL"; \
    fi

ENV MEMOVO_EMBEDDING_MODEL=${MEMOVO_EMBEDDING_MODEL}

EXPOSE 8000

# Liveness only. Readiness is /ready and belongs to the orchestrator, which
# can act on it -- Docker's HEALTHCHECK cannot take an instance out of a load
# balancer, and wiring readiness here would only restart a container that is
# loading a model perfectly normally.
HEALTHCHECK --interval=30s --timeout=3s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health').read()"

# One worker per container. The embedding model is loaded once per process
# (doc 02, section 7) and is over a gigabyte resident, so a second worker
# doubles the memory for no shared benefit. Scale with replicas instead.
CMD ["uvicorn", "memovo_ai.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--no-access-log"]
