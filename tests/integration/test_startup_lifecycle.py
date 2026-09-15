"""Cold-start lifecycle: HTTP serves before the model is built.

The deployment failure this pins: the embedding model is downloaded and
loaded on first start, which takes minutes, and Uvicorn does not bind its
socket until the lifespan has yielded. With the build inside the lifespan,
``/health`` was unreachable for the whole load and the platform's
healthcheck killed the deployment.

The build is stood in for by a callable that blocks its worker thread on a
``threading.Event``, so each phase is observed deterministically -- no
timing-based waits anywhere.
"""

import asyncio
import logging
import threading
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager

import httpx2 as httpx
import pytest
from fastapi import FastAPI

from memovo_ai.api.dependencies import Services
from memovo_ai.core.config import InvalidConfigurationError
from memovo_ai.core.readiness import ReadinessState, readiness_of
from memovo_ai.embeddings import EMBEDDING_DIMENSION, Embedding
from memovo_ai.embeddings.models import ModelUnavailableError
from memovo_ai.main import create_app, wait_for_initialization
from memovo_ai.providers.vector_search import FakeVectorSearchProvider
from memovo_ai.services import MemoryProcessorService, MemorySearchService

pytestmark = pytest.mark.integration

HEALTH = "/health"
READY = "/ready"
SEARCH = "/ai/memories/search"
SEARCH_BODY = {"userId": "user_1", "query": "anything"}

OK = 200
UNAVAILABLE = 503

#: Upper bound on how long a test waits for the worker thread to reach a
#: synchronization point. Only ever hit when the implementation is broken.
DEADLINE = 5.0


class StubEmbeddings:
    async def embed_documents(self, texts: Sequence[str]) -> list[Embedding]:
        return [[0.1] * EMBEDDING_DIMENSION for _ in texts]

    async def embed_query(self, query: str) -> Embedding:
        return [0.1] * EMBEDDING_DIMENSION


class ClosableVectorSearch(FakeVectorSearchProvider):
    """Counts how many times startup's teardown released it."""

    def __init__(self) -> None:
        super().__init__()
        self.closed = 0

    async def aclose(self) -> None:
        self.closed += 1


def working_services() -> tuple[Services, ClosableVectorSearch]:
    embeddings = StubEmbeddings()
    vector_search = ClosableVectorSearch()
    services = Services(
        memory_search=MemorySearchService(
            embedding_provider=embeddings,  # type: ignore[arg-type]
            vector_search=vector_search,
        ),
        memory_processor=MemoryProcessorService(embedding_provider=embeddings),  # type: ignore[arg-type]
        vector_search=vector_search,
    )
    return services, vector_search


class BlockingBuild:
    """Stands in for ``build_services``.

    Blocks the thread that calls it until ``release`` is set, then returns
    the prepared services or raises the prepared error. ``started`` lets a
    test know the build is genuinely in progress before it asserts anything
    about the state the service is in meanwhile.
    """

    def __init__(self, outcome: Services | BaseException) -> None:
        self.outcome = outcome
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls = 0

    def __call__(self) -> Services:
        self.calls += 1
        self.started.set()
        if not self.release.wait(timeout=DEADLINE):  # pragma: no cover - broken implementation
            message = "the initializer was never released"
            raise RuntimeError(message)
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome

    async def in_progress(self) -> bool:
        """True once the build has been entered on its worker thread."""
        return await asyncio.to_thread(self.started.wait, DEADLINE)


@asynccontextmanager
async def serving(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    """The real lifespan over an ASGI transport, like Uvicorn would run it.

    ``lifespan_context`` returning is the property under test: the moment it
    yields is the moment Uvicorn binds its socket and the platform can reach
    ``/health``.
    """
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://ai") as client:
            yield client


def events(caplog: pytest.LogCaptureFixture, name: str) -> list[dict[str, object]]:
    return [
        dict(fields)
        for item in caplog.records
        if (fields := getattr(item, "memovo_fields", None)) is not None
        and getattr(item, "memovo_event", None) == name
    ]


def installed_services(app: FastAPI) -> list[object | None]:
    return [
        getattr(app.state, name, None)
        for name in (
            "memory_search_service",
            "memory_processor_service",
            "generation_provider",
            "memory_chat_service",
            "note_preparation_service",
        )
    ]


@pytest.fixture
def logs(caplog: pytest.LogCaptureFixture) -> pytest.LogCaptureFixture:
    caplog.set_level(logging.DEBUG)
    return caplog


# ---------------------------------------------------------------------------
# The cold-start requirement
# ---------------------------------------------------------------------------


async def test_startup_completes_while_the_build_is_still_blocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole point: the lifespan yields before the model is built."""
    services, _ = working_services()
    build = BlockingBuild(services)
    monkeypatch.setattr("memovo_ai.main.build_services", build)
    app = create_app()

    async with serving(app):
        assert await build.in_progress()
        assert not build.release.is_set(), "startup must not have waited for the build"
        assert readiness_of(app.state) is ReadinessState.LOADING
        build.release.set()
        await wait_for_initialization(app)


async def test_liveness_answers_while_the_model_is_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    services, _ = working_services()
    build = BlockingBuild(services)
    monkeypatch.setattr("memovo_ai.main.build_services", build)
    app = create_app()

    async with serving(app) as client:
        assert await build.in_progress()

        health = await client.get(HEALTH)
        assert health.status_code == OK
        assert health.json() == {"status": "alive"}

        build.release.set()
        await wait_for_initialization(app)


async def test_readiness_reports_loading_while_the_model_is_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    services, _ = working_services()
    build = BlockingBuild(services)
    monkeypatch.setattr("memovo_ai.main.build_services", build)
    app = create_app()

    async with serving(app) as client:
        assert await build.in_progress()

        ready = await client.get(READY)
        assert ready.status_code == UNAVAILABLE
        assert ready.json() == {"status": "loading"}

        build.release.set()
        await wait_for_initialization(app)


async def test_ai_routes_are_unavailable_until_the_build_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The existing request-time mechanism: an unset service is MODEL_UNAVAILABLE."""
    services, _ = working_services()
    build = BlockingBuild(services)
    monkeypatch.setattr("memovo_ai.main.build_services", build)
    app = create_app()

    async with serving(app) as client:
        assert await build.in_progress()
        assert installed_services(app) == [None] * 5

        response = await client.post(SEARCH, json=SEARCH_BODY)
        assert response.status_code == UNAVAILABLE
        assert response.json()["error"]["code"] == "MODEL_UNAVAILABLE"

        build.release.set()
        await wait_for_initialization(app)


async def test_readiness_and_routes_follow_once_the_build_completes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    services, _ = working_services()
    build = BlockingBuild(services)
    monkeypatch.setattr("memovo_ai.main.build_services", build)
    app = create_app()

    async with serving(app) as client:
        assert await build.in_progress()
        build.release.set()
        await wait_for_initialization(app)

        ready = await client.get(READY)
        assert ready.status_code == OK
        assert ready.json() == {"status": "ready"}
        assert readiness_of(app.state) is ReadinessState.READY
        assert None not in installed_services(app)[:2]

        response = await client.post(SEARCH, json=SEARCH_BODY)
        assert response.status_code == OK
        assert response.json() == {"found": False, "results": []}


async def test_the_build_runs_exactly_once_and_is_closed_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    services, vector_search = working_services()
    build = BlockingBuild(services)
    monkeypatch.setattr("memovo_ai.main.build_services", build)
    app = create_app()

    async with serving(app):
        assert await build.in_progress()
        build.release.set()
        await wait_for_initialization(app)
        assert vector_search.closed == 0

    assert build.calls == 1
    assert vector_search.closed == 1
    assert installed_services(app) == [None] * 5
    assert readiness_of(app.state) is ReadinessState.UNAVAILABLE


async def test_started_is_logged_when_the_attempt_completes_not_when_http_starts(
    monkeypatch: pytest.MonkeyPatch, logs: pytest.LogCaptureFixture
) -> None:
    services, _ = working_services()
    build = BlockingBuild(services)
    monkeypatch.setattr("memovo_ai.main.build_services", build)
    app = create_app()

    async with serving(app):
        assert await build.in_progress()
        assert events(logs, "service.started") == []

        build.release.set()
        await wait_for_initialization(app)

        entry = events(logs, "service.started")[0]
        assert entry["services_available"] is True
        assert set(entry) == {
            "correlation_id",
            "services_available",
            "vector_provider",
            "generation_enabled",
            "generation_provider",
            "generation_available",
        }


# ---------------------------------------------------------------------------
# Tolerated failure: alive, unready, no crash
# ---------------------------------------------------------------------------


async def test_a_tolerated_build_failure_leaves_the_process_alive_and_unready(
    monkeypatch: pytest.MonkeyPatch, logs: pytest.LogCaptureFixture
) -> None:
    build = BlockingBuild(ModelUnavailableError("model weights are missing"))
    monkeypatch.setattr("memovo_ai.main.build_services", build)
    app = create_app()

    async with serving(app) as client:
        assert await build.in_progress()
        build.release.set()
        await wait_for_initialization(app)

        assert (await client.get(HEALTH)).status_code == OK
        ready = await client.get(READY)
        assert ready.status_code == UNAVAILABLE
        assert ready.json() == {"status": "unavailable"}

        response = await client.post(SEARCH, json=SEARCH_BODY)
        assert response.status_code == UNAVAILABLE
        assert response.json()["error"]["code"] == "MODEL_UNAVAILABLE"

    assert events(logs, "service.startup_failed")[0]["error_type"] == "ModelUnavailableError"
    assert events(logs, "service.started")[0]["services_available"] is False


async def test_a_tolerated_build_failure_is_not_an_unhandled_task_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The failure is handled inside the task; nothing is left for asyncio
    to complain about at garbage collection."""
    build = BlockingBuild(ModelUnavailableError("model weights are missing"))
    monkeypatch.setattr("memovo_ai.main.build_services", build)
    app = create_app()

    async with serving(app):
        assert await build.in_progress()
        build.release.set()
        await wait_for_initialization(app)

        task = app.state.memovo_initialization.task
        assert task is not None
        assert task.done()
        assert task.exception() is None


# ---------------------------------------------------------------------------
# Fatal configuration: rejected before any build is scheduled
# ---------------------------------------------------------------------------


async def test_invalid_production_configuration_fails_startup_before_any_build(
    monkeypatch: pytest.MonkeyPatch, logs: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("MEMOVO_ENV", "production")
    monkeypatch.setenv("MEMOVO_VECTOR_PROVIDER", "fake")
    services, _ = working_services()
    build = BlockingBuild(services)
    monkeypatch.setattr("memovo_ai.main.build_services", build)
    app = create_app()

    with pytest.raises(InvalidConfigurationError):
        async with app.router.lifespan_context(app):
            pass  # pragma: no cover - startup raises before the body runs

    assert build.calls == 0
    assert readiness_of(app.state) is ReadinessState.UNAVAILABLE
    rejected = events(logs, "service.startup_rejected")
    assert rejected[0]["error_type"] == "InvalidConfigurationError"
    assert events(logs, "service.started") == []


# ---------------------------------------------------------------------------
# Shutdown while the build is still running
# ---------------------------------------------------------------------------


async def test_shutdown_during_the_build_never_installs_the_late_result(
    monkeypatch: pytest.MonkeyPatch, logs: pytest.LogCaptureFixture
) -> None:
    """A build that finishes after shutdown began is released, not served."""
    services, vector_search = working_services()
    build = BlockingBuild(services)
    monkeypatch.setattr("memovo_ai.main.build_services", build)
    app = create_app()

    lifespan = app.router.lifespan_context(app)
    await lifespan.__aenter__()
    assert await build.in_progress()

    shutdown = asyncio.create_task(lifespan.__aexit__(None, None, None))
    # One scheduling turn: enough for the shutdown task to run up to its
    # first suspension, which is after it has withdrawn readiness. Not a
    # timing wait -- the loop is single-threaded and the build is blocked.
    await asyncio.sleep(0)
    assert readiness_of(app.state) is ReadinessState.UNAVAILABLE
    assert not shutdown.done()

    build.release.set()
    await shutdown

    assert installed_services(app) == [None] * 5
    assert vector_search.closed == 1
    assert readiness_of(app.state) is ReadinessState.UNAVAILABLE
    assert events(logs, "service.started") == []
    assert events(logs, "service.stopped")


async def test_waiting_for_initialization_is_a_no_op_without_a_build() -> None:
    """``load_services=False`` builds nothing; the wait must not hang."""
    app = create_app(load_services=False)

    async with serving(app) as client:
        await wait_for_initialization(app)

        assert (await client.get(READY)).json() == {"status": "unavailable"}
