"""Liveness and readiness probes (doc 06, sections 8 and 9).

The distinction under test is the one that matters operationally: liveness
must not depend on the model, readiness must.
"""

from collections.abc import Iterator, Sequence

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from memovo_ai.api.dependencies import Services
from memovo_ai.core.readiness import ReadinessState, readiness_of, set_readiness
from memovo_ai.embeddings import EMBEDDING_DIMENSION, Embedding
from memovo_ai.main import create_app
from memovo_ai.providers.vector_search import FakeVectorSearchProvider
from memovo_ai.services import MemoryProcessorService, MemorySearchService

pytestmark = pytest.mark.integration

HEALTH = "/health"
READY = "/ready"

OK = 200
UNAVAILABLE = 503


class StubEmbeddings:
    async def embed_documents(self, texts: Sequence[str]) -> list[Embedding]:
        return [[0.1] * EMBEDDING_DIMENSION for _ in texts]

    async def embed_query(self, query: str) -> Embedding:
        return [0.1] * EMBEDDING_DIMENSION


def working_services() -> Services:
    embeddings = StubEmbeddings()
    vector_search = FakeVectorSearchProvider()

    return Services(
        memory_search=MemorySearchService(
            embedding_provider=embeddings,  # type: ignore[arg-type]
            vector_search=vector_search,
        ),
        memory_processor=MemoryProcessorService(embedding_provider=embeddings),  # type: ignore[arg-type]
        vector_search=vector_search,
    )


def ready_app(monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    """An application whose startup builds services successfully.

    Patched at the composition root rather than through
    ``dependency_overrides``: readiness reflects what *startup* achieved, and
    an override does not change that.
    """
    monkeypatch.setattr("memovo_ai.main.build_services", working_services)

    return create_app()


@pytest.fixture
def unavailable() -> Iterator[TestClient]:
    """Startup runs but builds nothing -- the failed-model-load shape."""
    with TestClient(create_app(load_services=False)) as client:
        yield client


# ---------------------------------------------------------------------------
# Liveness
# ---------------------------------------------------------------------------


def test_liveness_answers_while_the_process_serves(unavailable: TestClient) -> None:
    assert unavailable.get(HEALTH).status_code == OK


def test_liveness_does_not_depend_on_the_model(unavailable: TestClient) -> None:
    """The important one.

    If a failed model load made liveness fail, an orchestrator would kill the
    process and every replacement would fail identically -- a restart loop
    instead of a service that reports itself unready.
    """
    assert unavailable.get(READY).status_code == UNAVAILABLE
    assert unavailable.get(HEALTH).status_code == OK


def test_liveness_reports_a_fixed_status(unavailable: TestClient) -> None:
    assert unavailable.get(HEALTH).json() == {"status": "alive"}


def test_liveness_takes_no_input(unavailable: TestClient) -> None:
    """A probe must not become a second way into the service."""
    assert unavailable.post(HEALTH).status_code == 405


# ---------------------------------------------------------------------------
# Readiness
# ---------------------------------------------------------------------------


def test_readiness_is_refused_without_services(unavailable: TestClient) -> None:
    response = unavailable.get(READY)

    assert response.status_code == UNAVAILABLE
    assert response.json() == {"status": "unavailable"}


def test_readiness_succeeds_once_startup_built_the_services(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with TestClient(ready_app(monkeypatch)) as client:
        response = client.get(READY)

    assert response.status_code == OK
    assert response.json() == {"status": "ready"}


def test_readiness_is_withdrawn_at_shutdown(monkeypatch: pytest.MonkeyPatch) -> None:
    """A draining instance must stop attracting traffic."""
    application = ready_app(monkeypatch)

    with TestClient(application) as client:
        assert client.get(READY).status_code == OK

    assert readiness_of(application.state) is ReadinessState.UNAVAILABLE


def test_a_failed_startup_leaves_the_service_unready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exact path a missing model or a bad Atlas URI takes."""

    def explode() -> Services:
        message = "model weights are missing"
        raise ValueError(message)

    monkeypatch.setattr("memovo_ai.main.build_services", explode)

    with TestClient(create_app()) as client:
        assert client.get(HEALTH).status_code == OK
        assert client.get(READY).status_code == UNAVAILABLE


def test_readiness_before_startup_is_not_ready() -> None:
    """An application object that has never started must not claim ready."""
    assert readiness_of(create_app(load_services=False).state) is ReadinessState.STARTING


def test_an_unset_state_defaults_to_starting() -> None:
    """A blank state object must not read as ready by omission."""

    class Bare:
        pass

    assert readiness_of(Bare()) is ReadinessState.STARTING


def test_a_foreign_value_does_not_read_as_ready() -> None:
    """Only a real state counts -- a stray string must not pass for one."""

    class Bare:
        pass

    state = Bare()
    setattr(state, "memovo_readiness_state", "ready")  # noqa: B010 - the point

    assert readiness_of(state) is ReadinessState.STARTING


def test_the_state_round_trips() -> None:
    class Bare:
        pass

    state = Bare()
    set_readiness(state, ReadinessState.LOADING)

    assert readiness_of(state) is ReadinessState.LOADING


# ---------------------------------------------------------------------------
# The probes are operational, not part of the Backend contract
# ---------------------------------------------------------------------------


def test_the_probes_live_outside_the_ai_namespace() -> None:
    """So they can never be mistaken for a contract endpoint."""
    assert not HEALTH.startswith("/ai/")
    assert not READY.startswith("/ai/")


def test_the_service_publishes_exactly_the_expected_paths(
    unavailable: TestClient,
) -> None:
    """Catches an endpoint added by accident.

    The Backend contract is four endpoints (contract section 5); everything
    else must be a deliberate operational addition.
    """
    paths = set(unavailable.get("/openapi.json").json()["paths"])

    assert paths == {
        "/ai/memories/process",
        "/ai/memories/search",
        "/ai/chat/memories",
        "/ai/memories/prepare-note",
        HEALTH,
        READY,
    }


def test_the_probes_are_tagged_separately(unavailable: TestClient) -> None:
    schema = unavailable.get("/openapi.json").json()["paths"]

    assert schema[HEALTH]["get"]["tags"] == ["operations"]
    assert schema[READY]["get"]["tags"] == ["operations"]


@pytest.mark.parametrize("endpoint", [HEALTH, READY])
def test_a_probe_reveals_no_configuration(unavailable: TestClient, endpoint: str) -> None:
    """A probe is the most exposed surface; it must leak nothing.

    No model name, no provider, no database, no version, no path.
    """
    body = unavailable.get(endpoint).text.lower()

    for secret in ("qwen", "atlas", "mongo", "embedding", "model", "/", "password"):
        assert secret not in body


@pytest.mark.parametrize("endpoint", [HEALTH, READY])
def test_a_probe_returns_only_a_status_field(unavailable: TestClient, endpoint: str) -> None:
    assert set(unavailable.get(endpoint).json()) == {"status"}


@pytest.mark.parametrize("endpoint", [HEALTH, READY])
def test_a_probe_is_correlated_like_any_other_request(
    unavailable: TestClient, endpoint: str
) -> None:
    assert (
        unavailable.get(endpoint, headers={"X-Request-Id": "probe-1"}).headers["X-Request-Id"]
        == "probe-1"
    )
