"""Startup configuration safety for the handoff.

``fake`` is the development default vector provider, and it is an empty
in-memory index. An *unset* ``MEMOVO_VECTOR_PROVIDER`` therefore selects it
silently -- including when the variable NAME was mistyped, which leaves the
real one unset. In production that would start a service that loads its
model, reports itself ready, answers 200, and returns "no memories found" for
every query: a deployment that looks healthy from every angle and serves
nothing.

A mistyped provider *value* is a different and safer failure -- it matches no
known provider and raises ``ValueError`` at startup. Both are covered below.
"""

import json
import logging
from collections.abc import Callable, Iterator, Sequence

import pytest
from fastapi.testclient import TestClient

from memovo_ai.api.dependencies import Services, build_services, build_vector_search_provider
from memovo_ai.core.config import (
    ATLAS_VECTOR_PROVIDER,
    FAKE_VECTOR_PROVIDER,
    AtlasSettings,
    InvalidConfigurationError,
    ProviderSettings,
    RuntimeSettings,
)
from memovo_ai.core.logging import JsonFormatter
from memovo_ai.core.readiness import ReadinessState, readiness_of
from memovo_ai.embeddings import EMBEDDING_DIMENSION, Embedding
from memovo_ai.embeddings.models import ModelUnavailableError
from memovo_ai.main import create_app
from memovo_ai.providers.vector_search import FakeVectorSearchProvider
from memovo_ai.services import MemoryProcessorService, MemorySearchService

pytestmark = pytest.mark.integration

READY = "/ready"
OK = 200
UNAVAILABLE = 503

#: A connection string shaped like a real one, used only to prove it never
#: reaches a log line.
ATLAS_URI = "mongodb+srv://admin:hunter2@prod-cluster.example.mongodb.net/"


def provider_settings(name: str) -> ProviderSettings:
    return ProviderSettings(_env_file=None, vector_provider=name)  # type: ignore[call-arg]


def runtime(env: str) -> RuntimeSettings:
    return RuntimeSettings(_env_file=None, env=env)  # type: ignore[call-arg]


class StubEmbeddings:
    async def embed_documents(self, texts: Sequence[str]) -> list[Embedding]:
        return [[0.1] * EMBEDDING_DIMENSION for _ in texts]

    async def embed_query(self, query: str) -> Embedding:
        return [0.1] * EMBEDDING_DIMENSION


def working_services() -> Services:
    """What a successful startup would produce, without loading a model."""
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


@pytest.fixture
def logs(caplog: pytest.LogCaptureFixture) -> Iterator[pytest.LogCaptureFixture]:
    with caplog.at_level(logging.DEBUG):
        yield caplog


def events(caplog: pytest.LogCaptureFixture, name: str) -> list[dict[str, object]]:
    return [
        dict(fields)
        for item in caplog.records
        if (fields := getattr(item, "memovo_fields", None)) is not None
        and getattr(item, "memovo_event", None) == name
    ]


def rendered(caplog: pytest.LogCaptureFixture) -> str:
    formatter = JsonFormatter()

    return "\n".join(formatter.format(item) for item in caplog.records)


def production_with_the_fake_provider(monkeypatch: pytest.MonkeyPatch) -> list[bool]:
    """The configuration startup must refuse, plus a build that records
    whether it was ever reached. Returns that record."""
    monkeypatch.setenv("MEMOVO_ENV", "production")
    monkeypatch.setenv("MEMOVO_VECTOR_PROVIDER", FAKE_VECTOR_PROVIDER)
    built: list[bool] = []

    def must_not_build() -> Services:
        built.append(True)
        return working_services()  # pragma: no cover - reaching here is the failure

    monkeypatch.setattr("memovo_ai.main.build_services", must_not_build)

    return built


# ---------------------------------------------------------------------------
# 1. production + fake -> rejected
# ---------------------------------------------------------------------------


def test_the_fake_provider_is_refused_in_production() -> None:
    with pytest.raises(InvalidConfigurationError):
        build_vector_search_provider(provider_settings(FAKE_VECTOR_PROVIDER), runtime("production"))


def test_the_refusal_names_the_variable_to_set() -> None:
    """An operator must be able to act on the message without the source."""
    with pytest.raises(InvalidConfigurationError) as caught:
        build_vector_search_provider(provider_settings(FAKE_VECTOR_PROVIDER), runtime("production"))

    message = str(caught.value)

    assert "MEMOVO_VECTOR_PROVIDER" in message
    assert "MEMOVO_ENV=production" in message
    assert ATLAS_VECTOR_PROVIDER in message


def test_the_environment_check_is_not_case_sensitive() -> None:
    """``MEMOVO_ENV=Production`` must not slip past the guard."""
    for spelling in ("Production", "PRODUCTION", "  production  "):
        with pytest.raises(InvalidConfigurationError):
            build_vector_search_provider(provider_settings(FAKE_VECTOR_PROVIDER), runtime(spelling))


def test_building_every_service_is_refused_in_production(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The guard sits on the path startup actually takes."""
    monkeypatch.setattr(
        "memovo_ai.api.dependencies.build_embedding_provider", lambda *_: StubEmbeddings()
    )

    with pytest.raises(InvalidConfigurationError):
        build_services(
            provider_settings=provider_settings(FAKE_VECTOR_PROVIDER),
            runtime_settings=runtime("production"),
        )


def test_the_configuration_is_rejected_before_the_model_is_loaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ordering, and it matters twice over.

    An operator should learn a setting is wrong in the first second, not
    after a gigabyte of weights has been read. And when *both* are wrong --
    which is the realistic case on a fresh deployment -- a model error raised
    first would be tolerated, the service would start unready, and the
    misconfiguration would never be reported at all.
    """
    loaded: list[bool] = []

    def must_not_run(*_: object) -> StubEmbeddings:
        loaded.append(True)
        return StubEmbeddings()  # pragma: no cover - reaching here is the failure

    monkeypatch.setattr("memovo_ai.api.dependencies.build_embedding_provider", must_not_run)

    with pytest.raises(InvalidConfigurationError):
        build_services(
            provider_settings=provider_settings(FAKE_VECTOR_PROVIDER),
            runtime_settings=runtime("production"),
        )

    assert loaded == []


def test_a_misconfiguration_is_reported_even_when_the_model_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The case that first exposed the ordering bug.

    With no model installed, ``ModelUnavailableError`` used to be raised
    first and tolerated -- so a production deployment with the fake provider
    started unready and said nothing about the real problem.
    """

    def no_model(*_: object) -> StubEmbeddings:
        message = "model weights are missing"
        raise ModelUnavailableError(message)

    monkeypatch.setattr("memovo_ai.api.dependencies.build_embedding_provider", no_model)

    with pytest.raises(InvalidConfigurationError):
        build_services(
            provider_settings=provider_settings(FAKE_VECTOR_PROVIDER),
            runtime_settings=runtime("production"),
        )


def test_startup_fails_rather_than_starting_unready(monkeypatch: pytest.MonkeyPatch) -> None:
    """A wrong setting cannot fix itself, so it is not tolerated.

    Contrast ``test_a_failed_build_leaves_the_service_unready`` in
    test_health.py: a missing model *is* tolerated, because it may recover
    and because a crash loop tells an operator less. A misconfiguration is
    the opposite case.
    """
    production_with_the_fake_provider(monkeypatch)

    with pytest.raises(InvalidConfigurationError), TestClient(create_app()):
        pass  # pragma: no cover - startup raises before the body runs


def test_a_rejected_startup_never_schedules_the_build(monkeypatch: pytest.MonkeyPatch) -> None:
    """The build runs in the background now, so the refusal has to happen
    before it is scheduled -- otherwise the model would load for a process
    that was never allowed to serve."""
    built = production_with_the_fake_provider(monkeypatch)

    with pytest.raises(InvalidConfigurationError), TestClient(create_app()):
        pass  # pragma: no cover - startup raises before the body runs

    assert built == []


def test_readiness_never_becomes_ready_when_startup_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    production_with_the_fake_provider(monkeypatch)
    application = create_app()

    with pytest.raises(InvalidConfigurationError), TestClient(application):
        pass  # pragma: no cover - startup raises before the body runs

    assert readiness_of(application.state) is not ReadinessState.READY
    assert readiness_of(application.state) is ReadinessState.UNAVAILABLE


def test_the_rejection_is_logged_as_critical(
    monkeypatch: pytest.MonkeyPatch, logs: pytest.LogCaptureFixture
) -> None:
    production_with_the_fake_provider(monkeypatch)

    with pytest.raises(InvalidConfigurationError), TestClient(create_app()):
        pass  # pragma: no cover - startup raises before the body runs

    entry = events(logs, "service.startup_rejected")[0]

    assert entry["error_type"] == "InvalidConfigurationError"
    assert [
        record.levelno
        for record in logs.records
        if getattr(record, "memovo_event", None) == "service.startup_rejected"
    ] == [logging.CRITICAL]


def test_a_rejected_startup_never_reports_started(
    monkeypatch: pytest.MonkeyPatch, logs: pytest.LogCaptureFixture
) -> None:
    production_with_the_fake_provider(monkeypatch)

    with pytest.raises(InvalidConfigurationError), TestClient(create_app()):
        pass  # pragma: no cover - startup raises before the body runs

    assert events(logs, "service.started") == []
    assert events(logs, "service.initialization_started") == []


# ---------------------------------------------------------------------------
# 2. non-production + fake -> still allowed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("environment", ["development", "test", "staging", ""])
def test_the_fake_provider_is_allowed_outside_production(environment: str) -> None:
    """Tests and local work depend on it; nothing here removes it."""
    provider = build_vector_search_provider(
        provider_settings(FAKE_VECTOR_PROVIDER), runtime(environment)
    )

    assert isinstance(provider, FakeVectorSearchProvider)


def test_the_default_environment_is_not_production() -> None:
    """So a developer who sets nothing keeps the working default."""
    assert RuntimeSettings(_env_file=None).is_production is False  # type: ignore[call-arg]


def test_the_fake_provider_remains_the_default_provider() -> None:
    """A locked default; changing it would break local development."""
    assert ProviderSettings(_env_file=None).vector_provider == FAKE_VECTOR_PROVIDER  # type: ignore[call-arg]


def test_an_unknown_provider_value_still_raises_value_error() -> None:
    """A mistyped *value*, as opposed to a mistyped variable name.

    This one was never the dangerous case: it matches no provider and fails
    loudly in every environment. The guard above exists for the other one,
    where the variable ends up unset and ``fake`` is chosen silently.
    """
    with pytest.raises(ValueError, match="unknown vector provider"):
        build_vector_search_provider(provider_settings("elasticsearch"), runtime("development"))


def test_a_tolerated_failure_still_starts_unready(
    monkeypatch: pytest.MonkeyPatch, settled: Callable[[TestClient], None]
) -> None:
    """The new clause must not have widened what startup refuses."""

    def unavailable() -> Services:
        message = "model weights are missing"
        raise ValueError(message)

    monkeypatch.setattr("memovo_ai.main.build_services", unavailable)

    with TestClient(create_app()) as client:
        settled(client)
        assert client.get(READY).status_code == UNAVAILABLE


# ---------------------------------------------------------------------------
# 2b. production + atlas + no URI -- documented, deliberately unchanged
# ---------------------------------------------------------------------------


def test_a_missing_atlas_uri_is_tolerated_rather_than_fatal(
    monkeypatch: pytest.MonkeyPatch, settled: Callable[[TestClient], None]
) -> None:
    """Unlike the fake provider, this is **not** a startup rejection.

    Deliberately preserved. Unlike ``fake``, this configuration cannot serve
    anything silently: ``/ready`` stays 503 so the instance takes no traffic,
    and every request fails. Making it fatal is a separate decision.
    """
    monkeypatch.setenv("MEMOVO_VECTOR_PROVIDER", ATLAS_VECTOR_PROVIDER)
    monkeypatch.setenv("MEMOVO_ATLAS_URI", "")
    monkeypatch.setenv("MEMOVO_ENV", "production")
    monkeypatch.setattr(
        "memovo_ai.api.dependencies.build_embedding_provider", lambda *_: StubEmbeddings()
    )

    with TestClient(create_app()) as client:
        settled(client)
        assert client.get(READY).status_code == UNAVAILABLE
        assert client.get("/health").status_code == OK


def test_a_missing_atlas_uri_reports_model_unavailable_not_vector_search_failed(
    monkeypatch: pytest.MonkeyPatch, settled: Callable[[TestClient], None]
) -> None:
    """The actual behaviour, pinned because it is **misleading**.

    Startup logs ``VectorSearchUnavailableError``, but the request-level code
    is ``MODEL_UNAVAILABLE`` -- "Embedding model is temporarily unavailable"
    -- even though the model loaded fine. ``build_services`` is
    all-or-nothing: any tolerated failure leaves *both* services unset, and
    the request dependencies have exactly one failure mode.

    Contract section 12.3 documents this so the Backend does not spend a day
    debugging a model that was never the problem. Changing the code here
    would mean per-service degradation, which is out of scope.
    """
    monkeypatch.setenv("MEMOVO_VECTOR_PROVIDER", ATLAS_VECTOR_PROVIDER)
    monkeypatch.setenv("MEMOVO_ATLAS_URI", "")
    monkeypatch.setenv("MEMOVO_ENV", "production")
    monkeypatch.setattr(
        "memovo_ai.api.dependencies.build_embedding_provider", lambda *_: StubEmbeddings()
    )

    with TestClient(create_app(), raise_server_exceptions=False) as client:
        settled(client)
        response = client.post("/ai/memories/search", json={"userId": "u1", "query": "anything"})

    assert response.status_code == UNAVAILABLE
    assert response.json()["error"]["code"] == "MODEL_UNAVAILABLE"


def test_the_startup_log_names_the_real_cause_of_a_missing_uri(
    monkeypatch: pytest.MonkeyPatch,
    logs: pytest.LogCaptureFixture,
    settled: Callable[[TestClient], None],
) -> None:
    """The log is accurate even though the response code is not.

    This is what an operator has to go on, so it must keep naming the vector
    failure rather than the model.
    """
    monkeypatch.setenv("MEMOVO_VECTOR_PROVIDER", ATLAS_VECTOR_PROVIDER)
    monkeypatch.setenv("MEMOVO_ATLAS_URI", "")
    monkeypatch.setenv("MEMOVO_ENV", "production")
    monkeypatch.setattr(
        "memovo_ai.api.dependencies.build_embedding_provider", lambda *_: StubEmbeddings()
    )

    with TestClient(create_app()) as client:
        settled(client)

    assert events(logs, "service.startup_failed")[0]["error_type"] == (
        "VectorSearchUnavailableError"
    )
    assert events(logs, "service.started")[0]["vector_provider"] == ATLAS_VECTOR_PROVIDER


# ---------------------------------------------------------------------------
# 3. the active provider appears in service.started
# ---------------------------------------------------------------------------


def test_the_configured_provider_is_logged_at_startup(
    monkeypatch: pytest.MonkeyPatch,
    logs: pytest.LogCaptureFixture,
    settled: Callable[[TestClient], None],
) -> None:
    monkeypatch.setenv("MEMOVO_VECTOR_PROVIDER", ATLAS_VECTOR_PROVIDER)
    monkeypatch.setattr("memovo_ai.main.build_services", working_services)

    with TestClient(create_app()) as client:
        settled(client)
        assert client.get(READY).status_code == OK

    assert events(logs, "service.started")[0]["vector_provider"] == ATLAS_VECTOR_PROVIDER


def test_the_provider_is_logged_even_when_startup_degraded(
    monkeypatch: pytest.MonkeyPatch,
    logs: pytest.LogCaptureFixture,
    settled: Callable[[TestClient], None],
) -> None:
    """The case an operator most needs it: something went wrong."""
    monkeypatch.setenv("MEMOVO_VECTOR_PROVIDER", ATLAS_VECTOR_PROVIDER)

    def unavailable() -> Services:
        message = "model weights are missing"
        raise ValueError(message)

    monkeypatch.setattr("memovo_ai.main.build_services", unavailable)

    with TestClient(create_app()) as client:
        settled(client)

    entry = events(logs, "service.started")[0]

    assert entry["vector_provider"] == ATLAS_VECTOR_PROVIDER
    assert entry["services_available"] is False


def test_the_fake_provider_is_visible_in_the_log(logs: pytest.LogCaptureFixture) -> None:
    """So "why does search return nothing" is answerable from one log line."""
    with TestClient(create_app(load_services=False)):
        pass

    assert events(logs, "service.started")[0]["vector_provider"] == FAKE_VECTOR_PROVIDER


# ---------------------------------------------------------------------------
# 4. the provider name is the only thing that leaks
# ---------------------------------------------------------------------------


def test_the_atlas_connection_string_never_reaches_a_log(
    monkeypatch: pytest.MonkeyPatch,
    logs: pytest.LogCaptureFixture,
    settled: Callable[[TestClient], None],
) -> None:
    """Logging the provider name must not drag its configuration along."""
    monkeypatch.setenv("MEMOVO_VECTOR_PROVIDER", ATLAS_VECTOR_PROVIDER)
    monkeypatch.setenv("MEMOVO_ATLAS_URI", ATLAS_URI)
    monkeypatch.setattr("memovo_ai.main.build_services", working_services)

    with TestClient(create_app()) as client:
        settled(client)

    output = rendered(logs)

    assert ATLAS_VECTOR_PROVIDER in output
    assert ATLAS_URI not in output
    assert "hunter2" not in output
    assert "prod-cluster.example" not in output


def test_the_startup_log_carries_only_operational_fields(
    logs: pytest.LogCaptureFixture,
) -> None:
    with TestClient(create_app(load_services=False)):
        pass

    assert set(events(logs, "service.started")[0]) == {
        "correlation_id",
        "services_available",
        "vector_provider",
        "generation_enabled",
        "generation_provider",
        "generation_available",
    }


def test_a_rejected_startup_logs_no_connection_string(
    monkeypatch: pytest.MonkeyPatch, logs: pytest.LogCaptureFixture
) -> None:
    """The refusal path must be as careful as the success path."""
    monkeypatch.setenv("MEMOVO_ATLAS_URI", ATLAS_URI)
    production_with_the_fake_provider(monkeypatch)

    with pytest.raises(InvalidConfigurationError), TestClient(create_app()):
        pass  # pragma: no cover - startup raises before the body runs

    output = rendered(logs)

    assert "hunter2" not in output
    assert ATLAS_URI not in output


def test_the_refusal_message_itself_carries_no_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """It names variables, never their values."""
    monkeypatch.setenv("MEMOVO_ATLAS_URI", ATLAS_URI)

    with pytest.raises(InvalidConfigurationError) as caught:
        build_vector_search_provider(provider_settings(FAKE_VECTOR_PROVIDER), runtime("production"))

    assert "hunter2" not in str(caught.value)
    assert ATLAS_URI not in str(caught.value)


def test_the_atlas_uri_is_not_rendered_by_its_settings_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The property the log line depends on, asserted directly."""
    monkeypatch.setenv("MEMOVO_ATLAS_URI", ATLAS_URI)
    settings = AtlasSettings()

    assert "hunter2" not in repr(settings)
    assert "hunter2" not in json.dumps(settings.model_dump(mode="json"))
