"""Generation configuration at startup and its failures on the wire.

No network. The real adapter is only *constructed* (with a key that is never
sent anywhere), and failures are driven through a service double so the
transport mapping -- status, code, ``retryable``, ``Retry-After`` -- is
checked end to end on the existing search route.
"""

import logging
from collections.abc import Iterator, Sequence

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr

from memovo_ai.api.dependencies import (
    build_generation_provider,
    build_services,
    get_memory_search_service,
)
from memovo_ai.core.config import (
    DEFAULT_GENERATION_MODEL,
    DEFAULT_OPENROUTER_BASE_URL,
    GenerationSettings,
    OpenRouterSettings,
)
from memovo_ai.core.logging import JsonFormatter
from memovo_ai.generation import (
    BoundedGenerationProvider,
    GenerationRateLimitedError,
    GenerationTimeoutError,
    GenerationUnavailableError,
    InvalidGenerationOutputError,
)
from memovo_ai.main import create_app

pytestmark = pytest.mark.integration

SEARCH = "/ai/memories/search"
BODY = {"userId": "user_123", "query": "anything"}
KEY = "sk-or-v1-ZZQ-runtime-test-key"


def generation(**overrides: object) -> GenerationSettings:
    return GenerationSettings(_env_file=None, **overrides)  # type: ignore[arg-type]


def openrouter(**overrides: object) -> OpenRouterSettings:
    return OpenRouterSettings(_env_file=None, **overrides)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------
def test_generation_is_disabled_by_default() -> None:
    assert generation().enabled is False


def test_the_approved_model_and_provider_are_the_defaults() -> None:
    settings = generation()

    assert settings.provider == "openrouter"
    assert settings.model == "nvidia/nemotron-3-super-120b-a12b:free"
    assert DEFAULT_GENERATION_MODEL == "nvidia/nemotron-3-super-120b-a12b:free"
    assert openrouter().base_url == DEFAULT_OPENROUTER_BASE_URL == "https://openrouter.ai/api/v1"


def test_the_prepared_environment_names_are_read(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEMOVO_GENERATION_ENABLED", "true")
    monkeypatch.setenv("MEMOVO_GENERATION_PROVIDER", "openrouter")
    monkeypatch.setenv("MEMOVO_GENERATION_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")
    monkeypatch.setenv("MEMOVO_OPENROUTER_BASE_URL", "https://router.example/v1")
    monkeypatch.setenv("MEMOVO_OPENROUTER_API_KEY", KEY)

    settings = generation()
    credentials = openrouter()

    assert settings.enabled is True
    assert settings.model == "nvidia/nemotron-3-super-120b-a12b:free"
    assert credentials.base_url == "https://router.example/v1"
    assert credentials.is_configured is True
    assert credentials.api_key.get_secret_value() == KEY


def test_the_key_is_a_secret() -> None:
    credentials = openrouter(api_key=SecretStr(KEY))

    assert KEY not in repr(credentials)
    assert KEY not in str(credentials.model_dump())


def test_a_blank_key_is_not_configured() -> None:
    assert openrouter(api_key=SecretStr("   ")).is_configured is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("timeout_seconds", 0),
        ("max_concurrency", 0),
        ("chat_max_output_tokens", 0),
        ("note_max_output_tokens", 0),
        ("context_window_tokens", 0),
        ("chars_per_token", 0),
        ("retry_after_max_seconds", -1),
    ],
)
def test_out_of_range_limits_are_rejected(field: str, value: int) -> None:
    with pytest.raises(ValueError):
        generation(**{field: value})


# --------------------------------------------------------------------------
# Building the provider
# --------------------------------------------------------------------------
def test_disabled_generation_builds_nothing_and_needs_no_key() -> None:
    assert build_generation_provider(generation(enabled=False), openrouter()) is None


def test_enabled_generation_without_a_key_is_refused() -> None:
    with pytest.raises(GenerationUnavailableError):
        build_generation_provider(generation(enabled=True), openrouter())


def test_an_unknown_provider_name_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown generation provider"):
        build_generation_provider(
            generation(enabled=True, provider="fake"), openrouter(api_key=SecretStr(KEY))
        )


def test_there_is_no_configurable_fake_provider() -> None:
    """Fake generation is constructed in code only; production cannot pick it."""
    for name in ("fake", "stub", "mock", "test"):
        with pytest.raises(ValueError, match="unknown generation provider"):
            build_generation_provider(
                generation(enabled=True, provider=name), openrouter(api_key=SecretStr(KEY))
            )


def test_enabled_generation_with_a_key_builds_the_bounded_openrouter_adapter() -> None:
    provider = build_generation_provider(
        generation(enabled=True, max_concurrency=2, timeout_seconds=3),
        openrouter(api_key=SecretStr(KEY)),
    )

    assert isinstance(provider, BoundedGenerationProvider)


def test_processing_and_search_are_built_without_generation(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Enabled but unusable generation degrades; it never blocks the rest."""
    from memovo_ai.api import dependencies
    from memovo_ai.embeddings import EMBEDDING_DIMENSION

    class StubEmbeddings:
        async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
            return [[0.1] * EMBEDDING_DIMENSION for _ in texts]

        async def embed_query(self, query: str) -> list[float]:
            return [0.1] * EMBEDDING_DIMENSION

    monkeypatch.setattr(
        dependencies, "build_embedding_provider", lambda settings=None: StubEmbeddings()
    )

    with caplog.at_level(logging.ERROR):
        services = build_services(
            generation_settings=generation(enabled=True),
            openrouter_settings=openrouter(),
        )

    assert services.memory_search is not None
    assert services.memory_processor is not None
    assert services.generation is None
    events = [getattr(r, "memovo_event", None) for r in caplog.records]
    assert "generation.startup_failed" in events
    assert KEY not in "\n".join(JsonFormatter().format(r) for r in caplog.records)


# --------------------------------------------------------------------------
# Startup log
# --------------------------------------------------------------------------
def test_startup_reports_generation_disabled(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO), TestClient(create_app(load_services=False)):
        pass

    started = next(
        getattr(r, "memovo_fields", {})
        for r in caplog.records
        if getattr(r, "memovo_event", None) == "service.started"
    )
    assert started["generation_available"] is False
    assert started["generation_provider"] in {"disabled", "openrouter"}
    assert isinstance(started["generation_enabled"], bool)


# --------------------------------------------------------------------------
# Wire mapping of generation failures (contract section 21.9)
# --------------------------------------------------------------------------
class FailingSearch:
    def __init__(self, error: BaseException) -> None:
        self.error = error

    async def search(self, request: object) -> object:
        raise self.error


def app_failing_with(error: BaseException) -> FastAPI:
    application = create_app(load_services=False)
    application.dependency_overrides[get_memory_search_service] = lambda: FailingSearch(error)
    return application


def client_for(error: BaseException) -> Iterator[TestClient]:
    with TestClient(app_failing_with(error), raise_server_exceptions=False) as client:
        yield client


def test_malformed_model_output_is_502_and_never_retryable() -> None:
    for client in client_for(InvalidGenerationOutputError("bad json")):
        response = client.post(SEARCH, json=BODY)

        assert response.status_code == 502
        assert response.json()["error"] == {
            "code": "AI_INVALID_RESPONSE",
            "message": "The generated output was invalid",
            "retryable": False,
        }
        assert "Retry-After" not in response.headers


def test_an_unavailable_provider_is_503_and_retryable() -> None:
    for client in client_for(GenerationUnavailableError("down")):
        response = client.post(SEARCH, json=BODY)

        assert response.status_code == 503
        assert response.json()["error"]["code"] == "GENERATION_UNAVAILABLE"
        assert response.json()["error"]["retryable"] is True


def test_rate_limiting_is_429_with_a_bounded_retry_after() -> None:
    for client in client_for(GenerationRateLimitedError("slow", retry_after_seconds=9)):
        response = client.post(SEARCH, json=BODY)

        assert response.status_code == 429
        assert response.json()["error"]["code"] == "RATE_LIMITED"
        assert response.json()["error"]["retryable"] is True
        assert response.headers["Retry-After"] == "9"


def test_rate_limiting_without_a_hint_sends_no_header() -> None:
    for client in client_for(GenerationRateLimitedError("slow")):
        response = client.post(SEARCH, json=BODY)

        assert response.status_code == 429
        assert "Retry-After" not in response.headers


def test_a_generation_timeout_is_504_and_retryable() -> None:
    for client in client_for(GenerationTimeoutError("slow")):
        response = client.post(SEARCH, json=BODY)

        assert response.status_code == 504
        assert response.json()["error"]["code"] == "TIMEOUT"
        assert response.json()["error"]["retryable"] is True


def test_provider_internals_never_reach_the_caller() -> None:
    private = "ZZQ-provider-said-this"

    for client in client_for(GenerationUnavailableError(private)):
        assert private not in client.post(SEARCH, json=BODY).text
