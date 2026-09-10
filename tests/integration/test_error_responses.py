"""Every public failure leaves as the standardized envelope.

Doc 05 section 10: cover every code, and prove that stack traces, raw
infrastructure exceptions, secrets and model paths never reach the caller.
"""

from collections.abc import Iterator, Sequence

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from memovo_ai.api.dependencies import get_memory_search_service
from memovo_ai.core.config import SearchSettings
from memovo_ai.core.errors import ERROR_HTTP_STATUS, ERROR_MESSAGE, AiServiceError
from memovo_ai.embeddings import EMBEDDING_DIMENSION, Embedding
from memovo_ai.embeddings.models import EmbeddingInferenceError, ModelUnavailableError
from memovo_ai.main import create_app
from memovo_ai.providers.vector_search import FakeVectorSearchProvider, VectorRecord
from memovo_ai.retrieval import VectorSearchHit
from memovo_ai.schemas import ErrorCode, ErrorResponse
from memovo_ai.services import MemorySearchService

pytestmark = pytest.mark.integration

USER = "user_123"
ENDPOINT = "/ai/memories/search"
VALID_BODY = {"query": "What did I save about MongoDB?", "userId": USER}

#: Strings that must never appear in a public error response.
LEAK_MARKERS = (
    "Traceback",
    "/opt/models",
    "C:\\Users",
    "mongodb://",
    "sk-secret",
    "site-packages",
    ".py",
)


class StubEmbeddings:
    async def embed_documents(self, texts: Sequence[str]) -> list[Embedding]:
        return [[0.1] * EMBEDDING_DIMENSION for _ in texts]

    async def embed_query(self, query: str) -> Embedding:
        return [0.1] * EMBEDDING_DIMENSION


class FailingEmbeddings(StubEmbeddings):
    def __init__(self, error: Exception) -> None:
        self._error = error

    async def embed_query(self, query: str) -> Embedding:
        raise self._error


class LeakingVectorSearch:
    async def search(
        self, *, user_id: str, query_embedding: Sequence[float], top_k: int
    ) -> list[VectorSearchHit]:
        return [
            VectorSearchHit(
                user_id="somebody_else",
                memory_id="memory_x",
                chunk_id="chunk_x",
                chunk_index=0,
                content="another user's private content",
                score=0.99,
                title="Their Memory",
            )
        ]


class BrokenVectorSearch:
    async def search(
        self, *, user_id: str, query_embedding: Sequence[float], top_k: int
    ) -> list[VectorSearchHit]:
        message = "connection to mongodb://user:sk-secret@db:27017 failed"
        raise RuntimeError(message)


def app_with(
    *,
    embeddings: StubEmbeddings | None = None,
    vector_search: object | None = None,
    records: Sequence[VectorRecord] = (),
    service: object | None = None,
) -> FastAPI:
    application = create_app(load_services=False)

    if service is None:
        service = MemorySearchService(
            embedding_provider=embeddings or StubEmbeddings(),
            vector_search=vector_search or FakeVectorSearchProvider(records),  # type: ignore[arg-type]
            settings=SearchSettings(_env_file=None),  # type: ignore[call-arg]
        )

    application.dependency_overrides[get_memory_search_service] = lambda: service
    return application


def client_of(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


@pytest.fixture
def client() -> Iterator[TestClient]:
    yield from client_of(app_with())


def assert_envelope(payload: object) -> ErrorResponse:
    """The body must validate against the contract model itself."""
    return ErrorResponse.model_validate(payload)


# --------------------------------------------------------------------------
# Shape
# --------------------------------------------------------------------------
def test_a_validation_failure_uses_the_envelope(client: TestClient) -> None:
    response = client.post(ENDPOINT, json={"query": "q"})

    assert response.status_code == 422
    body = response.json()
    assert set(body) == {"error"}
    assert set(body["error"]) == {"code", "message", "retryable"}
    assert body["error"]["code"] == "INVALID_INPUT"
    assert body["error"]["retryable"] is False


def test_the_error_body_validates_against_the_contract_model(client: TestClient) -> None:
    envelope = assert_envelope(client.post(ENDPOINT, json={}).json())

    assert envelope.error.code is ErrorCode.INVALID_INPUT


def test_no_fastapi_detail_key_survives(client: TestClient) -> None:
    """FastAPI's default `{"detail": ...}` must not reach the Backend."""
    assert "detail" not in client.post(ENDPOINT, json={}).json()


# --------------------------------------------------------------------------
# Codes reachable over HTTP
# --------------------------------------------------------------------------
def test_invalid_input_on_a_bad_body(client: TestClient) -> None:
    assert client.post(ENDPOINT, json={"query": 1, "userId": USER}).json()["error"]["code"] == (
        "INVALID_INPUT"
    )


def test_invalid_request_on_an_unknown_path(client: TestClient) -> None:
    response = client.get("/does-not-exist")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "INVALID_REQUEST"


def test_invalid_request_on_a_wrong_method(client: TestClient) -> None:
    response = client.get(ENDPOINT)

    assert response.status_code == 405
    assert response.json()["error"]["code"] == "INVALID_REQUEST"


def test_model_unavailable_when_the_model_failed_to_load() -> None:
    for http in client_of(create_app(load_services=False)):
        response = http.post(ENDPOINT, json=VALID_BODY)

        assert response.status_code == 503
        assert response.json()["error"] == {
            "code": "MODEL_UNAVAILABLE",
            "message": "Embedding model is temporarily unavailable",
            "retryable": True,
        }


def test_embedding_failed_when_inference_breaks() -> None:
    app = app_with(embeddings=FailingEmbeddings(EmbeddingInferenceError("inference failed")))

    for http in client_of(app):
        response = http.post(ENDPOINT, json=VALID_BODY)

        assert response.status_code == 503
        assert response.json()["error"]["code"] == "EMBEDDING_FAILED"
        assert response.json()["error"]["retryable"] is True


def test_model_unavailable_from_a_provider_error() -> None:
    app = app_with(embeddings=FailingEmbeddings(ModelUnavailableError("not loaded")))

    for http in client_of(app):
        assert http.post(ENDPOINT, json=VALID_BODY).json()["error"]["code"] == ("MODEL_UNAVAILABLE")


def test_internal_error_for_a_broken_user_pre_filter() -> None:
    """A leak is a defect, reported as INTERNAL_ERROR and never retryable."""
    app = app_with(vector_search=LeakingVectorSearch())

    for http in client_of(app):
        response = http.post(ENDPOINT, json=VALID_BODY)

        assert response.status_code == 500
        assert response.json()["error"]["code"] == "INTERNAL_ERROR"
        assert response.json()["error"]["retryable"] is False


def test_internal_error_for_an_unknown_failure() -> None:
    app = app_with(vector_search=BrokenVectorSearch())

    for http in client_of(app):
        response = http.post(ENDPOINT, json=VALID_BODY)

        assert response.status_code == 500
        assert response.json()["error"]["code"] == "INTERNAL_ERROR"


def test_invalid_input_for_a_blank_user_id() -> None:
    """The schema permits a blank string; scoping a search on it does not."""
    app = app_with()

    for http in client_of(app):
        response = http.post(ENDPOINT, json={"query": "q", "userId": "   "})

        assert response.status_code == 422
        assert response.json()["error"]["code"] == "INVALID_INPUT"


@pytest.mark.parametrize("code", list(ErrorCode))
def test_every_code_renders_correctly(code: ErrorCode) -> None:
    """Doc 05 section 10: cover every known code."""

    class RaisingService:
        async def search(self, request: object) -> object:
            raise AiServiceError(code)

    for http in client_of(app_with(service=RaisingService())):
        response = http.post(ENDPOINT, json=VALID_BODY)

        assert response.status_code == ERROR_HTTP_STATUS[code]
        assert response.json()["error"]["code"] == code.value
        assert response.json()["error"]["message"] == ERROR_MESSAGE[code]


# --------------------------------------------------------------------------
# Nothing leaks
# --------------------------------------------------------------------------
def test_an_infrastructure_exception_message_never_reaches_the_caller() -> None:
    """The raised text contains a connection string and a secret."""
    app = app_with(vector_search=BrokenVectorSearch())

    for http in client_of(app):
        body = http.post(ENDPOINT, json=VALID_BODY).text

        assert "mongodb://" not in body
        assert "sk-secret" not in body
        assert "27017" not in body


def test_another_users_content_never_appears_in_an_error() -> None:
    app = app_with(vector_search=LeakingVectorSearch())

    for http in client_of(app):
        body = http.post(ENDPOINT, json=VALID_BODY).text

        assert "another user's private content" not in body
        assert "somebody_else" not in body
        assert "memory_x" not in body


def test_a_rejected_value_is_not_echoed_back(client: TestClient) -> None:
    """Pydantic's own detail carries the rejected input; ours must not."""
    private_query = "my private memory about therapy"

    body = client.post(ENDPOINT, json={"query": private_query}).text

    assert private_query not in body
    assert "userId" in body  # the field name is contract, and is useful


@pytest.mark.parametrize(
    "request_body",
    [{}, {"query": "q"}, {"query": 1, "userId": USER}, {"query": "q", "userId": USER}],
)
def test_no_response_leaks_internals(request_body: dict[str, object]) -> None:
    for http in client_of(app_with(vector_search=BrokenVectorSearch())):
        body = http.post(ENDPOINT, json=request_body).text

        for marker in LEAK_MARKERS:
            assert marker not in body


def test_a_successful_search_is_unaffected() -> None:
    record = VectorRecord(
        user_id=USER,
        memory_id="memory_456",
        chunk_id="chunk_1",
        chunk_index=0,
        content="MongoDB Vector Search...",
        title="MongoDB Vector Search",
        tags=("mongodb",),
        score=0.94,
    )

    for http in client_of(app_with(records=[record])):
        response = http.post(ENDPOINT, json=VALID_BODY)

        assert response.status_code == 200
        assert "error" not in response.json()


def test_a_no_match_is_not_turned_into_an_error() -> None:
    for http in client_of(app_with()):
        response = http.post(ENDPOINT, json=VALID_BODY)

        assert response.status_code == 200
        assert response.json()["found"] is False
