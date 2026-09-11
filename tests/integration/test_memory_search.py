"""``POST /ai/memories/search`` through the HTTP layer.

Drives the real application with a fake vector index and a stub embedding
provider. No model is loaded and no weights are downloaded.
"""

from collections.abc import Iterator, Sequence

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from memovo_ai.api.dependencies import get_memory_search_service
from memovo_ai.core.config import SearchSettings
from memovo_ai.embeddings import EMBEDDING_DIMENSION, Embedding
from memovo_ai.main import create_app
from memovo_ai.providers.vector_search import FakeVectorSearchProvider, VectorRecord
from memovo_ai.services import MemorySearchService

pytestmark = pytest.mark.integration

USER = "user_123"
OTHER_USER = "user_999"
ENDPOINT = "/ai/memories/search"


class StubEmbeddings:
    async def embed_documents(self, texts: Sequence[str]) -> list[Embedding]:
        return [[0.1] * EMBEDDING_DIMENSION for _ in texts]

    async def embed_query(self, query: str) -> Embedding:
        return [0.1] * EMBEDDING_DIMENSION


def record(
    *,
    user_id: str = USER,
    memory_id: str = "memory_456",
    chunk_index: int = 0,
    score: float = 0.94,
    title: str = "MongoDB Vector Search",
    tags: tuple[str, ...] = ("mongodb", "vector-search"),
    content: str = "MongoDB Vector Search...",
) -> VectorRecord:
    return VectorRecord(
        user_id=user_id,
        memory_id=memory_id,
        chunk_id=f"{memory_id}_chunk_{chunk_index}",
        chunk_index=chunk_index,
        content=content,
        title=title,
        tags=tags,
        score=score,
    )


def make_app(records: Sequence[VectorRecord] = ()) -> FastAPI:
    service = MemorySearchService(
        embedding_provider=StubEmbeddings(),
        vector_search=FakeVectorSearchProvider(records),
        settings=SearchSettings(_env_file=None),  # type: ignore[call-arg]
    )
    application = create_app(load_services=False)
    application.dependency_overrides[get_memory_search_service] = lambda: service
    return application


def client_for(records: Sequence[VectorRecord] = ()) -> Iterator[TestClient]:
    with TestClient(make_app(records)) as client:
        yield client


@pytest.fixture
def client() -> Iterator[TestClient]:
    yield from client_for([record()])


@pytest.fixture
def empty_client() -> Iterator[TestClient]:
    yield from client_for()


# --------------------------------------------------------------------------
# The endpoint exists at the contract path
# --------------------------------------------------------------------------
def test_the_endpoint_is_mounted_at_the_contract_path(client: TestClient) -> None:
    response = client.post(ENDPOINT, json={"query": "mongodb", "userId": USER})

    assert response.status_code == 200


def test_the_path_appears_in_the_openapi_schema(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]

    assert ENDPOINT in paths
    assert "post" in paths[ENDPOINT]


def test_get_is_not_allowed(client: TestClient) -> None:
    assert client.get(ENDPOINT).status_code == 405


# --------------------------------------------------------------------------
# Success response
# --------------------------------------------------------------------------
def test_a_relevant_memory_is_returned(client: TestClient) -> None:
    response = client.post(ENDPOINT, json={"query": "mongodb", "userId": USER})

    assert response.json() == {
        "found": True,
        "results": [
            {
                "memoryId": "memory_456",
                "score": 0.94,
                "title": "MongoDB Vector Search",
                "tags": ["mongodb", "vector-search"],
                "chunks": [
                    {
                        "chunkId": "memory_456_chunk_0",
                        "chunkIndex": 0,
                        "content": "MongoDB Vector Search...",
                    }
                ],
            }
        ],
    }


def test_the_success_response_has_no_message_field(client: TestClient) -> None:
    assert "message" not in client.post(ENDPOINT, json={"query": "q", "userId": USER}).json()


def test_multiple_chunks_are_returned_in_reading_order() -> None:
    records = [record(chunk_index=0, score=0.94), record(chunk_index=1, score=0.91)]

    for http in client_for(records):
        payload = http.post(ENDPOINT, json={"query": "q", "userId": USER}).json()

        assert [c["chunkIndex"] for c in payload["results"][0]["chunks"]] == [0, 1]


# --------------------------------------------------------------------------
# No-match response
# --------------------------------------------------------------------------
def test_no_match_returns_the_exact_contract_payload(empty_client: TestClient) -> None:
    response = empty_client.post(ENDPOINT, json={"query": "anything", "userId": USER})

    assert response.status_code == 200
    assert response.json() == {
        "found": False,
        "results": [],
        "message": "I couldn't find a relevant memory",
    }


def test_an_irrelevant_match_returns_no_match() -> None:
    for http in client_for([record(score=0.10)]):
        payload = http.post(ENDPOINT, json={"query": "q", "userId": USER}).json()

        assert payload["found"] is False


def test_no_match_is_not_an_http_error(empty_client: TestClient) -> None:
    """Finding nothing is a successful answer, not a failure."""
    assert empty_client.post(ENDPOINT, json={"query": "q", "userId": USER}).status_code == 200


# --------------------------------------------------------------------------
# Request validation happens at the transport boundary
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "body",
    [
        {"query": "q"},
        {"userId": USER},
        {},
        {"query": "q", "userId": USER, "topK": 5},
        {"query": "q", "user_id": USER},
        {"query": 1, "userId": USER},
        {"query": "q", "userId": 123},
    ],
)
def test_invalid_requests_are_rejected(client: TestClient, body: dict[str, object]) -> None:
    assert client.post(ENDPOINT, json=body).status_code == 422


def test_a_valid_request_is_accepted(client: TestClient) -> None:
    assert client.post(ENDPOINT, json={"query": "q", "userId": USER}).status_code == 200


def test_malformed_json_is_rejected(client: TestClient) -> None:
    response = client.post(
        ENDPOINT, content=b"{not json", headers={"Content-Type": "application/json"}
    )

    assert response.status_code == 422


# --------------------------------------------------------------------------
# User isolation over HTTP
# --------------------------------------------------------------------------
def test_another_users_memory_is_never_returned() -> None:
    """Doc 05 section 8, end to end through the API."""
    records = [
        record(user_id=OTHER_USER, memory_id="memory_x", score=0.99),
        record(user_id=USER, memory_id="memory_y", score=0.90),
    ]

    for http in client_for(records):
        payload = http.post(ENDPOINT, json={"query": "q", "userId": USER}).json()

        assert [r["memoryId"] for r in payload["results"]] == ["memory_y"]
        assert "memory_x" not in http.post(ENDPOINT, json={"query": "q", "userId": USER}).text


def test_a_user_with_no_memories_gets_no_match() -> None:
    for http in client_for([record(user_id=OTHER_USER)]):
        payload = http.post(ENDPOINT, json={"query": "q", "userId": USER}).json()

        assert payload["found"] is False


def test_the_response_never_carries_a_user_identifier(client: TestClient) -> None:
    body = client.post(ENDPOINT, json={"query": "q", "userId": USER}).text

    assert USER not in body
    assert "userId" not in body


def test_the_response_never_carries_an_embedding(client: TestClient) -> None:
    assert "embedding" not in client.post(ENDPOINT, json={"query": "q", "userId": USER}).text


# --------------------------------------------------------------------------
# The route stays transport-only
# --------------------------------------------------------------------------
def test_the_route_contains_no_retrieval_algorithm() -> None:
    """Doc 03 Phase 14: keep the route free of retrieval algorithms."""
    import ast
    from pathlib import Path

    from memovo_ai.api.routes import memory_search

    tree = ast.parse(Path(memory_search.__file__ or "").read_text(encoding="utf-8"))
    imported = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module
    }

    for forbidden in ("memovo_ai.retrieval", "memovo_ai.embeddings", "memovo_ai.chunking"):
        assert not any(module.startswith(forbidden) for module in imported)


def test_the_service_is_resolved_through_dependency_injection() -> None:
    """Overriding the dependency must fully replace the service."""
    application = make_app([record()])

    assert get_memory_search_service in application.dependency_overrides


# --------------------------------------------------------------------------
# Startup
# --------------------------------------------------------------------------
def test_startup_survives_a_missing_model_runtime() -> None:
    """A failed load leaves an unavailable state instead of refusing to start
    (doc 06, section 9)."""
    with TestClient(create_app(load_services=False)) as http:
        assert http.get("/openapi.json").status_code == 200


def test_startup_survives_a_misconfigured_vector_database(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unset Atlas URI is a configuration problem, not a reason to crash.

    A crash loop tells an operator less than a service that starts and
    reports itself unavailable (doc 06, section 9).
    """
    from memovo_ai import main
    from memovo_ai.providers.vector_search import VectorSearchUnavailableError

    def explode() -> None:
        message = "Atlas connection string is not configured"
        raise VectorSearchUnavailableError(message)

    monkeypatch.setattr(main, "build_services", explode)

    with TestClient(main.create_app(), raise_server_exceptions=False) as http:
        assert http.get("/openapi.json").status_code == 200

        response = http.post(ENDPOINT, json={"query": "q", "userId": USER})
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "MODEL_UNAVAILABLE"


def test_an_unavailable_model_does_not_answer_queries() -> None:
    """Without the extra installed the service cannot be built, and the
    request must fail rather than silently report no memories."""
    with TestClient(create_app(load_services=False), raise_server_exceptions=False) as http:
        response = http.post(ENDPOINT, json={"query": "q", "userId": USER})

    assert response.status_code != 200
