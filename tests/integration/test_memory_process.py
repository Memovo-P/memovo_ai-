"""``POST /ai/memories/process`` through the HTTP layer (Note path).

Drives the real application with a stub embedding provider. No model is
loaded and no weights are downloaded. Link-specific behaviour is in
``test_link_process.py``.
"""

from collections.abc import Iterator, Sequence

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from memovo_ai.api.dependencies import get_memory_processor_service
from memovo_ai.chunking import ChunkingConfig, HybridChunker
from memovo_ai.embeddings import EMBEDDING_DIMENSION, Embedding
from memovo_ai.main import create_app
from memovo_ai.services import MemoryProcessorService

pytestmark = pytest.mark.integration

ENDPOINT = "/ai/memories/process"

VALID_BODY: dict[str, object] = {
    "type": "note",
    "memoryId": "memory_456",
    "title": "MongoDB Vector Search",
    "content": "How Atlas Vector Search works...",
    "tags": ["mongodb", "vector-search", "ai"],
}


class StubEmbeddings:
    async def embed_documents(self, texts: Sequence[str]) -> list[Embedding]:
        return [[0.001 * (index + 1)] * EMBEDDING_DIMENSION for index in range(len(texts))]

    async def embed_query(self, query: str) -> Embedding:
        return [0.1] * EMBEDDING_DIMENSION


def app_with(*, target: int = 500, overlap: int = 50) -> FastAPI:
    service = MemoryProcessorService(
        embedding_provider=StubEmbeddings(),
        chunker=HybridChunker(config=ChunkingConfig(target_tokens=target, overlap_tokens=overlap)),
    )
    application = create_app(load_services=False)
    application.dependency_overrides[get_memory_processor_service] = lambda: service
    return application


def client_of(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


@pytest.fixture
def client() -> Iterator[TestClient]:
    yield from client_of(app_with())


# --------------------------------------------------------------------------
# The endpoint
# --------------------------------------------------------------------------
def test_the_endpoint_is_mounted_at_the_contract_path(client: TestClient) -> None:
    assert client.post(ENDPOINT, json=VALID_BODY).status_code == 200


def test_the_retrieval_endpoints_are_published(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]

    assert "/ai/memories/process" in paths
    assert "/ai/memories/search" in paths


def test_get_is_not_allowed(client: TestClient) -> None:
    assert client.get(ENDPOINT).status_code == 405


# --------------------------------------------------------------------------
# Response shape (contract section 9)
# --------------------------------------------------------------------------
def test_the_response_uses_the_public_field_names(client: TestClient) -> None:
    payload = client.post(ENDPOINT, json=VALID_BODY).json()

    assert set(payload) == {"memoryId", "chunks"}
    assert set(payload["chunks"][0]) == {"chunkId", "chunkIndex", "content", "embedding"}


def test_the_memory_id_echoes_the_request(client: TestClient) -> None:
    assert client.post(ENDPOINT, json=VALID_BODY).json()["memoryId"] == "memory_456"


def test_no_legacy_top_level_fields_are_returned(client: TestClient) -> None:
    payload = client.post(ENDPOINT, json=VALID_BODY).json()

    assert "intent" not in payload
    assert "content" not in payload


def test_chunk_content_is_the_canonical_labelled_text(client: TestClient) -> None:
    content = client.post(ENDPOINT, json=VALID_BODY).json()["chunks"][0]["content"]

    assert content == (
        "Title:\nMongoDB Vector Search\n\n"
        "Content:\nHow Atlas Vector Search works...\n\n"
        "Tags:\nmongodb, vector-search, ai"
    )


def test_every_chunk_carries_an_embedding_of_the_locked_dimension(client: TestClient) -> None:
    payload = client.post(ENDPOINT, json=VALID_BODY).json()

    assert payload["chunks"]
    for chunk in payload["chunks"]:
        assert len(chunk["embedding"]) == EMBEDDING_DIMENSION
        assert all(isinstance(value, float) for value in chunk["embedding"])


def test_chunk_indices_are_contiguous_from_zero() -> None:
    body = {**VALID_BODY, "content": " ".join(f"w{n:03d}" for n in range(300))}

    for http in client_of(app_with(target=40, overlap=0)):
        chunks = http.post(ENDPOINT, json=body).json()["chunks"]

        assert len(chunks) > 1
        assert [c["chunkIndex"] for c in chunks] == list(range(len(chunks)))


def test_the_response_never_emits_snake_case_keys(client: TestClient) -> None:
    body = client.post(ENDPOINT, json=VALID_BODY).text

    for snake in ("chunk_id", "chunk_index", "memory_id"):
        assert snake not in body


# --------------------------------------------------------------------------
# Reprocessing (contract section 13)
# --------------------------------------------------------------------------
def test_reprocessing_unchanged_input_returns_identical_chunk_ids(client: TestClient) -> None:
    first = client.post(ENDPOINT, json=VALID_BODY).json()
    second = client.post(ENDPOINT, json=VALID_BODY).json()

    assert first == second


def test_a_changed_memory_returns_the_complete_chunk_set() -> None:
    """Never only the changed chunks."""
    paragraphs = ["Alpha. " + "alpha sentence. " * 8, "Beta. " + "beta sentence. " * 8]
    original = {**VALID_BODY, "content": "\n\n".join([*paragraphs, "Gamma. gamma text."])}
    edited = {**VALID_BODY, "content": "\n\n".join([*paragraphs, "Delta. delta text."])}

    for http in client_of(app_with(target=40, overlap=0)):
        first = http.post(ENDPOINT, json=original).json()["chunks"]
        second = http.post(ENDPOINT, json=edited).json()["chunks"]

        assert len(first) > 1
        assert len(second) > 1
        assert [c["chunkIndex"] for c in second] == list(range(len(second)))


def test_no_delta_fields_appear_in_the_response(client: TestClient) -> None:
    payload = client.post(ENDPOINT, json=VALID_BODY).json()

    for absent in ("changedChunks", "removedChunkIds", "delta", "partial"):
        assert absent not in payload


# --------------------------------------------------------------------------
# Request validation at the HTTP boundary (contract section 8.1)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("field", ["type", "memoryId", "title", "content"])
def test_a_missing_required_field_is_rejected(client: TestClient, field: str) -> None:
    body = {key: value for key, value in VALID_BODY.items() if key != field}

    response = client.post(ENDPOINT, json=body)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_INPUT"


@pytest.mark.parametrize(
    "extra", ["whySaved", "description", "about", "userId", "embeddingVersion", "jobId"]
)
def test_a_legacy_or_out_of_scope_field_is_rejected(client: TestClient, extra: str) -> None:
    assert client.post(ENDPOINT, json={**VALID_BODY, extra: "x"}).status_code == 422


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("type", "memo"),
        ("memoryId", ""),
        ("title", "t" * 201),
        ("content", ""),
        ("content", "c" * 10_001),
        ("tags", [f"t{n}" for n in range(21)]),
        ("tags", ["t" * 51]),
        ("tags", "mongodb"),
    ],
)
def test_a_contract_limit_is_enforced_at_the_boundary(
    client: TestClient, field: str, value: object
) -> None:
    response = client.post(ENDPOINT, json={**VALID_BODY, field: value})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_INPUT"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("title", ""),
        ("title", "t" * 200),
        ("content", "x"),
        ("content", "c" * 10_000),
        ("tags", None),
        ("tags", []),
        ("tags", [f"t{n}" for n in range(20)]),
        ("tags", ["t" * 50]),
    ],
)
def test_a_value_inside_the_contract_limits_is_accepted(
    client: TestClient, field: str, value: object
) -> None:
    assert client.post(ENDPOINT, json={**VALID_BODY, field: value}).status_code == 200


def test_absent_tags_are_accepted(client: TestClient) -> None:
    body = {key: value for key, value in VALID_BODY.items() if key != "tags"}

    assert client.post(ENDPOINT, json=body).status_code == 200


def test_snake_case_input_is_rejected(client: TestClient) -> None:
    body = {"type": "note", "memory_id": "memory_456", "title": "t", "content": "c"}

    assert client.post(ENDPOINT, json=body).status_code == 422


def test_validation_errors_use_the_error_envelope(client: TestClient) -> None:
    body = client.post(ENDPOINT, json={}).json()

    assert set(body) == {"error"}
    assert set(body["error"]) == {"code", "message", "retryable"}


def test_a_rejected_value_is_not_echoed_back(client: TestClient) -> None:
    private = "a private memory about therapy"

    body = client.post(ENDPOINT, json={"type": "note", "memoryId": "m", "content": private}).text

    assert private not in body


# --------------------------------------------------------------------------
# Content variations
# --------------------------------------------------------------------------
def test_empty_tags_are_accepted(client: TestClient) -> None:
    response = client.post(ENDPOINT, json={**VALID_BODY, "tags": []})

    assert response.status_code == 200
    assert "Tags:" not in response.json()["chunks"][0]["content"]


def test_arabic_content_is_accepted(client: TestClient) -> None:
    body = {**VALID_BODY, "title": "بحث المتجهات", "content": "كيف يعمل البحث الدلالي"}

    response = client.post(ENDPOINT, json=body)

    assert response.status_code == 200
    assert "بحث المتجهات" in response.json()["chunks"][0]["content"]


def test_a_memory_that_normalizes_to_nothing_returns_no_chunks(client: TestClient) -> None:
    """``content`` must be at least one character; whitespace satisfies that
    and still yields nothing to index. An empty chunk set is a valid response."""
    body = {"type": "note", "memoryId": "m", "title": "", "content": "   ", "tags": []}

    payload = client.post(ENDPOINT, json=body).json()

    assert payload == {"memoryId": "m", "chunks": []}


# --------------------------------------------------------------------------
# Failure handling
# --------------------------------------------------------------------------
def test_an_unavailable_model_reports_model_unavailable() -> None:
    """No dependency override: startup could not build the service."""
    for http in client_of(create_app(load_services=False)):
        response = http.post(ENDPOINT, json=VALID_BODY)

        assert response.status_code == 503
        assert response.json()["error"]["code"] == "MODEL_UNAVAILABLE"


def test_the_route_contains_no_ingestion_algorithm() -> None:
    """No chunking, hashing, embedding or persistence in the route."""
    import ast
    from pathlib import Path

    from memovo_ai.api.routes import memory_process

    tree = ast.parse(Path(memory_process.__file__ or "").read_text(encoding="utf-8"))
    imported = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module
    }

    for forbidden in ("memovo_ai.chunking", "memovo_ai.embeddings", "memovo_ai.understanding"):
        assert not any(module.startswith(forbidden) for module in imported)
    assert not any("hashing" in module for module in imported)
