"""``POST /ai/memories/process`` through the HTTP layer.

Drives the real application with a stub embedding provider. No model is
loaded and no weights are downloaded.
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
    "memoryId": "memory_456",
    "title": "MongoDB Vector Search",
    "description": "How Atlas Vector Search works...",
    "whySaved": "Useful for the memory project",
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


def test_both_sprint_1_endpoints_are_published(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]

    assert "/ai/memories/process" in paths
    assert "/ai/memories/search" in paths


def test_get_is_not_allowed(client: TestClient) -> None:
    assert client.get(ENDPOINT).status_code == 405


# --------------------------------------------------------------------------
# Response shape
# --------------------------------------------------------------------------
def test_the_response_uses_the_public_field_names(client: TestClient) -> None:
    payload = client.post(ENDPOINT, json=VALID_BODY).json()

    assert set(payload) == {"intent", "content", "chunks"}
    assert set(payload["chunks"][0]) == {"chunkId", "chunkIndex", "content", "embedding"}


def test_the_intent_is_save_memory(client: TestClient) -> None:
    assert client.post(ENDPOINT, json=VALID_BODY).json()["intent"] == "save_memory"


def test_content_is_the_canonical_labelled_text(client: TestClient) -> None:
    content = client.post(ENDPOINT, json=VALID_BODY).json()["content"]

    assert content.startswith("Title:\nMongoDB Vector Search")
    assert "Content:\nHow Atlas Vector Search works..." in content
    assert "Why Saved:\nUseful for the memory project" in content
    assert content.endswith("Tags:\nmongodb, vector-search, ai")


def test_every_chunk_carries_an_embedding_of_the_locked_dimension(client: TestClient) -> None:
    payload = client.post(ENDPOINT, json=VALID_BODY).json()

    assert payload["chunks"]
    for chunk in payload["chunks"]:
        assert len(chunk["embedding"]) == EMBEDDING_DIMENSION


def test_chunk_indices_are_contiguous_from_zero() -> None:
    body = {**VALID_BODY, "description": " ".join(f"w{n:03d}" for n in range(300))}

    for http in client_of(app_with(target=40, overlap=0)):
        chunks = http.post(ENDPOINT, json=body).json()["chunks"]

        assert len(chunks) > 1
        assert [c["chunkIndex"] for c in chunks] == list(range(len(chunks)))


def test_the_response_never_emits_snake_case_keys(client: TestClient) -> None:
    body = client.post(ENDPOINT, json=VALID_BODY).text

    for snake in ("chunk_id", "chunk_index", "memory_id", "why_saved"):
        assert snake not in body


# --------------------------------------------------------------------------
# Reprocessing (doc 01, decision 25)
# --------------------------------------------------------------------------
def test_reprocessing_unchanged_input_returns_identical_chunk_ids(client: TestClient) -> None:
    first = client.post(ENDPOINT, json=VALID_BODY).json()
    second = client.post(ENDPOINT, json=VALID_BODY).json()

    assert first == second


def test_a_changed_memory_returns_the_complete_chunk_set() -> None:
    """Never only the changed chunks (doc 01, decision 25)."""
    paragraphs = ["Alpha. " + "alpha sentence. " * 8, "Beta. " + "beta sentence. " * 8]
    original = {**VALID_BODY, "description": "\n\n".join([*paragraphs, "Gamma. gamma text."])}
    edited = {**VALID_BODY, "description": "\n\n".join([*paragraphs, "Delta. delta text."])}

    for http in client_of(app_with(target=40, overlap=0)):
        first = http.post(ENDPOINT, json=original).json()["chunks"]
        second = http.post(ENDPOINT, json=edited).json()["chunks"]

        assert len(first) > 1
        assert len(second) > 1
        # The whole set comes back, not just what changed.
        assert [c["chunkIndex"] for c in second] == list(range(len(second)))


def test_no_delta_fields_appear_in_the_response(client: TestClient) -> None:
    payload = client.post(ENDPOINT, json=VALID_BODY).json()

    for absent in ("changedChunks", "removedChunkIds", "delta", "partial"):
        assert absent not in payload


# --------------------------------------------------------------------------
# Request validation
# --------------------------------------------------------------------------
@pytest.mark.parametrize("field", ["memoryId", "title", "description", "whySaved", "tags"])
def test_a_missing_required_field_is_rejected(client: TestClient, field: str) -> None:
    body = {key: value for key, value in VALID_BODY.items() if key != field}

    response = client.post(ENDPOINT, json=body)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_INPUT"


@pytest.mark.parametrize("extra", ["userId", "embeddingVersion", "jobId", "createdAt"])
def test_an_out_of_scope_field_is_rejected(client: TestClient, extra: str) -> None:
    response = client.post(ENDPOINT, json={**VALID_BODY, extra: "x"})

    assert response.status_code == 422


def test_snake_case_input_is_rejected(client: TestClient) -> None:
    body = {
        "memory_id": "memory_456",
        "title": "t",
        "description": "d",
        "why_saved": "w",
        "tags": [],
    }

    assert client.post(ENDPOINT, json=body).status_code == 422


def test_a_wrong_type_is_rejected(client: TestClient) -> None:
    assert client.post(ENDPOINT, json={**VALID_BODY, "tags": "mongodb"}).status_code == 422


def test_validation_errors_use_the_error_envelope(client: TestClient) -> None:
    body = client.post(ENDPOINT, json={}).json()

    assert set(body) == {"error"}
    assert set(body["error"]) == {"code", "message", "retryable"}


def test_a_rejected_value_is_not_echoed_back(client: TestClient) -> None:
    private = "a private memory about therapy"

    body = client.post(ENDPOINT, json={"memoryId": "m", "description": private}).text

    assert private not in body


# --------------------------------------------------------------------------
# Content variations
# --------------------------------------------------------------------------
def test_empty_tags_are_accepted(client: TestClient) -> None:
    response = client.post(ENDPOINT, json={**VALID_BODY, "tags": []})

    assert response.status_code == 200
    assert "Tags:" not in response.json()["content"]


def test_arabic_content_is_accepted(client: TestClient) -> None:
    body = {**VALID_BODY, "title": "بحث المتجهات", "description": "كيف يعمل البحث الدلالي"}

    response = client.post(ENDPOINT, json=body)

    assert response.status_code == 200
    assert "بحث المتجهات" in response.json()["content"]


def test_an_entirely_empty_memory_returns_no_chunks(client: TestClient) -> None:
    body = {"memoryId": "m", "title": "", "description": "", "whySaved": "", "tags": []}

    payload = client.post(ENDPOINT, json=body).json()

    assert payload["content"] == ""
    assert payload["chunks"] == []


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
    """Doc 03 Phase 08: no chunking, hashing, embedding or persistence."""
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
