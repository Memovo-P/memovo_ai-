"""Link ingestion through ``POST /ai/memories/process``.

Phase 17. Extraction is Backend-owned: the payload already contains the page
content, and the AI Service never fetches a URL.
"""

from collections.abc import Iterator, Sequence

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from memovo_ai.api.dependencies import get_memory_processor_service
from memovo_ai.embeddings import EMBEDDING_DIMENSION, Embedding
from memovo_ai.main import create_app
from memovo_ai.services import MemoryProcessorService

pytestmark = pytest.mark.integration

ENDPOINT = "/ai/memories/process"

NOTE_BODY: dict[str, object] = {
    "memoryId": "memory_456",
    "title": "MongoDB Vector Search",
    "description": "How Atlas Vector Search works...",
    "whySaved": "Useful for the memory project",
    "tags": ["mongodb", "vector-search", "ai"],
}

LINK_BODY: dict[str, object] = {
    "memoryId": "memory_link_1",
    "title": "Atlas Vector Search docs",
    "description": "Atlas Vector Search indexes embeddings for similarity queries.",
    "whySaved": "",
    "tags": ["mongodb", "docs"],
    "about": "Read this before the vector index migration",
    "source": {
        "siteName": "MongoDB Docs",
        "favicon": "https://cdn.example/favicon.ico",
        "ogImage": "https://cdn.example/og.png",
        "publishedAt": "2024-01-15",
    },
}


class StubEmbeddings:
    async def embed_documents(self, texts: Sequence[str]) -> list[Embedding]:
        return [[0.001 * (index + 1)] * EMBEDDING_DIMENSION for index in range(len(texts))]

    async def embed_query(self, query: str) -> Embedding:
        return [0.1] * EMBEDDING_DIMENSION


def build_app() -> FastAPI:
    application = create_app()
    application.dependency_overrides[get_memory_processor_service] = lambda: MemoryProcessorService(
        embedding_provider=StubEmbeddings()
    )
    return application


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(build_app(), raise_server_exceptions=False) as http:
        yield http


# --------------------------------------------------------------------------
# A Link is accepted and processed
# --------------------------------------------------------------------------
def test_a_link_request_is_accepted(client: TestClient) -> None:
    assert client.post(ENDPOINT, json=LINK_BODY).status_code == 200


def test_a_link_produces_chunks_with_embeddings(client: TestClient) -> None:
    payload = client.post(ENDPOINT, json=LINK_BODY).json()

    assert payload["chunks"]
    for chunk in payload["chunks"]:
        assert len(chunk["embedding"]) == EMBEDDING_DIMENSION
        assert chunk["chunkId"].startswith("chunk_")


def test_a_link_uses_the_same_response_shape_as_a_note(client: TestClient) -> None:
    link = client.post(ENDPOINT, json=LINK_BODY).json()
    note = client.post(ENDPOINT, json=NOTE_BODY).json()

    assert set(link) == set(note) == {"intent", "content", "chunks"}
    assert link["intent"] == note["intent"] == "save_memory"


def test_link_canonical_content_includes_the_available_fields(client: TestClient) -> None:
    content = client.post(ENDPOINT, json=LINK_BODY).json()["content"]

    assert content == (
        "Title:\nAtlas Vector Search docs\n\n"
        "Content:\nAtlas Vector Search indexes embeddings for similarity queries.\n\n"
        "About:\nRead this before the vector index migration\n\n"
        "Source:\nMongoDB Docs, 2024-01-15\n\n"
        "Tags:\nmongodb, docs"
    )


def test_asset_urls_never_reach_the_embedding_input(client: TestClient) -> None:
    """favicon and ogImage are asset URLs: no semantic value, so excluded."""
    payload = client.post(ENDPOINT, json=LINK_BODY).json()

    assert "favicon" not in payload["content"]
    assert "og.png" not in payload["content"]
    for chunk in payload["chunks"]:
        assert "cdn.example" not in chunk["content"]


def test_an_empty_why_saved_is_omitted_from_a_link(client: TestClient) -> None:
    assert "Why Saved:" not in client.post(ENDPOINT, json=LINK_BODY).json()["content"]


# --------------------------------------------------------------------------
# Graceful handling of missing metadata
# --------------------------------------------------------------------------
def test_a_link_with_an_empty_source_object_is_accepted(client: TestClient) -> None:
    response = client.post(ENDPOINT, json={**LINK_BODY, "source": {}})

    assert response.status_code == 200
    assert "Source:" not in response.json()["content"]


def test_a_link_with_all_null_source_fields_is_accepted(client: TestClient) -> None:
    source = dict.fromkeys(("siteName", "favicon", "ogImage", "publishedAt"))

    response = client.post(ENDPOINT, json={**LINK_BODY, "source": source})

    assert response.status_code == 200
    assert "Source:" not in response.json()["content"]
    assert response.json()["chunks"]


def test_a_link_without_about_is_accepted(client: TestClient) -> None:
    body = {key: value for key, value in LINK_BODY.items() if key != "about"}

    response = client.post(ENDPOINT, json=body)

    assert response.status_code == 200
    assert "About:" not in response.json()["content"]


def test_a_null_about_is_accepted(client: TestClient) -> None:
    response = client.post(ENDPOINT, json={**LINK_BODY, "about": None})

    assert response.status_code == 200
    assert "About:" not in response.json()["content"]


def test_only_site_name_available(client: TestClient) -> None:
    response = client.post(ENDPOINT, json={**LINK_BODY, "source": {"siteName": "MongoDB Docs"}})

    assert "Source:\nMongoDB Docs" in response.json()["content"]


def test_only_published_at_available(client: TestClient) -> None:
    response = client.post(ENDPOINT, json={**LINK_BODY, "source": {"publishedAt": "2024-01-15"}})

    assert "Source:\n2024-01-15" in response.json()["content"]


def test_only_asset_urls_available_yields_no_source_section(client: TestClient) -> None:
    source = {"favicon": "https://cdn.example/f.ico", "ogImage": "https://cdn.example/o.png"}

    response = client.post(ENDPOINT, json={**LINK_BODY, "source": source})

    assert response.status_code == 200
    assert "Source:" not in response.json()["content"]


# --------------------------------------------------------------------------
# Determinism and reprocessing hold for Links
# --------------------------------------------------------------------------
def test_reprocessing_a_link_reproduces_identical_chunk_ids(client: TestClient) -> None:
    first = client.post(ENDPOINT, json=LINK_BODY).json()
    second = client.post(ENDPOINT, json=LINK_BODY).json()

    assert first == second


def test_changing_about_changes_the_chunk_ids(client: TestClient) -> None:
    first = client.post(ENDPOINT, json=LINK_BODY).json()
    second = client.post(ENDPOINT, json={**LINK_BODY, "about": "A different note"}).json()

    assert [c["chunkId"] for c in first["chunks"]] != [c["chunkId"] for c in second["chunks"]]


def test_changing_only_a_non_embedded_source_field_keeps_chunk_ids(client: TestClient) -> None:
    """favicon does not reach the content, so it cannot move a chunk ID."""
    other = {**LINK_BODY["source"], "favicon": "https://cdn.example/changed.ico"}  # type: ignore[dict-item]

    first = client.post(ENDPOINT, json=LINK_BODY).json()
    second = client.post(ENDPOINT, json={**LINK_BODY, "source": other}).json()

    assert first == second


def test_a_note_is_unaffected_by_link_support(client: TestClient) -> None:
    """The Note path must be byte-identical to before Phase 17."""
    payload = client.post(ENDPOINT, json=NOTE_BODY).json()

    assert payload["content"] == (
        "Title:\nMongoDB Vector Search\n\n"
        "Content:\nHow Atlas Vector Search works...\n\n"
        "Why Saved:\nUseful for the memory project\n\n"
        "Tags:\nmongodb, vector-search, ai"
    )


# --------------------------------------------------------------------------
# Rejections
# --------------------------------------------------------------------------
def test_an_undocumented_source_field_is_rejected(client: TestClient) -> None:
    response = client.post(ENDPOINT, json={**LINK_BODY, "source": {"author": "someone"}})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_INPUT"


def test_a_non_object_source_is_rejected(client: TestClient) -> None:
    assert client.post(ENDPOINT, json={**LINK_BODY, "source": "mongodb.com"}).status_code == 422


def test_a_link_still_requires_the_note_fields(client: TestClient) -> None:
    body = {key: value for key, value in LINK_BODY.items() if key != "title"}

    assert client.post(ENDPOINT, json=body).status_code == 422


# --------------------------------------------------------------------------
# Extraction stays Backend-owned
# --------------------------------------------------------------------------
def test_the_service_imports_no_http_client() -> None:
    """The AI must not scrape, fetch or crawl. Checked across every layer that
    touches a Link."""
    import ast
    from pathlib import Path

    from memovo_ai.schemas import process as process_schema
    from memovo_ai.services import memory_processor
    from memovo_ai.understanding import content as content_module

    network = ("httpx", "requests", "aiohttp", "urllib", "http.client", "socket", "selenium")

    for module in (memory_processor, content_module, process_schema):
        tree = ast.parse(Path(module.__file__ or "").read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)

        leaked = [name for name in imported if any(name.startswith(n) for n in network)]
        assert not leaked, f"{module.__name__} imports {leaked}"


def test_a_url_in_the_payload_is_never_dereferenced(client: TestClient) -> None:
    """An unreachable host must not cause a failure or a delay."""
    source = {"siteName": "Example", "favicon": "https://127.0.0.1:9/nope.ico"}

    response = client.post(ENDPOINT, json={**LINK_BODY, "source": source})

    assert response.status_code == 200
