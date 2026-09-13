"""Link ingestion through ``POST /ai/memories/process``.

Contract v1.9, sections 8.2-8.5 and 19. Extraction is Backend-owned: the
payload already contains the page text, and the AI Service never fetches a
URL.
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
    "type": "note",
    "memoryId": "memory_456",
    "title": "MongoDB Vector Search",
    "content": "How Atlas Vector Search works...",
    "tags": ["mongodb", "vector-search", "ai"],
}

SOURCE: dict[str, object] = {
    "sourceTitle": "Tech Blog",
    "sourceDescription": "Articles about AI and ML",
    "authorName": "Jane Doe",
    "publicationDate": "2026-09-01",
}

NULL_SOURCE: dict[str, object] = dict.fromkeys(SOURCE)

LINK_BODY: dict[str, object] = {
    "type": "link",
    "memoryId": "memory_link_1",
    "url": "https://example.com/article",
    "title": "Understanding Vector Embeddings",
    "content": "Good reference for understanding embeddings",
    "tags": ["ai", "embeddings"],
    "source": SOURCE,
    "extractedContent": "Vector embeddings are numerical representations...",
}

LINK_CANONICAL = (
    "Title:\nUnderstanding Vector Embeddings\n\n"
    "Content:\nGood reference for understanding embeddings\n\n"
    "Source:\nTech Blog, Articles about AI and ML, Jane Doe, 2026-09-01\n\n"
    "Extracted Content:\nVector embeddings are numerical representations...\n\n"
    "Tags:\nai, embeddings"
)


class StubEmbeddings:
    async def embed_documents(self, texts: Sequence[str]) -> list[Embedding]:
        return [[0.001 * (index + 1)] * EMBEDDING_DIMENSION for index in range(len(texts))]

    async def embed_query(self, query: str) -> Embedding:
        return [0.1] * EMBEDDING_DIMENSION


def build_app() -> FastAPI:
    application = create_app(load_services=False)
    application.dependency_overrides[get_memory_processor_service] = lambda: MemoryProcessorService(
        embedding_provider=StubEmbeddings()
    )
    return application


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(build_app(), raise_server_exceptions=False) as http:
        yield http


def first_chunk(client: TestClient, body: dict[str, object]) -> str:
    response = client.post(ENDPOINT, json=body)
    assert response.status_code == 200, response.text
    return str(response.json()["chunks"][0]["content"])


# --------------------------------------------------------------------------
# A Link is accepted and processed
# --------------------------------------------------------------------------
def test_the_documented_link_is_accepted(client: TestClient) -> None:
    assert client.post(ENDPOINT, json=LINK_BODY).status_code == 200


def test_a_link_produces_chunks_with_embeddings(client: TestClient) -> None:
    payload = client.post(ENDPOINT, json=LINK_BODY).json()

    assert payload["memoryId"] == "memory_link_1"
    assert payload["chunks"]
    for chunk in payload["chunks"]:
        assert len(chunk["embedding"]) == EMBEDDING_DIMENSION
        assert chunk["chunkId"].startswith("chunk_")


def test_a_link_uses_the_same_response_shape_as_a_note(client: TestClient) -> None:
    link = client.post(ENDPOINT, json=LINK_BODY).json()
    note = client.post(ENDPOINT, json=NOTE_BODY).json()

    assert set(link) == set(note) == {"memoryId", "chunks"}


def test_link_canonical_content_includes_every_documented_input(client: TestClient) -> None:
    assert first_chunk(client, LINK_BODY) == LINK_CANONICAL


def test_the_url_never_reaches_the_embedding_input(client: TestClient) -> None:
    body = {**LINK_BODY, "url": "https://unique-host.example/secret-path"}

    for chunk in client.post(ENDPOINT, json=body).json()["chunks"]:
        assert "unique-host.example" not in chunk["content"]


# --------------------------------------------------------------------------
# Graceful handling of missing metadata and text
# --------------------------------------------------------------------------
def test_a_link_with_all_null_source_fields_is_accepted(client: TestClient) -> None:
    content = first_chunk(client, {**LINK_BODY, "source": NULL_SOURCE})

    assert "Source:" not in content
    assert "None" not in content


@pytest.mark.parametrize("field", list(SOURCE))
def test_a_single_available_source_field_is_embedded(client: TestClient, field: str) -> None:
    source = {**NULL_SOURCE, field: SOURCE[field]}

    assert f"Source:\n{SOURCE[field]}" in first_chunk(client, {**LINK_BODY, "source": source})


@pytest.mark.parametrize("value", [None, ""])
def test_null_and_empty_user_content_are_accepted(client: TestClient, value: str | None) -> None:
    content = first_chunk(client, {**LINK_BODY, "content": value})

    assert "Content:\nGood" not in content
    assert content.startswith("Title:\nUnderstanding Vector Embeddings\n\nSource:")
    assert "Extracted Content:" in content


def test_absent_user_content_is_accepted(client: TestClient) -> None:
    body = {key: value for key, value in LINK_BODY.items() if key != "content"}

    assert "Content:\nGood" not in first_chunk(client, body)


@pytest.mark.parametrize("value", [None, ""])
def test_null_and_empty_extracted_content_are_accepted(
    client: TestClient, value: str | None
) -> None:
    assert "Extracted Content:" not in first_chunk(client, {**LINK_BODY, "extractedContent": value})


def test_absent_extracted_content_is_accepted(client: TestClient) -> None:
    body = {key: value for key, value in LINK_BODY.items() if key != "extractedContent"}

    assert "Extracted Content:" not in first_chunk(client, body)


def test_a_long_extracted_page_is_accepted_and_chunked(client: TestClient) -> None:
    """No AI-side character cap (section 8.4); it just becomes more chunks."""
    body = {**LINK_BODY, "extractedContent": " ".join(f"word{n:05d}" for n in range(6000))}

    response = client.post(ENDPOINT, json=body)

    assert response.status_code == 200
    assert len(response.json()["chunks"]) > 1


def test_null_tags_are_accepted_on_a_link(client: TestClient) -> None:
    assert "Tags:" not in first_chunk(client, {**LINK_BODY, "tags": None})


# --------------------------------------------------------------------------
# Determinism and reprocessing hold for Links
# --------------------------------------------------------------------------
def test_reprocessing_a_link_reproduces_identical_chunk_ids(client: TestClient) -> None:
    first = client.post(ENDPOINT, json=LINK_BODY).json()
    second = client.post(ENDPOINT, json=LINK_BODY).json()

    assert first == second


def test_changing_user_content_changes_the_chunk_ids(client: TestClient) -> None:
    first = client.post(ENDPOINT, json=LINK_BODY).json()
    second = client.post(ENDPOINT, json={**LINK_BODY, "content": "A different note"}).json()

    assert [c["chunkId"] for c in first["chunks"]] != [c["chunkId"] for c in second["chunks"]]


def test_changing_only_the_url_keeps_chunk_ids(client: TestClient) -> None:
    first = client.post(ENDPOINT, json=LINK_BODY).json()
    second = client.post(ENDPOINT, json={**LINK_BODY, "url": "https://example.com/other"}).json()

    assert first == second


def test_a_note_is_unaffected_by_link_support(client: TestClient) -> None:
    assert first_chunk(client, NOTE_BODY) == (
        "Title:\nMongoDB Vector Search\n\n"
        "Content:\nHow Atlas Vector Search works...\n\n"
        "Tags:\nmongodb, vector-search, ai"
    )


# --------------------------------------------------------------------------
# Rejections (contract sections 8.2, 8.3, 8.5)
# --------------------------------------------------------------------------
def test_an_empty_source_object_is_rejected(client: TestClient) -> None:
    response = client.post(ENDPOINT, json={**LINK_BODY, "source": {}})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_INPUT"


@pytest.mark.parametrize(
    "field", ["platform", "contentType", "thumbnailUrl", "canonicalUrl", "siteName", "favicon"]
)
def test_a_backend_only_or_legacy_source_field_is_rejected(client: TestClient, field: str) -> None:
    response = client.post(ENDPOINT, json={**LINK_BODY, "source": {**SOURCE, field: "x"}})

    assert response.status_code == 422


def test_a_non_object_source_is_rejected(client: TestClient) -> None:
    assert client.post(ENDPOINT, json={**LINK_BODY, "source": "mongodb.com"}).status_code == 422


@pytest.mark.parametrize("field", ["url", "title", "source", "memoryId"])
def test_a_missing_required_link_field_is_rejected(client: TestClient, field: str) -> None:
    body = {key: value for key, value in LINK_BODY.items() if key != field}

    assert client.post(ENDPOINT, json=body).status_code == 422


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("url", "ftp://example.com/file"),
        ("url", "https://example.com/" + "a" * 2040),
        ("title", ""),
        ("title", "t" * 501),
        ("content", "c" * 1001),
        ("whySaved", "legacy"),
        ("about", "legacy"),
        ("description", "legacy"),
    ],
)
def test_a_link_limit_or_legacy_field_is_enforced(
    client: TestClient, field: str, value: object
) -> None:
    assert client.post(ENDPOINT, json={**LINK_BODY, field: value}).status_code == 422


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
    response = client.post(ENDPOINT, json={**LINK_BODY, "url": "https://127.0.0.1:9/nope"})

    assert response.status_code == 200
