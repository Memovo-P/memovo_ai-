"""Sprint 1 end to end: Memory -> ingestion -> embedding -> retrieval.

Runs the real endpoints against each other. ``/ai/memories/process`` produces
chunks and embeddings; the Backend's role -- persisting them to the vector
database -- is played by loading the returned chunks into the in-memory
provider; ``/ai/memories/search`` then retrieves them.

The embedding provider is a deterministic bag-of-words stand-in rather than
Qwen3, so no weights are downloaded. It is real enough for the property under
test: text about the same topic scores higher than text about another.
"""

import hashlib
from collections.abc import Iterator, Sequence

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from memovo_ai.api.dependencies import (
    get_memory_processor_service,
    get_memory_search_service,
)
from memovo_ai.core.config import SearchSettings
from memovo_ai.embeddings import EMBEDDING_DIMENSION, Embedding
from memovo_ai.main import create_app
from memovo_ai.providers.vector_search import FakeVectorSearchProvider, VectorRecord
from memovo_ai.services import MemoryProcessorService, MemorySearchService

pytestmark = pytest.mark.integration

PROCESS = "/ai/memories/process"
SEARCH = "/ai/memories/search"

OWNER = "user_123"
INTRUDER = "user_999"

MONGO_MEMORY: dict[str, object] = {
    "memoryId": "memory_mongo",
    "title": "MongoDB Atlas Vector Search",
    "description": "Atlas Vector Search indexes embeddings for similarity queries.",
    "whySaved": "Useful for the memory project",
    "tags": ["mongodb", "vector-search"],
}

BREAD_MEMORY: dict[str, object] = {
    "memoryId": "memory_bread",
    "title": "Sourdough starter routine",
    "description": "Feed the starter with flour and water every morning.",
    "whySaved": "So the bread rises properly",
    "tags": ["baking", "sourdough"],
}


def _token_dimension(token: str) -> int:
    """A stable dimension per token: Python's str hash is salted per process."""
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % EMBEDDING_DIMENSION


def _embed(text: str) -> Embedding:
    """Bag-of-words vector, L2-normalized so cosine is the dot product."""
    vector = [0.0] * EMBEDDING_DIMENSION
    for token in text.lower().split():
        cleaned = token.strip(".,:!?()[]\"'")
        if cleaned:
            vector[_token_dimension(cleaned)] += 1.0

    norm = sum(value * value for value in vector) ** 0.5
    if norm == 0.0:
        return vector

    return [value / norm for value in vector]


class BagOfWordsEmbeddings:
    async def embed_documents(self, texts: Sequence[str]) -> list[Embedding]:
        return [_embed(text) for text in texts]

    async def embed_query(self, query: str) -> Embedding:
        return _embed(query)


def build_app(records: Sequence[VectorRecord] = ()) -> FastAPI:
    """One embedding provider shared by both services, as in production."""
    embeddings = BagOfWordsEmbeddings()
    application = create_app(load_services=False)
    application.dependency_overrides[get_memory_processor_service] = lambda: MemoryProcessorService(
        embedding_provider=embeddings
    )
    application.dependency_overrides[get_memory_search_service] = lambda: MemorySearchService(
        embedding_provider=embeddings,
        vector_search=FakeVectorSearchProvider(records),
        settings=SearchSettings(_env_file=None),  # type: ignore[call-arg]
    )
    return application


def index_response(
    payload: dict[str, object], *, memory: dict[str, object], user_id: str
) -> list[VectorRecord]:
    """Stand in for the Backend persisting a process response to the vector DB."""
    chunks: list[dict[str, object]] = payload["chunks"]  # type: ignore[assignment]

    return [
        VectorRecord(
            user_id=user_id,
            memory_id=str(memory["memoryId"]),
            chunk_id=str(chunk["chunkId"]),
            chunk_index=int(chunk["chunkIndex"]),  # type: ignore[call-overload]
            content=str(chunk["content"]),
            title=str(memory["title"]),
            tags=tuple(memory["tags"]),  # type: ignore[arg-type]
            embedding=tuple(chunk["embedding"]),  # type: ignore[arg-type]
        )
        for chunk in chunks
    ]


def ingest(memories: Sequence[tuple[dict[str, object], str]]) -> list[VectorRecord]:
    """Process each Memory through the real endpoint and index the result."""
    records: list[VectorRecord] = []

    with TestClient(build_app()) as client:
        for memory, owner in memories:
            response = client.post(PROCESS, json=memory)
            assert response.status_code == 200, response.text
            records.extend(index_response(response.json(), memory=memory, user_id=owner))

    return records


@pytest.fixture(scope="module")
def indexed() -> list[VectorRecord]:
    return ingest([(MONGO_MEMORY, OWNER), (BREAD_MEMORY, OWNER)])


def search_client(records: Sequence[VectorRecord]) -> Iterator[TestClient]:
    with TestClient(build_app(records)) as client:
        yield client


# --------------------------------------------------------------------------
# Ingestion produces something indexable
# --------------------------------------------------------------------------
def test_ingestion_produces_chunks_with_embeddings(indexed: list[VectorRecord]) -> None:
    assert len(indexed) >= 2
    for record in indexed:
        assert record.embedding is not None
        assert len(record.embedding) == EMBEDDING_DIMENSION


def test_every_indexed_chunk_has_a_deterministic_id(indexed: list[VectorRecord]) -> None:
    ids = [record.chunk_id for record in indexed]

    assert len(set(ids)) == len(ids)
    assert all(chunk_id.startswith("chunk_") for chunk_id in ids)


def test_reingesting_reproduces_the_same_chunk_ids() -> None:
    """A retry must not orphan what the Backend already stored."""
    first = ingest([(MONGO_MEMORY, OWNER)])
    second = ingest([(MONGO_MEMORY, OWNER)])

    assert [r.chunk_id for r in first] == [r.chunk_id for r in second]


# --------------------------------------------------------------------------
# Retrieval finds what ingestion indexed
# --------------------------------------------------------------------------
def test_a_relevant_query_retrieves_the_right_memory(indexed: list[VectorRecord]) -> None:
    query = "Atlas Vector Search indexes embeddings for similarity queries"

    for client in search_client(indexed):
        payload = client.post(SEARCH, json={"query": query, "userId": OWNER}).json()

        assert payload["found"] is True
        assert payload["results"][0]["memoryId"] == "memory_mongo"


def test_the_retrieved_memory_carries_its_title_and_tags(indexed: list[VectorRecord]) -> None:
    query = "Atlas Vector Search indexes embeddings for similarity queries"

    for client in search_client(indexed):
        result = client.post(SEARCH, json={"query": query, "userId": OWNER}).json()["results"][0]

        assert result["title"] == "MongoDB Atlas Vector Search"
        assert result["tags"] == ["mongodb", "vector-search"]
        assert result["chunks"]


def test_the_unrelated_memory_is_not_returned(indexed: list[VectorRecord]) -> None:
    query = "Atlas Vector Search indexes embeddings for similarity queries"

    for client in search_client(indexed):
        payload = client.post(SEARCH, json={"query": query, "userId": OWNER}).json()

        assert "memory_bread" not in [r["memoryId"] for r in payload["results"]]


def test_an_unrelated_query_finds_nothing(indexed: list[VectorRecord]) -> None:
    """Prefer no answer over a confident wrong one (doc 05, section 12)."""
    for client in search_client(indexed):
        payload = client.post(
            SEARCH, json={"query": "quarterly tax filing deadlines", "userId": OWNER}
        ).json()

        assert payload["found"] is False
        assert payload["message"] == "I couldn't find a relevant memory"


def test_search_results_never_expose_embeddings(indexed: list[VectorRecord]) -> None:
    """Checked as a field, not a substring: the indexed text itself contains
    the word "embeddings"."""
    query = "Atlas Vector Search indexes embeddings for similarity queries"

    for client in search_client(indexed):
        payload = client.post(SEARCH, json={"query": query, "userId": OWNER}).json()

        for result in payload["results"]:
            assert "embedding" not in result
            for chunk in result["chunks"]:
                assert set(chunk) == {"chunkId", "chunkIndex", "content"}


# --------------------------------------------------------------------------
# Isolation survives the full round trip
# --------------------------------------------------------------------------
def test_another_user_cannot_retrieve_the_indexed_memory(indexed: list[VectorRecord]) -> None:
    query = "Atlas Vector Search indexes embeddings for similarity queries"

    for client in search_client(indexed):
        payload = client.post(SEARCH, json={"query": query, "userId": INTRUDER}).json()

        assert payload["found"] is False


def test_a_higher_scoring_foreign_memory_is_never_returned() -> None:
    """The doc 05 section 8 scenario, over both real endpoints."""
    records = ingest([(MONGO_MEMORY, INTRUDER), (BREAD_MEMORY, OWNER)])
    query = "Atlas Vector Search indexes embeddings for similarity queries"

    for client in search_client(records):
        body = client.post(SEARCH, json={"query": query, "userId": OWNER}).text

        assert "memory_mongo" not in body
        assert "Atlas" not in body
