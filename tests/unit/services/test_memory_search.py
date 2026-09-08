"""The memory search service.

Covers the doc 05 section 7 search matrix, end to end through the service
with a fake vector index and a fake embedding provider. No model is loaded.
"""

import json
from collections.abc import Sequence

import pytest

from memovo_ai.core.config import SearchSettings
from memovo_ai.embeddings import EMBEDDING_DIMENSION, Embedding
from memovo_ai.providers.vector_search import (
    FakeVectorSearchProvider,
    UserIsolationError,
    VectorRecord,
)
from memovo_ai.retrieval import VectorSearchHit
from memovo_ai.schemas import (
    NO_MATCH_MESSAGE,
    SearchMemoryNoMatchResponse,
    SearchMemoryRequest,
    SearchMemorySuccessResponse,
)
from memovo_ai.services import MemorySearchService

pytestmark = pytest.mark.unit

USER = "user_123"
OTHER_USER = "user_999"


class StubEmbeddings:
    """A deterministic embedding provider. Records what it was asked to embed."""

    def __init__(self) -> None:
        self.queries: list[str] = []
        self.document_calls: list[list[str]] = []

    async def embed_documents(self, texts: Sequence[str]) -> list[Embedding]:
        self.document_calls.append(list(texts))
        return [[0.1] * EMBEDDING_DIMENSION for _ in texts]

    async def embed_query(self, query: str) -> Embedding:
        self.queries.append(query)
        return [0.1] * EMBEDDING_DIMENSION


class ExplodingEmbeddings(StubEmbeddings):
    async def embed_query(self, query: str) -> Embedding:
        message = "model unavailable"
        raise RuntimeError(message)


class LeakingVectorSearch:
    """Ignores `user_id`: stands in for a misconfigured production adapter."""

    def __init__(self, records: Sequence[VectorRecord]) -> None:
        self._records = list(records)

    async def search(
        self, *, user_id: str, query_embedding: Sequence[float], top_k: int
    ) -> list[VectorSearchHit]:
        return [
            VectorSearchHit(
                user_id=record.user_id,
                memory_id=record.memory_id,
                chunk_id=record.chunk_id,
                chunk_index=record.chunk_index,
                content=record.content,
                score=record.score or 0.0,
                title=record.title,
                tags=record.tags,
            )
            for record in self._records
        ][:top_k]


def chunk(
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


def build(
    records: Sequence[VectorRecord] = (),
    *,
    embeddings: StubEmbeddings | None = None,
    vector_search: object | None = None,
    **settings: object,
) -> tuple[MemorySearchService, StubEmbeddings, object]:
    provider = vector_search if vector_search is not None else FakeVectorSearchProvider(records)
    stub = embeddings if embeddings is not None else StubEmbeddings()
    service = MemorySearchService(
        embedding_provider=stub,
        vector_search=provider,  # type: ignore[arg-type]
        settings=SearchSettings(_env_file=None, **settings),  # type: ignore[call-arg]
    )
    return service, stub, provider


def request(query: str = "What did I save about MongoDB vector search?") -> SearchMemoryRequest:
    return SearchMemoryRequest.model_validate({"query": query, "userId": USER})


# --------------------------------------------------------------------------
# The documented pipeline
# --------------------------------------------------------------------------
async def test_a_relevant_memory_is_returned() -> None:
    service, _, _ = build([chunk()])

    response = await service.search(request())

    assert isinstance(response, SearchMemorySuccessResponse)
    assert response.found is True
    assert [result.memory_id for result in response.results] == ["memory_456"]
    assert response.results[0].score == pytest.approx(0.94)


async def test_the_query_is_embedded_once() -> None:
    service, embeddings, _ = build([chunk()])

    await service.search(request("what did I save?"))

    assert embeddings.queries == ["what did I save?"]


async def test_documents_are_never_embedded_at_search_time() -> None:
    """Retrieval runs over already-indexed chunks; nothing is rechunked."""
    service, embeddings, _ = build([chunk()])

    await service.search(request())

    assert embeddings.document_calls == []


async def test_the_query_embedding_is_passed_to_the_vector_engine() -> None:
    service, _, provider = build([chunk()])

    await service.search(request())

    assert provider.calls[0].query_embedding == tuple([0.1] * EMBEDDING_DIMENSION)  # type: ignore[attr-defined]


async def test_results_carry_the_full_contract_payload() -> None:
    service, _, _ = build([chunk(chunk_index=0), chunk(chunk_index=1, score=0.91)])

    response = await service.search(request())

    result = response.results[0]  # type: ignore[union-attr]
    assert result.title == "MongoDB Vector Search"
    assert result.tags == ["mongodb", "vector-search"]
    assert [c.chunk_index for c in result.chunks] == [0, 1]
    assert result.chunks[0].content == "MongoDB Vector Search..."


# --------------------------------------------------------------------------
# Top-K
# --------------------------------------------------------------------------
async def test_top_k_five_is_requested_from_the_engine() -> None:
    """Doc 01, decision 10."""
    service, _, provider = build([chunk()])

    await service.search(request())

    assert provider.calls[0].top_k == 5  # type: ignore[attr-defined]


async def test_a_configured_top_k_is_honoured() -> None:
    service, _, provider = build([chunk()], top_k=3)

    await service.search(request())

    assert provider.calls[0].top_k == 3  # type: ignore[attr-defined]


async def test_only_five_chunks_are_considered() -> None:
    records = [chunk(memory_id=f"memory_{n}", score=0.99 - n / 100) for n in range(10)]
    service, _, _ = build(records)

    response = await service.search(request())

    assert len(response.results) == 5  # type: ignore[union-attr]


# --------------------------------------------------------------------------
# Threshold
# --------------------------------------------------------------------------
async def test_chunks_below_the_threshold_are_excluded() -> None:
    service, _, _ = build([chunk(memory_id="good", score=0.80), chunk(memory_id="bad", score=0.50)])

    response = await service.search(request())

    assert [r.memory_id for r in response.results] == ["good"]  # type: ignore[union-attr]


async def test_a_chunk_exactly_at_the_threshold_is_relevant() -> None:
    """Doc 05, section 7: threshold equality behaviour."""
    service, _, _ = build([chunk(score=0.75)])

    response = await service.search(request())

    assert isinstance(response, SearchMemorySuccessResponse)


async def test_a_chunk_just_below_the_threshold_is_not() -> None:
    service, _, _ = build([chunk(score=0.7499)])

    response = await service.search(request())

    assert isinstance(response, SearchMemoryNoMatchResponse)


async def test_a_configured_threshold_is_honoured() -> None:
    service, _, _ = build([chunk(score=0.65)], similarity_threshold=0.60)

    response = await service.search(request())

    assert isinstance(response, SearchMemorySuccessResponse)


# --------------------------------------------------------------------------
# Deduplication and ranking
# --------------------------------------------------------------------------
async def test_multiple_chunks_of_one_memory_collapse_to_one_result() -> None:
    service, _, _ = build([chunk(chunk_index=0, score=0.94), chunk(chunk_index=1, score=0.91)])

    response = await service.search(request())

    assert len(response.results) == 1  # type: ignore[union-attr]
    assert len(response.results[0].chunks) == 2  # type: ignore[union-attr]


async def test_memory_score_is_the_best_chunk_score() -> None:
    service, _, _ = build([chunk(chunk_index=0, score=0.80), chunk(chunk_index=1, score=0.96)])

    response = await service.search(request())

    assert response.results[0].score == pytest.approx(0.96)  # type: ignore[union-attr]


async def test_memories_are_returned_in_descending_score_order() -> None:
    service, _, _ = build(
        [
            chunk(memory_id="mid", score=0.85),
            chunk(memory_id="high", score=0.95),
            chunk(memory_id="low", score=0.78),
        ]
    )

    response = await service.search(request())

    assert [r.memory_id for r in response.results] == ["high", "mid", "low"]  # type: ignore[union-attr]


# --------------------------------------------------------------------------
# No match
# --------------------------------------------------------------------------
async def test_no_relevant_chunk_returns_the_contract_no_match_response() -> None:
    service, _, _ = build([chunk(score=0.10)])

    response = await service.search(request())

    assert isinstance(response, SearchMemoryNoMatchResponse)
    assert response.found is False
    assert response.results == []
    assert response.message == NO_MATCH_MESSAGE


async def test_an_empty_index_returns_no_match() -> None:
    service, _, _ = build([])

    assert isinstance(await service.search(request()), SearchMemoryNoMatchResponse)


async def test_the_no_match_payload_matches_the_contract_exactly() -> None:
    service, _, _ = build([])

    response = await service.search(request())

    assert json.loads(response.model_dump_json()) == {
        "found": False,
        "results": [],
        "message": "I couldn't find a relevant memory",
    }


# --------------------------------------------------------------------------
# User isolation
# --------------------------------------------------------------------------
async def test_another_users_memory_is_never_returned() -> None:
    """Doc 05, section 8: the higher-scoring foreign chunk must not appear."""
    service, _, _ = build(
        [
            chunk(user_id=OTHER_USER, memory_id="memory_x", score=0.99),
            chunk(user_id=USER, memory_id="memory_y", score=0.90),
        ]
    )

    response = await service.search(request())

    assert [r.memory_id for r in response.results] == ["memory_y"]  # type: ignore[union-attr]


async def test_the_trusted_user_id_is_used_as_the_pre_filter() -> None:
    service, _, provider = build([chunk()])

    await service.search(request())

    assert [call.user_id for call in provider.calls] == [USER]  # type: ignore[attr-defined]


async def test_isolation_is_enforced_even_for_an_unwrapped_provider() -> None:
    """The service wraps the provider itself; the guarantee is not opt-in."""
    records = [chunk(user_id=OTHER_USER, memory_id="memory_x", score=0.99)]
    service, _, _ = build(vector_search=LeakingVectorSearch(records))

    with pytest.raises(UserIsolationError):
        await service.search(request())


async def test_only_the_requesting_user_sees_their_own_memory() -> None:
    records = [
        chunk(user_id=USER, memory_id="mine", score=0.90),
        chunk(user_id=OTHER_USER, memory_id="theirs", score=0.95),
    ]
    service, _, _ = build(records)

    response = await service.search(request())
    payload = response.model_dump_json()

    assert "theirs" not in payload


# --------------------------------------------------------------------------
# The response never leaks internals
# --------------------------------------------------------------------------
async def test_the_response_carries_no_user_identifier() -> None:
    service, _, _ = build([chunk()])

    payload = (await service.search(request())).model_dump_json()

    assert USER not in payload
    assert "userId" not in payload
    assert "user_id" not in payload


async def test_the_response_carries_no_embedding() -> None:
    service, _, _ = build([chunk()])

    payload = (await service.search(request())).model_dump_json()

    assert "embedding" not in payload


async def test_the_response_uses_the_public_field_names() -> None:
    service, _, _ = build([chunk()])

    payload = json.loads((await service.search(request())).model_dump_json())

    assert set(payload) == {"found", "results"}
    assert set(payload["results"][0]) == {"memoryId", "score", "title", "tags", "chunks"}
    assert set(payload["results"][0]["chunks"][0]) == {"chunkId", "chunkIndex", "content"}


# --------------------------------------------------------------------------
# Failures propagate for Phase 15 to map
# --------------------------------------------------------------------------
async def test_an_embedding_failure_is_not_swallowed() -> None:
    """Mapping to MODEL_UNAVAILABLE / EMBEDDING_FAILED is Phase 15."""
    service, _, _ = build([chunk()], embeddings=ExplodingEmbeddings())

    with pytest.raises(RuntimeError, match="model unavailable"):
        await service.search(request())


async def test_a_vector_search_failure_is_not_swallowed() -> None:
    class BrokenVectorSearch:
        async def search(
            self, *, user_id: str, query_embedding: Sequence[float], top_k: int
        ) -> list[VectorSearchHit]:
            message = "vector engine unreachable"
            raise RuntimeError(message)

    service, _, _ = build(vector_search=BrokenVectorSearch())

    with pytest.raises(RuntimeError, match="vector engine unreachable"):
        await service.search(request())


async def test_a_failure_never_degrades_into_a_no_match_response() -> None:
    """Reporting "no memories" for an outage would be a silent wrong answer."""
    service, _, _ = build([chunk()], embeddings=ExplodingEmbeddings())

    with pytest.raises(RuntimeError):
        await service.search(request())


# --------------------------------------------------------------------------
# Scope
# --------------------------------------------------------------------------
def test_the_service_imports_no_persistence_or_auth_library() -> None:
    """Doc 03 Phase 07 forbids persistence and auth inside the service.

    Checked against the module's actual imports rather than its text, so the
    docstring describing what the service does *not* do cannot trip it.
    """
    import ast
    from pathlib import Path

    from memovo_ai.services import memory_search

    tree = ast.parse(Path(memory_search.__file__ or "").read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    forbidden = ("pymongo", "motor", "jwt", "jose", "passlib", "qdrant", "pinecone", "boto3")
    assert not [name for name in imported if any(bad in name for bad in forbidden)]


def test_the_service_depends_only_on_allowed_layers() -> None:
    """API -> services -> domain / provider interfaces (doc 04)."""
    import ast
    from pathlib import Path

    from memovo_ai.services import memory_search

    tree = ast.parse(Path(memory_search.__file__ or "").read_text(encoding="utf-8"))
    internal = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("memovo_ai")
    }

    allowed_roots = {
        "memovo_ai.core",
        "memovo_ai.embeddings",
        "memovo_ai.providers",
        "memovo_ai.retrieval",
        "memovo_ai.schemas",
    }
    for module in internal:
        assert any(module.startswith(root) for root in allowed_roots), module

    # A service must never reach up into the transport layer.
    assert not any(module.startswith("memovo_ai.api") for module in internal)


async def test_the_service_is_stateless_across_calls() -> None:
    service, _, _ = build([chunk()])

    first = await service.search(request())
    second = await service.search(request())

    assert first.model_dump_json() == second.model_dump_json()
