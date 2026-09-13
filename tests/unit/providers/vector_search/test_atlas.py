"""The MongoDB Atlas Vector Search adapter.

Driven by a stub collection, so these run in CI without the ``atlas`` extra
and without a cluster. The property that matters most -- that ``userId`` is a
**pre-filter inside** the ``$vectorSearch`` stage rather than a ``$match``
afterwards -- is asserted against the generated pipeline, which is exactly
where a regression would appear.
"""

from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any

import pytest

from memovo_ai.core.config import AtlasSettings
from memovo_ai.providers.vector_search import (
    VECTOR_WRITE_OPERATIONS,
    InvalidUserScopeError,
    MongoAtlasVectorSearchProvider,
    VectorSearchProvider,
    VectorSearchUnavailableError,
)
from memovo_ai.providers.vector_search.atlas import build_search_pipeline

pytestmark = pytest.mark.unit

USER = "user_123"
QUERY = [0.1] * 8


def settings(**overrides: object) -> AtlasSettings:
    base: dict[str, object] = {
        "uri": "mongodb+srv://user:pw@cluster.example/",
        "database": "memovo",
        "collection": "memory_vectors",
        "index": "vector_index",
        "path": "embedding",
    }
    return AtlasSettings(_env_file=None, **{**base, **overrides})  # type: ignore[arg-type]


def document(
    *,
    user_id: str = USER,
    memory_id: str = "memory_456",
    chunk_id: str = "chunk_001",
    chunk_index: int = 0,
    content: str = "MongoDB Vector Search...",
    title: str | None = "MongoDB Vector Search",
    tags: object = ("mongodb",),
    score: float | None = 0.94,
) -> dict[str, Any]:
    doc: dict[str, Any] = {
        "userId": user_id,
        "memoryId": memory_id,
        "chunkId": chunk_id,
        "chunkIndex": chunk_index,
        "content": content,
        "title": title,
        "tags": tags,
    }
    if score is not None:
        doc["score"] = score
    return doc


class StubCollection:
    """Records the pipeline it was given and replays fixed documents."""

    def __init__(
        self,
        documents: Sequence[Mapping[str, Any]] = (),
        *,
        error: Exception | None = None,
        awaitable_cursor: bool = False,
    ) -> None:
        self._documents = list(documents)
        self._error = error
        self._awaitable = awaitable_cursor
        self.pipelines: list[list[Mapping[str, Any]]] = []

    def aggregate(self, pipeline: Sequence[Mapping[str, Any]], /) -> object:
        self.pipelines.append([dict(stage) for stage in pipeline])
        if self._error is not None:
            raise self._error

        if self._awaitable:
            return self._awaited()
        return self._iterate()

    async def _awaited(self) -> AsyncIterator[Mapping[str, Any]]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[Mapping[str, Any]]:
        for doc in self._documents:
            yield doc


def provider(collection: StubCollection, **overrides: object) -> MongoAtlasVectorSearchProvider:
    return MongoAtlasVectorSearchProvider(collection, settings=settings(**overrides))


def vector_stage(collection: StubCollection) -> dict[str, Any]:
    return dict(collection.pipelines[0][0]["$vectorSearch"])


# --------------------------------------------------------------------------
# The user pre-filter -- the critical property
# --------------------------------------------------------------------------
async def test_the_user_filter_lives_inside_the_vector_search_stage() -> None:
    """Inside the stage means Atlas applies it during index traversal."""
    collection = StubCollection([document()])

    await provider(collection).search(user_id=USER, query_embedding=QUERY, top_k=5)

    assert vector_stage(collection)["filter"] == {"userId": {"$eq": USER}}


async def test_no_match_stage_filters_users_after_the_search() -> None:
    """A $match on userId after $vectorSearch would be a post-filter."""
    collection = StubCollection([document()])

    await provider(collection).search(user_id=USER, query_embedding=QUERY, top_k=5)

    stages = [name for stage in collection.pipelines[0] for name in stage]
    assert "$match" not in stages
    assert stages[0] == "$vectorSearch"


async def test_the_trusted_user_id_reaches_the_engine_verbatim() -> None:
    collection = StubCollection([])
    identifier = "  user_with_spaces  "

    await provider(collection).search(user_id=identifier, query_embedding=QUERY, top_k=5)

    assert vector_stage(collection)["filter"]["userId"]["$eq"] == identifier


@pytest.mark.parametrize("user_id", ["", "   ", "\t"])
async def test_a_blank_user_id_never_reaches_the_engine(user_id: str) -> None:
    collection = StubCollection([document()])

    with pytest.raises(InvalidUserScopeError):
        await provider(collection).search(user_id=user_id, query_embedding=QUERY, top_k=5)

    assert collection.pipelines == []


# --------------------------------------------------------------------------
# Pipeline shape
# --------------------------------------------------------------------------
def test_the_pipeline_targets_the_configured_index_and_path() -> None:
    pipeline = build_search_pipeline(
        user_id=USER, query_embedding=QUERY, top_k=5, settings=settings()
    )
    stage = pipeline[0]["$vectorSearch"]

    assert stage["index"] == "vector_index"
    assert stage["path"] == "embedding"
    assert stage["queryVector"] == QUERY


def test_the_limit_is_top_k() -> None:
    pipeline = build_search_pipeline(
        user_id=USER, query_embedding=QUERY, top_k=5, settings=settings()
    )

    assert pipeline[0]["$vectorSearch"]["limit"] == 5


def test_num_candidates_exceeds_the_limit() -> None:
    """Atlas requires it, and too few candidates costs recall."""
    pipeline = build_search_pipeline(
        user_id=USER, query_embedding=QUERY, top_k=5, settings=settings()
    )
    stage = pipeline[0]["$vectorSearch"]

    assert stage["numCandidates"] >= stage["limit"]
    assert stage["numCandidates"] == 100


def test_num_candidates_scales_with_a_large_top_k() -> None:
    pipeline = build_search_pipeline(
        user_id=USER, query_embedding=QUERY, top_k=50, settings=settings()
    )

    assert pipeline[0]["$vectorSearch"]["numCandidates"] == 500


def test_the_projection_returns_every_field_a_search_result_needs() -> None:
    pipeline = build_search_pipeline(
        user_id=USER, query_embedding=QUERY, top_k=5, settings=settings()
    )
    projection = pipeline[1]["$project"]

    assert projection["score"] == {"$meta": "vectorSearchScore"}
    assert projection["_id"] == 0
    for field in ("userId", "memoryId", "chunkId", "chunkIndex", "content", "title", "tags"):
        assert projection[field] == 1


def test_the_projection_never_returns_the_stored_vector() -> None:
    """Embeddings must not travel back through search results."""
    pipeline = build_search_pipeline(
        user_id=USER, query_embedding=QUERY, top_k=5, settings=settings()
    )

    assert "embedding" not in pipeline[1]["$project"]


# --------------------------------------------------------------------------
# Reading results
# --------------------------------------------------------------------------
async def test_a_document_becomes_a_hit() -> None:
    hits = await provider(StubCollection([document()])).search(
        user_id=USER, query_embedding=QUERY, top_k=5
    )

    assert len(hits) == 1
    assert hits[0].memory_id == "memory_456"
    assert hits[0].chunk_id == "chunk_001"
    assert hits[0].chunk_index == 0
    assert hits[0].score == pytest.approx(0.94)
    assert hits[0].title == "MongoDB Vector Search"
    assert hits[0].tags == ("mongodb",)


async def test_hits_come_back_best_first() -> None:
    collection = StubCollection(
        [
            document(chunk_id="chunk_low", score=0.10),
            document(chunk_id="chunk_high", score=0.99),
            document(chunk_id="chunk_mid", score=0.50),
        ]
    )

    hits = await provider(collection).search(user_id=USER, query_embedding=QUERY, top_k=5)

    assert [hit.chunk_id for hit in hits] == ["chunk_high", "chunk_mid", "chunk_low"]


async def test_an_empty_result_set_is_not_an_error() -> None:
    assert (
        await provider(StubCollection([])).search(user_id=USER, query_embedding=QUERY, top_k=5)
        == []
    )


async def test_a_missing_title_becomes_an_empty_string() -> None:
    hits = await provider(StubCollection([document(title=None)])).search(
        user_id=USER, query_embedding=QUERY, top_k=5
    )

    assert hits[0].title == ""


async def test_missing_tags_become_an_empty_tuple() -> None:
    hits = await provider(StubCollection([document(tags=None)])).search(
        user_id=USER, query_embedding=QUERY, top_k=5
    )

    assert hits[0].tags == ()


async def test_an_awaitable_cursor_is_supported() -> None:
    """pymongo's async client returns a coroutine wrapping the cursor."""
    collection = StubCollection([document()], awaitable_cursor=True)

    assert len(await provider(collection).search(user_id=USER, query_embedding=QUERY, top_k=5)) == 1


# --------------------------------------------------------------------------
# Malformed index documents
# --------------------------------------------------------------------------
@pytest.mark.parametrize("field", ["userId", "memoryId", "chunkId", "chunkIndex", "content"])
async def test_a_document_missing_a_required_field_is_refused(field: str) -> None:
    """Skipping it would silently return fewer results than the index holds."""
    broken = document()
    del broken[field]

    with pytest.raises(VectorSearchUnavailableError, match=field):
        await provider(StubCollection([broken])).search(
            user_id=USER, query_embedding=QUERY, top_k=5
        )


async def test_a_string_tags_field_is_refused() -> None:
    with pytest.raises(VectorSearchUnavailableError, match="tags"):
        await provider(StubCollection([document(tags="mongodb")])).search(
            user_id=USER, query_embedding=QUERY, top_k=5
        )


async def test_an_unreadable_chunk_index_is_refused() -> None:
    with pytest.raises(VectorSearchUnavailableError):
        await provider(StubCollection([document(chunk_index="not-a-number")])).search(  # type: ignore[arg-type]
            user_id=USER, query_embedding=QUERY, top_k=5
        )


async def test_a_malformed_document_error_reveals_no_content() -> None:
    """Messages reach logs; chunk text must never appear in them."""
    private_text = "the user's private memory text"
    broken = document(content=private_text)
    del broken["memoryId"]

    with pytest.raises(VectorSearchUnavailableError) as raised:
        await provider(StubCollection([broken])).search(
            user_id=USER, query_embedding=QUERY, top_k=5
        )

    assert private_text not in str(raised.value)


# --------------------------------------------------------------------------
# Failure and timeout handling
# --------------------------------------------------------------------------
# These deliberately copy pymongo's own class names -- `_is_timeout` matches
# by type name so the driver stays an optional import, so the names are the
# thing under test. N818 wants an Error suffix pymongo itself does not use.
class ServerSelectionTimeoutError(Exception):
    """Mimics the pymongo exception without importing the driver."""


class ExecutionTimeout(Exception):  # noqa: N818
    """Mimics pymongo.errors.ExecutionTimeout."""


class AutoReconnect(Exception):  # noqa: N818
    """Mimics pymongo.errors.AutoReconnect."""


@pytest.mark.parametrize("error", [ServerSelectionTimeoutError(), ExecutionTimeout()])
async def test_driver_timeouts_become_timeout_errors(error: Exception) -> None:
    """Phase 15 maps TimeoutError to TIMEOUT (504)."""
    with pytest.raises(TimeoutError):
        await provider(StubCollection(error=error)).search(
            user_id=USER, query_embedding=QUERY, top_k=5
        )


async def test_a_builtin_timeout_propagates_unchanged() -> None:
    with pytest.raises(TimeoutError):
        await provider(StubCollection(error=TimeoutError("slow"))).search(
            user_id=USER, query_embedding=QUERY, top_k=5
        )


async def test_other_driver_failures_become_vector_search_unavailable() -> None:
    """Phase 15 maps this to VECTOR_SEARCH_FAILED (503, retryable)."""
    with pytest.raises(VectorSearchUnavailableError, match="vector search failed"):
        await provider(StubCollection(error=AutoReconnect("connection reset"))).search(
            user_id=USER, query_embedding=QUERY, top_k=5
        )


async def test_a_driver_error_never_leaks_the_connection_string() -> None:
    leaky = AutoReconnect("failed connecting to mongodb+srv://user:hunter2@cluster.example/")

    with pytest.raises(VectorSearchUnavailableError) as raised:
        await provider(StubCollection(error=leaky)).search(
            user_id=USER, query_embedding=QUERY, top_k=5
        )

    message = str(raised.value)
    assert "mongodb+srv://" not in message
    assert "hunter2" not in message


async def test_the_original_error_stays_available_as_the_cause() -> None:
    original = AutoReconnect("connection reset")

    with pytest.raises(VectorSearchUnavailableError) as raised:
        await provider(StubCollection(error=original)).search(
            user_id=USER, query_embedding=QUERY, top_k=5
        )

    assert raised.value.__cause__ is original


@pytest.mark.parametrize("top_k", [0, -1])
async def test_a_non_positive_top_k_is_rejected(top_k: int) -> None:
    with pytest.raises(ValueError, match="top_k"):
        await provider(StubCollection()).search(user_id=USER, query_embedding=QUERY, top_k=top_k)


# --------------------------------------------------------------------------
# The adapter stays read-only
# --------------------------------------------------------------------------
def test_the_adapter_satisfies_the_provider_protocol() -> None:
    assert isinstance(provider(StubCollection()), VectorSearchProvider)


@pytest.mark.parametrize("operation", VECTOR_WRITE_OPERATIONS)
def test_the_adapter_exposes_no_write_operation(operation: str) -> None:
    """The Backend owns every write to the collection."""
    assert not hasattr(provider(StubCollection()), operation)


def test_the_adapter_issues_no_write_command() -> None:
    """Only aggregate is ever called on the collection."""
    import ast
    from pathlib import Path

    from memovo_ai.providers.vector_search import atlas

    source = Path(atlas.__file__ or "").read_text(encoding="utf-8")
    tree = ast.parse(source)
    called = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }

    for write in (
        "insert_one",
        "insert_many",
        "update_one",
        "update_many",
        "replace_one",
        "delete_one",
        "delete_many",
        "bulk_write",
        "find_one_and_update",
        "create_index",
    ):
        assert write not in called


# --------------------------------------------------------------------------
# Connecting
# --------------------------------------------------------------------------
def test_connecting_without_a_uri_fails_loudly() -> None:
    """Better than a provider that answers every query with nothing."""
    with pytest.raises(VectorSearchUnavailableError, match="not configured"):
        MongoAtlasVectorSearchProvider.connect(settings(uri=""))


def test_a_configured_uri_is_recognised() -> None:
    assert settings().is_configured is True
    assert settings(uri="   ").is_configured is False


def test_the_connection_string_is_not_exposed_in_a_repr() -> None:
    """A SecretStr keeps credentials out of logs and tracebacks."""
    rendered = repr(settings())

    assert "hunter2" not in rendered
    assert "pw@cluster" not in rendered


async def test_closing_a_provider_without_a_client_is_harmless() -> None:
    await provider(StubCollection()).aclose()
