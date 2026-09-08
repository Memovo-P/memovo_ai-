"""Cross-user isolation. The critical security invariant of Sprint 1.

Doc 05 section 8 and doc 03 Phase 10 define the required scenario. Doc 06
section 3 forbids satisfying it by post-filtering, so these tests check the
*mechanism* -- that the pre-filter reaches the provider and that a foreign hit
is refused rather than quietly dropped -- not merely the visible output.
"""

import pytest

from memovo_ai.providers.vector_search import (
    VECTOR_WRITE_OPERATIONS,
    FakeVectorSearchProvider,
    UserIsolationError,
    UserScopedVectorSearchProvider,
    VectorRecord,
    VectorSearchProvider,
    validate_user_scope,
)
from memovo_ai.retrieval import VectorSearchHit

pytestmark = pytest.mark.unit

QUERY = (1.0, 0.0, 0.0)

USER_A = "user_a"
USER_B = "user_b"


def index() -> FakeVectorSearchProvider:
    """The doc 05 section 8 dataset.

    User A owns Memory X scoring 0.99; User B owns Memory Y scoring 0.90.
    A's chunk is the better match, so any implementation that ranks before it
    filters will surface it.
    """
    return FakeVectorSearchProvider(
        [
            VectorRecord(
                user_id=USER_A,
                memory_id="memory_x",
                chunk_id="chunk_x",
                chunk_index=0,
                content="User A private content",
                title="Memory X",
                tags=("private",),
                score=0.99,
            ),
            VectorRecord(
                user_id=USER_B,
                memory_id="memory_y",
                chunk_id="chunk_y",
                chunk_index=0,
                content="User B content",
                title="Memory Y",
                tags=("mine",),
                score=0.90,
            ),
        ]
    )


def scoped() -> UserScopedVectorSearchProvider:
    return UserScopedVectorSearchProvider(index())


class LeakingProvider:
    """A provider whose pre-filter is broken: it ignores `user_id` entirely.

    Stands in for a misconfigured production adapter -- a filter omitted from
    the query, or applied to the wrong metadata field.
    """

    def __init__(self, inner: FakeVectorSearchProvider) -> None:
        self._inner = inner

    async def search(
        self, *, user_id: str, query_embedding: tuple[float, ...], top_k: int
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
            for record in self._inner.records
        ][:top_k]


# --------------------------------------------------------------------------
# The required scenario (doc 05, section 8)
# --------------------------------------------------------------------------
async def test_user_a_memory_is_never_visible_to_user_b() -> None:
    hits = await scoped().search(user_id=USER_B, query_embedding=QUERY, top_k=5)

    assert [hit.memory_id for hit in hits] == ["memory_y"]
    assert all(hit.user_id == USER_B for hit in hits)


async def test_user_b_may_see_their_own_memory() -> None:
    hits = await scoped().search(user_id=USER_B, query_embedding=QUERY, top_k=5)

    assert hits[0].memory_id == "memory_y"
    assert hits[0].score == pytest.approx(0.90)


async def test_the_higher_scoring_foreign_chunk_never_consumes_a_slot() -> None:
    """With `top_k=1`, rank-then-filter would return nothing at all."""
    hits = await scoped().search(user_id=USER_B, query_embedding=QUERY, top_k=1)

    assert len(hits) == 1
    assert hits[0].memory_id == "memory_y"


async def test_isolation_holds_in_the_other_direction() -> None:
    hits = await scoped().search(user_id=USER_A, query_embedding=QUERY, top_k=5)

    assert [hit.memory_id for hit in hits] == ["memory_x"]


async def test_no_foreign_content_appears_anywhere_in_the_results() -> None:
    hits = await scoped().search(user_id=USER_B, query_embedding=QUERY, top_k=5)
    payload = " ".join(f"{hit.memory_id}{hit.title}{hit.content}{hit.tags}" for hit in hits)

    assert "User A private content" not in payload
    assert "memory_x" not in payload
    assert "Memory X" not in payload


# --------------------------------------------------------------------------
# The mechanism: the pre-filter reaches the provider
# --------------------------------------------------------------------------
async def test_the_trusted_user_id_is_passed_down_to_the_engine() -> None:
    """Isolation must be a query parameter, not an afterthought."""
    inner = index()

    await UserScopedVectorSearchProvider(inner).search(
        user_id=USER_B, query_embedding=QUERY, top_k=5
    )

    assert [call.user_id for call in inner.calls] == [USER_B]


async def test_the_user_id_reaches_the_engine_unmodified() -> None:
    """Trimming or normalizing an identity would search as somebody else."""
    inner = index()
    identifier = "  user_b  "

    await UserScopedVectorSearchProvider(inner).search(
        user_id=identifier, query_embedding=QUERY, top_k=5
    )

    assert inner.calls[0].user_id == identifier


async def test_top_k_is_not_inflated_to_compensate_for_filtering() -> None:
    """Over-fetching then trimming is post-filtering with extra steps."""
    inner = index()

    await UserScopedVectorSearchProvider(inner).search(
        user_id=USER_B, query_embedding=QUERY, top_k=5
    )

    assert inner.calls[0].top_k == 5


# --------------------------------------------------------------------------
# Fail closed on a broken pre-filter
# --------------------------------------------------------------------------
async def test_a_leaking_provider_is_refused_not_silently_filtered() -> None:
    """A foreign hit means the pre-filter is broken; that must surface."""
    provider = UserScopedVectorSearchProvider(LeakingProvider(index()))

    with pytest.raises(UserIsolationError, match="pre-filter is not being applied"):
        await provider.search(user_id=USER_B, query_embedding=QUERY, top_k=5)


async def test_the_leak_error_reveals_no_identifiers_or_content() -> None:
    """The message reaches logs; CLAUDE.md forbids logging chunk text."""
    provider = UserScopedVectorSearchProvider(LeakingProvider(index()))

    with pytest.raises(UserIsolationError) as raised:
        await provider.search(user_id=USER_B, query_embedding=QUERY, top_k=5)

    message = str(raised.value)
    for secret in (USER_A, "memory_x", "Memory X", "User A private content"):
        assert secret not in message


async def test_a_correct_provider_passes_verification_untouched() -> None:
    inner = index()
    direct = await inner.search(user_id=USER_B, query_embedding=QUERY, top_k=5)
    verified = await UserScopedVectorSearchProvider(index()).search(
        user_id=USER_B, query_embedding=QUERY, top_k=5
    )

    assert direct == verified


# --------------------------------------------------------------------------
# A blank scope is refused
# --------------------------------------------------------------------------
@pytest.mark.parametrize("user_id", ["", " ", "\t", "\n", "   \n  "])
async def test_a_blank_user_id_is_refused_before_searching(user_id: str) -> None:
    """An empty filter can match every row in a shared collection."""
    inner = index()

    with pytest.raises(ValueError, match="non-blank"):
        await UserScopedVectorSearchProvider(inner).search(
            user_id=user_id, query_embedding=QUERY, top_k=5
        )

    assert inner.calls == []


@pytest.mark.parametrize("user_id", [None, 123, [], {}])
def test_a_non_string_user_id_is_refused(user_id: object) -> None:
    with pytest.raises(ValueError, match="non-blank"):
        validate_user_scope(user_id)  # type: ignore[arg-type]


def test_a_valid_user_id_is_returned_verbatim() -> None:
    assert validate_user_scope("user_123") == "user_123"
    assert validate_user_scope(" user_123 ") == " user_123 "


# --------------------------------------------------------------------------
# The wrapper stays within the read-only contract
# --------------------------------------------------------------------------
def test_scoped_provider_satisfies_the_protocol() -> None:
    assert isinstance(scoped(), VectorSearchProvider)


@pytest.mark.parametrize("operation", VECTOR_WRITE_OPERATIONS)
def test_scoped_provider_exposes_no_write_operation(operation: str) -> None:
    assert not hasattr(scoped(), operation)


def test_scoped_provider_exposes_only_search() -> None:
    provider = scoped()
    public = {name for name in dir(provider) if not name.startswith("_")}

    assert public == {"search"}


# --------------------------------------------------------------------------
# Isolation across many users
# --------------------------------------------------------------------------
async def test_each_user_sees_only_their_own_chunks() -> None:
    records = [
        VectorRecord(
            user_id=f"user_{n}",
            memory_id=f"memory_{n}",
            chunk_id=f"chunk_{n}",
            chunk_index=0,
            content="identical content across every user",
            title="Same Title",
            score=0.9,
        )
        for n in range(25)
    ]
    provider = UserScopedVectorSearchProvider(FakeVectorSearchProvider(records))

    for n in range(25):
        hits = await provider.search(user_id=f"user_{n}", query_embedding=QUERY, top_k=5)
        assert [hit.memory_id for hit in hits] == [f"memory_{n}"]


async def test_identical_content_does_not_merge_across_users() -> None:
    """Shared collection, same text, same score: still two separate owners."""
    shared = [
        VectorRecord(
            user_id=owner,
            memory_id=f"memory_{owner}",
            chunk_id=f"chunk_{owner}",
            chunk_index=0,
            content="MongoDB Vector Search...",
            title="MongoDB Vector Search",
            score=0.95,
        )
        for owner in (USER_A, USER_B)
    ]
    provider = UserScopedVectorSearchProvider(FakeVectorSearchProvider(shared))

    hits = await provider.search(user_id=USER_A, query_embedding=QUERY, top_k=5)

    assert len(hits) == 1
    assert hits[0].user_id == USER_A


# --------------------------------------------------------------------------
# The owner never reaches the public contract
# --------------------------------------------------------------------------
def test_the_search_result_schema_has_no_user_field() -> None:
    """`user_id` exists to verify isolation, never to be serialized."""
    from memovo_ai.schemas import SearchResult, SearchResultChunk

    assert "user_id" not in SearchResult.model_fields
    assert "user_id" not in SearchResultChunk.model_fields
    assert "userId" not in SearchResult.model_json_schema()["properties"]


def test_the_search_result_schema_rejects_a_user_field() -> None:
    from pydantic import ValidationError

    from memovo_ai.schemas import SearchResult

    with pytest.raises(ValidationError):
        SearchResult.model_validate(
            {
                "memoryId": "memory_456",
                "score": 0.94,
                "title": "t",
                "tags": [],
                "chunks": [],
                "userId": "user_a",
            }
        )
