"""The read-only vector search boundary and its in-memory implementation.

Phase 09 scope: the interface and the fake's own correctness. Enforcing
isolation through the search service, and the full doc 05 section 8 scenario,
is Phase 10.
"""

import math

import pytest

from memovo_ai.providers.vector_search import (
    VECTOR_WRITE_OPERATIONS,
    FakeVectorSearchProvider,
    VectorRecord,
    VectorSearchProvider,
)
from memovo_ai.retrieval import VectorSearchHit

pytestmark = pytest.mark.unit

QUERY = (1.0, 0.0, 0.0)


def record(
    *,
    user_id: str = "user_123",
    memory_id: str = "memory_456",
    chunk_id: str = "chunk_001",
    chunk_index: int = 0,
    content: str = "MongoDB Vector Search...",
    title: str = "MongoDB Vector Search",
    tags: tuple[str, ...] = ("mongodb",),
    embedding: tuple[float, ...] | None = None,
    score: float | None = None,
) -> VectorRecord:
    if embedding is None and score is None:
        embedding = QUERY
    return VectorRecord(
        user_id=user_id,
        memory_id=memory_id,
        chunk_id=chunk_id,
        chunk_index=chunk_index,
        content=content,
        title=title,
        tags=tags,
        embedding=embedding,
        score=score,
    )


# --------------------------------------------------------------------------
# The interface is read-only
# --------------------------------------------------------------------------
def test_fake_satisfies_the_provider_protocol() -> None:
    assert isinstance(FakeVectorSearchProvider(), VectorSearchProvider)


@pytest.mark.parametrize("operation", VECTOR_WRITE_OPERATIONS)
def test_protocol_exposes_no_write_operation(operation: str) -> None:
    """The Backend owns vector persistence (doc 06, section 4)."""
    assert not hasattr(VectorSearchProvider, operation)


@pytest.mark.parametrize("operation", VECTOR_WRITE_OPERATIONS)
def test_fake_exposes_no_write_operation(operation: str) -> None:
    assert not hasattr(FakeVectorSearchProvider(), operation)


def test_the_only_public_operation_is_search() -> None:
    provider = FakeVectorSearchProvider()
    public = {name for name in dir(provider) if not name.startswith("_")}
    callables = {name for name in public if callable(getattr(provider, name))}

    assert callables == {"search"}
    assert public == {"search", "records", "calls"}


def test_records_cannot_be_replaced_after_construction() -> None:
    """There is no write path into the fake at all."""
    provider = FakeVectorSearchProvider([record()])

    assert isinstance(provider.records, tuple)
    with pytest.raises((AttributeError, TypeError)):
        provider.records[0] = record()  # type: ignore[index]


async def test_search_arguments_are_keyword_only() -> None:
    """`user_id` cannot be omitted or passed by accident of position."""
    provider = FakeVectorSearchProvider()

    with pytest.raises(TypeError):
        await provider.search("user_123", QUERY, 5)  # type: ignore[misc]


# --------------------------------------------------------------------------
# The user pre-filter
# --------------------------------------------------------------------------
async def test_only_the_requesting_users_chunks_are_returned() -> None:
    provider = FakeVectorSearchProvider(
        [
            record(user_id="user_a", memory_id="memory_a", chunk_id="chunk_a"),
            record(user_id="user_b", memory_id="memory_b", chunk_id="chunk_b"),
        ]
    )

    hits = await provider.search(user_id="user_b", query_embedding=QUERY, top_k=5)

    assert [hit.memory_id for hit in hits] == ["memory_b"]


async def test_filtering_happens_before_ranking_not_after() -> None:
    """A higher-scoring foreign chunk must not consume a Top-K slot.

    This is the ordering that makes post-filtering detectable: with
    ``top_k=1``, a provider that ranked first and filtered second would
    return nothing at all.
    """
    provider = FakeVectorSearchProvider(
        [
            record(user_id="user_a", memory_id="memory_a", chunk_id="chunk_a", score=0.99),
            record(user_id="user_b", memory_id="memory_b", chunk_id="chunk_b", score=0.90),
        ]
    )

    hits = await provider.search(user_id="user_b", query_embedding=QUERY, top_k=1)

    assert [hit.memory_id for hit in hits] == ["memory_b"]
    assert hits[0].score == pytest.approx(0.90)


async def test_an_unknown_user_gets_no_hits() -> None:
    provider = FakeVectorSearchProvider([record(user_id="user_a")])

    assert await provider.search(user_id="user_zzz", query_embedding=QUERY, top_k=5) == []


async def test_user_ids_match_exactly() -> None:
    """No prefix or case-insensitive matching."""
    provider = FakeVectorSearchProvider([record(user_id="user_1")])

    assert await provider.search(user_id="user_12", query_embedding=QUERY, top_k=5) == []
    assert await provider.search(user_id="USER_1", query_embedding=QUERY, top_k=5) == []


async def test_an_empty_index_returns_no_hits() -> None:
    provider = FakeVectorSearchProvider()

    assert await provider.search(user_id="user_123", query_embedding=QUERY, top_k=5) == []


# --------------------------------------------------------------------------
# Recorded calls
# --------------------------------------------------------------------------
async def test_the_call_is_recorded_for_assertions() -> None:
    provider = FakeVectorSearchProvider()

    await provider.search(user_id="user_123", query_embedding=QUERY, top_k=5)

    assert len(provider.calls) == 1
    assert provider.calls[0].user_id == "user_123"
    assert provider.calls[0].top_k == 5
    assert provider.calls[0].query_embedding == QUERY


async def test_repeated_searches_accumulate_calls() -> None:
    provider = FakeVectorSearchProvider()

    await provider.search(user_id="a", query_embedding=QUERY, top_k=5)
    await provider.search(user_id="b", query_embedding=QUERY, top_k=3)

    assert [call.user_id for call in provider.calls] == ["a", "b"]


# --------------------------------------------------------------------------
# Ranking and Top-K
# --------------------------------------------------------------------------
async def test_hits_are_ordered_by_descending_score() -> None:
    provider = FakeVectorSearchProvider(
        [
            record(chunk_id="chunk_low", score=0.10),
            record(chunk_id="chunk_high", score=0.95),
            record(chunk_id="chunk_mid", score=0.50),
        ]
    )

    hits = await provider.search(user_id="user_123", query_embedding=QUERY, top_k=5)

    assert [hit.chunk_id for hit in hits] == ["chunk_high", "chunk_mid", "chunk_low"]


async def test_top_k_truncates_to_the_best_matches() -> None:
    provider = FakeVectorSearchProvider(
        [record(chunk_id=f"chunk_{n}", score=n / 10) for n in range(10)]
    )

    hits = await provider.search(user_id="user_123", query_embedding=QUERY, top_k=5)

    assert len(hits) == 5
    assert [hit.chunk_id for hit in hits] == ["chunk_9", "chunk_8", "chunk_7", "chunk_6", "chunk_5"]


async def test_fewer_records_than_top_k_is_fine() -> None:
    provider = FakeVectorSearchProvider([record()])

    assert len(await provider.search(user_id="user_123", query_embedding=QUERY, top_k=5)) == 1


async def test_equal_scores_order_deterministically() -> None:
    """Insertion order must not decide the result."""
    forward = FakeVectorSearchProvider(
        [record(chunk_id="chunk_b", score=0.8), record(chunk_id="chunk_a", score=0.8)]
    )
    reverse = FakeVectorSearchProvider(
        [record(chunk_id="chunk_a", score=0.8), record(chunk_id="chunk_b", score=0.8)]
    )

    first = await forward.search(user_id="user_123", query_embedding=QUERY, top_k=5)
    second = await reverse.search(user_id="user_123", query_embedding=QUERY, top_k=5)

    expected = ["chunk_a", "chunk_b"]
    assert [hit.chunk_id for hit in first] == expected
    assert [hit.chunk_id for hit in second] == expected


async def test_no_similarity_threshold_is_applied() -> None:
    """Score filtering is Phase 11; a provider that dropped hits would make
    the 0.75 threshold untestable."""
    provider = FakeVectorSearchProvider([record(chunk_id="chunk_weak", score=0.01)])

    hits = await provider.search(user_id="user_123", query_embedding=QUERY, top_k=5)

    assert [hit.score for hit in hits] == pytest.approx([0.01])


@pytest.mark.parametrize("top_k", [0, -1])
async def test_non_positive_top_k_is_rejected(top_k: int) -> None:
    with pytest.raises(ValueError, match="top_k"):
        await FakeVectorSearchProvider().search(
            user_id="user_123", query_embedding=QUERY, top_k=top_k
        )


# --------------------------------------------------------------------------
# Similarity
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("embedding", "expected"),
    [
        ((1.0, 0.0, 0.0), 1.0),
        ((0.0, 1.0, 0.0), 0.0),
        ((-1.0, 0.0, 0.0), -1.0),
        ((2.0, 0.0, 0.0), 1.0),
    ],
)
async def test_cosine_similarity_is_computed_from_embeddings(
    embedding: tuple[float, ...], expected: float
) -> None:
    provider = FakeVectorSearchProvider([record(embedding=embedding)])

    hits = await provider.search(user_id="user_123", query_embedding=QUERY, top_k=5)

    assert hits[0].score == pytest.approx(expected)


async def test_a_zero_vector_scores_zero_rather_than_dividing_by_zero() -> None:
    provider = FakeVectorSearchProvider([record(embedding=(0.0, 0.0, 0.0))])

    hits = await provider.search(user_id="user_123", query_embedding=QUERY, top_k=5)

    assert hits[0].score == 0.0


async def test_a_fixed_score_overrides_similarity() -> None:
    provider = FakeVectorSearchProvider([record(embedding=(0.0, 1.0, 0.0), score=0.77)])

    hits = await provider.search(user_id="user_123", query_embedding=QUERY, top_k=5)

    assert hits[0].score == pytest.approx(0.77)


async def test_dimension_mismatch_is_reported() -> None:
    provider = FakeVectorSearchProvider([record(embedding=(1.0, 0.0))])

    with pytest.raises(ValueError, match="dimension mismatch"):
        await provider.search(user_id="user_123", query_embedding=QUERY, top_k=5)


# --------------------------------------------------------------------------
# Hit payload
# --------------------------------------------------------------------------
async def test_hits_carry_everything_a_search_result_needs() -> None:
    """The AI Service never reads MongoDB, so title and tags travel with the hit."""
    provider = FakeVectorSearchProvider(
        [
            record(
                memory_id="memory_456",
                chunk_id="chunk_001",
                chunk_index=2,
                content="Vector indexes are used...",
                title="MongoDB Vector Search",
                tags=("mongodb", "vector-search"),
            )
        ]
    )

    hit = (await provider.search(user_id="user_123", query_embedding=QUERY, top_k=5))[0]

    assert hit.memory_id == "memory_456"
    assert hit.chunk_id == "chunk_001"
    assert hit.chunk_index == 2
    assert hit.content == "Vector indexes are used..."
    assert hit.title == "MongoDB Vector Search"
    assert hit.tags == ("mongodb", "vector-search")


async def test_hits_do_not_carry_embeddings() -> None:
    """Search results never expose vectors."""
    provider = FakeVectorSearchProvider([record()])

    hit = (await provider.search(user_id="user_123", query_embedding=QUERY, top_k=5))[0]

    assert not hasattr(hit, "embedding")


async def test_hits_are_immutable() -> None:
    provider = FakeVectorSearchProvider([record()])

    hit = (await provider.search(user_id="user_123", query_embedding=QUERY, top_k=5))[0]

    with pytest.raises(AttributeError):
        hit.score = 1.0  # type: ignore[misc]


# --------------------------------------------------------------------------
# Model validation
# --------------------------------------------------------------------------
def test_a_record_needs_an_embedding_or_a_score() -> None:
    with pytest.raises(ValueError, match="embedding or a fixed score"):
        VectorRecord(
            user_id="user_123",
            memory_id="memory_456",
            chunk_id="chunk_001",
            chunk_index=0,
            content="text",
        )


def test_a_record_rejects_a_negative_chunk_index() -> None:
    with pytest.raises(ValueError, match="must not be negative"):
        record(chunk_index=-1)


def test_a_hit_rejects_a_negative_chunk_index() -> None:
    with pytest.raises(ValueError, match="must not be negative"):
        VectorSearchHit(
            user_id="u",
            memory_id="m",
            chunk_id="c",
            chunk_index=-1,
            content="x",
            score=0.9,
            title="t",
        )


@pytest.mark.parametrize("score", [math.nan, math.inf, -math.inf])
def test_a_hit_rejects_a_non_finite_score(score: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        VectorSearchHit(
            user_id="u",
            memory_id="m",
            chunk_id="c",
            chunk_index=0,
            content="x",
            score=score,
            title="t",
        )


@pytest.mark.parametrize("score", [-1.0, 0.0, 0.75, 1.0, 1.5])
def test_a_hit_does_not_constrain_the_score_range(score: float) -> None:
    """The similarity metric is not locked (doc 01, section 7)."""
    hit = VectorSearchHit(
        user_id="u", memory_id="m", chunk_id="c", chunk_index=0, content="x", score=score, title="t"
    )

    assert hit.score == score


def test_tags_default_to_empty() -> None:
    hit = VectorSearchHit(
        user_id="u", memory_id="m", chunk_id="c", chunk_index=0, content="x", score=0.9, title="t"
    )

    assert hit.tags == ()
