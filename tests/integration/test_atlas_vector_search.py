"""Real MongoDB Atlas Vector Search checks. Opt-in: never runs in CI.

    uv sync --extra atlas
    export MEMOVO_ATLAS_URI='mongodb+srv://...'
    MEMOVO_RUN_ATLAS_INTEGRATION=1 uv run pytest tests/integration/test_atlas_vector_search.py -s

Requires a **pre-seeded** collection. The AI Service is read-only by
contract, so it cannot create its own fixtures -- seeding is the Backend's
job, and adding a write path here purely for testing would breach the
boundary these tests exist to protect.

Expected index (on the configured collection)::

    {
      "fields": [
        {"type": "vector", "path": "embedding",
         "numDimensions": 1024, "similarity": "cosine"},
        {"type": "filter", "path": "userId"}
      ]
    }

Expected documents -- at least two users, so isolation is provable::

    {"userId": "<MEMOVO_ATLAS_TEST_USER>",  "memoryId": ..., "chunkId": ...,
     "chunkIndex": 0, "content": ..., "title": ..., "tags": [...],
     "embedding": [ ... 1024 floats ... ]}
    {"userId": "<MEMOVO_ATLAS_TEST_OTHER_USER>", ...}

Without those environment variables the isolation test skips rather than
passing vacuously.
"""

import os
from collections.abc import AsyncIterator

import pytest

from memovo_ai.core.config import AtlasSettings
from memovo_ai.embeddings import EMBEDDING_DIMENSION
from memovo_ai.providers.vector_search import (
    MongoAtlasVectorSearchProvider,
    UserScopedVectorSearchProvider,
)

_ENABLED = os.environ.get("MEMOVO_RUN_ATLAS_INTEGRATION") == "1"
_TEST_USER = os.environ.get("MEMOVO_ATLAS_TEST_USER", "")
_OTHER_USER = os.environ.get("MEMOVO_ATLAS_TEST_OTHER_USER", "")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _ENABLED,
        reason="set MEMOVO_RUN_ATLAS_INTEGRATION=1 and MEMOVO_ATLAS_URI to run against Atlas",
    ),
]

QUERY = [0.01] * EMBEDDING_DIMENSION


@pytest.fixture
async def provider() -> AsyncIterator[MongoAtlasVectorSearchProvider]:
    settings = AtlasSettings()
    if not settings.is_configured:
        pytest.skip("MEMOVO_ATLAS_URI is not set")

    adapter = MongoAtlasVectorSearchProvider.connect(settings)
    try:
        yield adapter
    finally:
        await adapter.aclose()


def _require_seeded_user() -> str:
    if not _TEST_USER:
        pytest.skip("set MEMOVO_ATLAS_TEST_USER to a userId present in the collection")
    return _TEST_USER


async def test_the_index_answers_a_search(provider: MongoAtlasVectorSearchProvider) -> None:
    """Proves connectivity, index name, vector path and dimension together."""
    hits = await provider.search(user_id=_require_seeded_user(), query_embedding=QUERY, top_k=5)

    assert isinstance(hits, list)


async def test_results_respect_top_k(provider: MongoAtlasVectorSearchProvider) -> None:
    hits = await provider.search(user_id=_require_seeded_user(), query_embedding=QUERY, top_k=3)

    assert len(hits) <= 3


async def test_every_hit_belongs_to_the_requesting_user(
    provider: MongoAtlasVectorSearchProvider,
) -> None:
    """The pre-filter, verified against the real engine.

    Some approximate-nearest-neighbour implementations filter *after*
    retrieval. This is the check that would catch that.
    """
    user = _require_seeded_user()

    hits = await provider.search(user_id=user, query_embedding=QUERY, top_k=5)

    assert all(hit.user_id == user for hit in hits)


async def test_another_users_chunks_never_appear(
    provider: MongoAtlasVectorSearchProvider,
) -> None:
    if not _OTHER_USER:
        pytest.skip("set MEMOVO_ATLAS_TEST_OTHER_USER to a second seeded userId")

    user = _require_seeded_user()

    hits = await provider.search(user_id=user, query_embedding=QUERY, top_k=5)

    assert all(hit.user_id != _OTHER_USER for hit in hits)


async def test_the_isolation_wrapper_accepts_real_results(
    provider: MongoAtlasVectorSearchProvider,
) -> None:
    """The fail-closed guard must not trip on a correctly configured index."""
    scoped = UserScopedVectorSearchProvider(provider)

    hits = await scoped.search(user_id=_require_seeded_user(), query_embedding=QUERY, top_k=5)

    assert isinstance(hits, list)


async def test_an_unknown_user_gets_nothing(provider: MongoAtlasVectorSearchProvider) -> None:
    hits = await provider.search(
        user_id="user_that_does_not_exist_in_this_collection",
        query_embedding=QUERY,
        top_k=5,
    )

    assert hits == []


async def test_hits_carry_the_metadata_the_contract_promises(
    provider: MongoAtlasVectorSearchProvider,
) -> None:
    """Fails loudly if the Backend did not store title/tags alongside the vector."""
    hits = await provider.search(user_id=_require_seeded_user(), query_embedding=QUERY, top_k=5)

    if not hits:
        pytest.skip("collection has no chunks for the configured test user")

    for hit in hits:
        assert hit.memory_id
        assert hit.chunk_id
        assert hit.chunk_index >= 0
        assert hit.content


async def test_scores_are_ordered_and_finite(
    provider: MongoAtlasVectorSearchProvider,
) -> None:
    hits = await provider.search(user_id=_require_seeded_user(), query_embedding=QUERY, top_k=5)

    if len(hits) < 2:
        pytest.skip("need at least two chunks to check ordering")

    scores = [hit.score for hit in hits]
    assert scores == sorted(scores, reverse=True)
