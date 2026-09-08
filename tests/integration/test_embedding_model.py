"""Real Qwen3 model checks. Opt-in: never runs in CI.

These load actual weights, so they are skipped unless explicitly enabled::

    uv sync --extra embeddings
    MEMOVO_RUN_EMBEDDING_INTEGRATION=1 uv run pytest tests/integration -s

The first run downloads roughly 1.2 GB. Nothing else in the suite touches the
network or the model cache.

Their job is to answer the questions unit tests cannot: does the model return
1024 dimensions, is its output already unit-normalized, and does it define the
query instruction prompt. The normalization decision is made from this
evidence before production vector indexing.
"""

import os

import pytest

from memovo_ai.core.config import EmbeddingSettings
from memovo_ai.embeddings import (
    EMBEDDING_DIMENSION,
    Qwen3EmbeddingProvider,
    ValidatedEmbeddingProvider,
    l2_norm,
)

_ENABLED = os.environ.get("MEMOVO_RUN_EMBEDDING_INTEGRATION") == "1"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _ENABLED,
        reason="set MEMOVO_RUN_EMBEDDING_INTEGRATION=1 to load real model weights",
    ),
]

DOCUMENTS = [
    "Title:\nMongoDB Vector Search\n\nContent:\nAtlas Vector Search indexes embeddings.",
    "Title:\nSourdough starter\n\nContent:\nFeed the starter with flour and water daily.",
]


@pytest.fixture(scope="module")
def provider() -> Qwen3EmbeddingProvider:
    """Loaded once for the module, mirroring the startup-time lifecycle."""
    return Qwen3EmbeddingProvider.load(EmbeddingSettings())


async def test_documents_embed_at_the_locked_dimension(provider: Qwen3EmbeddingProvider) -> None:
    embeddings = await provider.embed_documents(DOCUMENTS)

    assert len(embeddings) == len(DOCUMENTS)
    assert all(len(embedding) == EMBEDDING_DIMENSION for embedding in embeddings)


async def test_query_embeds_at_the_locked_dimension(provider: Qwen3EmbeddingProvider) -> None:
    embedding = await provider.embed_query("What did I save about vector search?")

    assert len(embedding) == EMBEDDING_DIMENSION


async def test_output_passes_full_validation(provider: Qwen3EmbeddingProvider) -> None:
    """Real output must satisfy the same invariants as any other provider."""
    validated = ValidatedEmbeddingProvider(provider)

    assert len(await validated.embed_documents(DOCUMENTS)) == len(DOCUMENTS)


async def test_embedding_is_deterministic(provider: Qwen3EmbeddingProvider) -> None:
    """Reprocessing must not shift vectors for unchanged content."""
    first = await provider.embed_query("What did I save about vector search?")
    second = await provider.embed_query("What did I save about vector search?")

    assert first == pytest.approx(second, abs=1e-6)


async def test_report_output_normalization(provider: Qwen3EmbeddingProvider) -> None:
    """Measures the L2 norm so the normalization decision rests on evidence.

    Deliberately does not assert unit norm: doc 01, section 7 recommends but
    does not lock cosine normalization. If the norms below are ~1.0, enabling
    `expect_unit_norm` is safe and the 0.75 threshold is metric-consistent.
    """
    embeddings = await provider.embed_documents(DOCUMENTS)
    norms = [l2_norm(embedding) for embedding in embeddings]

    print(f"\n[normalization] model={provider.settings.model}")
    print(f"[normalization] L2 norms: {[round(norm, 6) for norm in norms]}")
    print(f"[normalization] unit-normalized: {all(abs(n - 1.0) < 1e-3 for n in norms)}")

    assert all(norm > 0 for norm in norms)


async def test_report_query_prompt_availability(provider: Qwen3EmbeddingProvider) -> None:
    """Records whether the model defines the query instruction prompt.

    `query_prompt_name` currently defaults to empty. If this reports that a
    'query' prompt exists, the default can be flipped with evidence behind it.
    """
    prompts = getattr(provider._encoder, "prompts", None)

    print(f"\n[query-prompt] available prompts: {prompts}")
    print(f"[query-prompt] 'query' defined: {bool(prompts) and 'query' in prompts}")


async def test_relevant_document_scores_above_an_unrelated_one(
    provider: Qwen3EmbeddingProvider,
) -> None:
    """A retrieval sanity check: the right memory must win on cosine similarity."""
    query = await provider.embed_query("What did I save about MongoDB vector search?")
    relevant, unrelated = await provider.embed_documents(DOCUMENTS)

    def cosine(left: list[float], right: list[float]) -> float:
        dot = sum(a * b for a, b in zip(left, right, strict=True))
        return dot / (l2_norm(left) * l2_norm(right))

    relevant_score = cosine(query, relevant)
    unrelated_score = cosine(query, unrelated)

    print(f"\n[similarity] relevant={relevant_score:.4f} unrelated={unrelated_score:.4f}")

    assert relevant_score > unrelated_score
