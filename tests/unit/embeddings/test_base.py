"""The embedding provider boundary and its validating wrapper."""

import math
from collections.abc import Sequence

import pytest

from memovo_ai.embeddings import (
    EMBEDDING_DIMENSION,
    Embedding,
    EmbeddingProvider,
    InvalidEmbeddingError,
    ValidatedEmbeddingProvider,
)

pytestmark = pytest.mark.unit


def vector(dimension: int = EMBEDDING_DIMENSION, value: float = 0.1) -> list[float]:
    return [value] * dimension


class RecordingProvider:
    """A well-behaved provider that records what it was asked to embed."""

    def __init__(self, *, dimension: int = EMBEDDING_DIMENSION) -> None:
        self.dimension = dimension
        self.document_calls: list[list[str]] = []
        self.query_calls: list[str] = []

    async def embed_documents(self, texts: Sequence[str]) -> list[Embedding]:
        self.document_calls.append(list(texts))
        return [vector(self.dimension, value=0.1 * (index + 1)) for index in range(len(texts))]

    async def embed_query(self, query: str) -> Embedding:
        self.query_calls.append(query)
        return vector(self.dimension)


class BrokenProvider:
    """A provider that returns whatever it was constructed with."""

    def __init__(self, *, documents: object = None, query: object = None) -> None:
        self._documents = documents
        self._query = query

    async def embed_documents(self, texts: Sequence[str]) -> list[Embedding]:
        return self._documents  # type: ignore[return-value]

    async def embed_query(self, query: str) -> Embedding:
        return self._query  # type: ignore[return-value]


class UnavailableProvider:
    """Stands in for a model that failed to load or crashed mid-inference."""

    async def embed_documents(self, texts: Sequence[str]) -> list[Embedding]:
        message = "model runtime is unavailable"
        raise RuntimeError(message)

    async def embed_query(self, query: str) -> Embedding:
        message = "inference failed"
        raise RuntimeError(message)


# --------------------------------------------------------------------------
# The protocol
# --------------------------------------------------------------------------
def test_a_conforming_object_satisfies_the_protocol() -> None:
    assert isinstance(RecordingProvider(), EmbeddingProvider)


def test_the_validating_wrapper_is_itself_a_provider() -> None:
    """It can be substituted anywhere a provider is expected."""
    assert isinstance(ValidatedEmbeddingProvider(RecordingProvider()), EmbeddingProvider)


def test_an_object_missing_a_method_does_not_satisfy_the_protocol() -> None:
    class QueryOnly:
        async def embed_query(self, query: str) -> Embedding:
            return vector()

    assert not isinstance(QueryOnly(), EmbeddingProvider)


def test_document_and_query_embedding_stay_separate_methods() -> None:
    """Doc 02, section 6: keep them distinct even if internals are shared."""
    assert hasattr(EmbeddingProvider, "embed_documents")
    assert hasattr(EmbeddingProvider, "embed_query")


# --------------------------------------------------------------------------
# Happy path
# --------------------------------------------------------------------------
async def test_document_embedding_returns_one_vector_per_input() -> None:
    provider = ValidatedEmbeddingProvider(RecordingProvider())

    embeddings = await provider.embed_documents(["first chunk", "second chunk"])

    assert len(embeddings) == 2
    assert all(len(embedding) == EMBEDDING_DIMENSION for embedding in embeddings)


async def test_query_embedding_returns_one_vector() -> None:
    provider = ValidatedEmbeddingProvider(RecordingProvider())

    embedding = await provider.embed_query("What did I save about vectors?")

    assert len(embedding) == EMBEDDING_DIMENSION


async def test_batch_embedding_preserves_input_order() -> None:
    inner = RecordingProvider()
    provider = ValidatedEmbeddingProvider(inner)

    embeddings = await provider.embed_documents(["a", "b", "c"])

    assert inner.document_calls == [["a", "b", "c"]]
    assert [embedding[0] for embedding in embeddings] == pytest.approx([0.1, 0.2, 0.3])


async def test_texts_are_passed_through_untouched() -> None:
    inner = RecordingProvider()

    await ValidatedEmbeddingProvider(inner).embed_documents(["بحث المتجهات", "日本語"])

    assert inner.document_calls == [["بحث المتجهات", "日本語"]]


async def test_an_empty_document_batch_is_allowed() -> None:
    provider = ValidatedEmbeddingProvider(RecordingProvider())

    assert await provider.embed_documents([]) == []


async def test_a_tuple_of_texts_is_accepted() -> None:
    provider = ValidatedEmbeddingProvider(RecordingProvider())

    assert len(await provider.embed_documents(("a", "b"))) == 2


# --------------------------------------------------------------------------
# Validation at the boundary
# --------------------------------------------------------------------------
async def test_wrong_dimension_from_documents_is_rejected() -> None:
    provider = ValidatedEmbeddingProvider(RecordingProvider(dimension=512))

    with pytest.raises(InvalidEmbeddingError, match="dimension 512"):
        await provider.embed_documents(["chunk"])


async def test_wrong_dimension_from_a_query_is_rejected() -> None:
    provider = ValidatedEmbeddingProvider(RecordingProvider(dimension=768))

    with pytest.raises(InvalidEmbeddingError, match="dimension 768"):
        await provider.embed_query("query")


async def test_a_short_batch_is_rejected() -> None:
    """A provider that silently drops rows would misalign chunks."""
    provider = ValidatedEmbeddingProvider(BrokenProvider(documents=[vector()]))

    with pytest.raises(InvalidEmbeddingError, match="expected 2 embeddings, got 1"):
        await provider.embed_documents(["a", "b"])


async def test_an_over_long_batch_is_rejected() -> None:
    provider = ValidatedEmbeddingProvider(BrokenProvider(documents=[vector(), vector()]))

    with pytest.raises(InvalidEmbeddingError, match="expected 1 embeddings, got 2"):
        await provider.embed_documents(["a"])


async def test_empty_model_output_is_rejected() -> None:
    provider = ValidatedEmbeddingProvider(BrokenProvider(query=[]))

    with pytest.raises(InvalidEmbeddingError, match="empty"):
        await provider.embed_query("query")


async def test_nan_from_a_provider_is_rejected() -> None:
    corrupted = [math.nan, *vector(EMBEDDING_DIMENSION - 1)]
    provider = ValidatedEmbeddingProvider(BrokenProvider(query=corrupted))

    with pytest.raises(InvalidEmbeddingError, match="NaN"):
        await provider.embed_query("query")


async def test_infinity_from_a_provider_is_rejected() -> None:
    corrupted = [math.inf, *vector(EMBEDDING_DIMENSION - 1)]
    provider = ValidatedEmbeddingProvider(BrokenProvider(documents=[corrupted]))

    with pytest.raises(InvalidEmbeddingError, match="infinity"):
        await provider.embed_documents(["a"])


@pytest.mark.parametrize("payload", [None, "text", {"embedding": [0.1]}])
async def test_non_sequence_output_is_rejected(payload: object) -> None:
    provider = ValidatedEmbeddingProvider(BrokenProvider(query=payload))

    with pytest.raises(InvalidEmbeddingError, match="sequence"):
        await provider.embed_query("query")


# --------------------------------------------------------------------------
# Configuration passthrough
# --------------------------------------------------------------------------
async def test_the_wrapper_honours_a_custom_dimension() -> None:
    provider = ValidatedEmbeddingProvider(RecordingProvider(dimension=8), dimension=8)

    assert len(await provider.embed_query("query")) == 8


async def test_the_wrapper_can_require_unit_norm() -> None:
    provider = ValidatedEmbeddingProvider(RecordingProvider(), expect_unit_norm=True)

    with pytest.raises(InvalidEmbeddingError, match="unit-normalized"):
        await provider.embed_query("query")


async def test_unit_norm_is_off_by_default() -> None:
    provider = ValidatedEmbeddingProvider(RecordingProvider())

    assert await provider.embed_query("query")


# --------------------------------------------------------------------------
# Provider failures pass through untouched
# --------------------------------------------------------------------------
async def test_provider_failures_are_not_swallowed() -> None:
    """Mapping these onto MODEL_UNAVAILABLE / EMBEDDING_FAILED is Phase 15.

    The wrapper must not convert them into a valid-looking empty result.
    """
    provider = ValidatedEmbeddingProvider(UnavailableProvider())

    with pytest.raises(RuntimeError, match="unavailable"):
        await provider.embed_documents(["a"])

    with pytest.raises(RuntimeError, match="inference failed"):
        await provider.embed_query("query")


# --------------------------------------------------------------------------
# Scope
# --------------------------------------------------------------------------
def test_no_model_runtime_is_imported() -> None:
    """Phase 05 stays model-free; Qwen3 arrives at Phase 06."""
    import sys

    for module in ("torch", "transformers", "sentence_transformers"):
        assert module not in sys.modules
