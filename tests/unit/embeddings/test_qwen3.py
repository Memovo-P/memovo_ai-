"""Qwen3 provider behaviour, exercised against a fake encoder.

No model is loaded and no weights are downloaded here. The real model is
covered by ``tests/integration/test_embedding_model.py``, which is opt-in.
"""

import asyncio
import sys
import threading

import pytest

from memovo_ai.core.config import DEFAULT_EMBEDDING_MODEL, EmbeddingSettings
from memovo_ai.embeddings import (
    EMBEDDING_DIMENSION,
    EmbeddingInferenceError,
    EmbeddingProvider,
    InvalidEmbeddingError,
    ModelUnavailableError,
    Qwen3EmbeddingProvider,
    ValidatedEmbeddingProvider,
)

pytestmark = pytest.mark.unit


def vector(dimension: int = EMBEDDING_DIMENSION, value: float = 0.1) -> list[float]:
    return [value] * dimension


class FakeEncoder:
    """Stands in for ``SentenceTransformer``, recording how it was called."""

    def __init__(self, *, dimension: int = EMBEDDING_DIMENSION) -> None:
        self.dimension = dimension
        self.calls: list[tuple[list[str], dict[str, object]]] = []
        self.threads: list[str] = []

    def encode(self, sentences: list[str], **kwargs: object) -> object:
        self.calls.append((list(sentences), dict(kwargs)))
        self.threads.append(threading.current_thread().name)
        return [vector(self.dimension, value=0.1 * (index + 1)) for index in range(len(sentences))]


class FakeArray(list[object]):
    """Mimics a numpy array: a sequence that also exposes ``tolist``."""

    def tolist(self) -> list[object]:
        return list(self)


class ArrayEncoder(FakeEncoder):
    def encode(self, sentences: list[str], **kwargs: object) -> object:
        rows = super().encode(sentences, **kwargs)
        return FakeArray(rows)  # type: ignore[arg-type]


class ExplodingEncoder:
    def __init__(self, error: Exception) -> None:
        self._error = error

    def encode(self, sentences: list[str], **kwargs: object) -> object:
        raise self._error


def settings(**overrides: object) -> EmbeddingSettings:
    base: dict[str, object] = {
        "model": DEFAULT_EMBEDDING_MODEL,
        "dimension": EMBEDDING_DIMENSION,
        "device": "auto",
        "batch_size": 32,
        "query_prompt_name": "",
    }
    return EmbeddingSettings(**{**base, **overrides})  # type: ignore[arg-type]


def make_provider(encoder: object | None = None, **overrides: object) -> Qwen3EmbeddingProvider:
    return Qwen3EmbeddingProvider(
        encoder if encoder is not None else FakeEncoder(),  # type: ignore[arg-type]
        settings=settings(**overrides),
    )


# --------------------------------------------------------------------------
# Abstraction
# --------------------------------------------------------------------------
def test_provider_satisfies_the_embedding_protocol() -> None:
    assert isinstance(make_provider(), EmbeddingProvider)


def test_no_model_runtime_is_imported_by_unit_tests() -> None:
    """Importing the module must not pull in the model runtime."""
    make_provider()

    for module in ("sentence_transformers", "torch", "transformers"):
        assert module not in sys.modules


def test_model_specifics_do_not_leak_upstream() -> None:
    """Only qwen3.py knows about encoders; the rest of the service sees the protocol."""
    from memovo_ai.embeddings import base

    source = base.__doc__ or ""
    assert "SentenceTransformer" not in source
    assert not hasattr(base, "TextEncoder")


# --------------------------------------------------------------------------
# Documents
# --------------------------------------------------------------------------
async def test_document_embedding_returns_one_vector_per_text() -> None:
    embeddings = await make_provider().embed_documents(["a", "b", "c"])

    assert len(embeddings) == 3
    assert all(len(embedding) == EMBEDDING_DIMENSION for embedding in embeddings)


async def test_document_embedding_preserves_order() -> None:
    encoder = FakeEncoder()

    embeddings = await make_provider(encoder).embed_documents(["a", "b", "c"])

    assert encoder.calls[0][0] == ["a", "b", "c"]
    assert [embedding[0] for embedding in embeddings] == pytest.approx([0.1, 0.2, 0.3])


async def test_empty_document_batch_short_circuits() -> None:
    encoder = FakeEncoder()

    assert await make_provider(encoder).embed_documents([]) == []
    assert encoder.calls == []


async def test_configured_batch_size_is_passed_to_the_encoder() -> None:
    encoder = FakeEncoder()

    await make_provider(encoder, batch_size=8).embed_documents(["a"])

    assert encoder.calls[0][1]["batch_size"] == 8


async def test_default_batch_size_is_32() -> None:
    encoder = FakeEncoder()

    await make_provider(encoder).embed_documents(["a"])

    assert encoder.calls[0][1]["batch_size"] == 32


async def test_numpy_style_output_is_converted() -> None:
    embeddings = await make_provider(ArrayEncoder()).embed_documents(["a", "b"])

    assert isinstance(embeddings, list)
    assert all(isinstance(entry, float) for entry in embeddings[0])


# --------------------------------------------------------------------------
# Queries
# --------------------------------------------------------------------------
async def test_query_embedding_returns_a_single_vector() -> None:
    embedding = await make_provider().embed_query("what did I save?")

    assert len(embedding) == EMBEDDING_DIMENSION


async def test_query_is_embedded_as_one_item() -> None:
    encoder = FakeEncoder()

    await make_provider(encoder).embed_query("query text")

    assert encoder.calls[0][0] == ["query text"]


async def test_no_query_prompt_is_sent_by_default() -> None:
    """Enabling the instruction prompt is deferred until it can be verified."""
    encoder = FakeEncoder()

    await make_provider(encoder).embed_query("query")

    assert "prompt_name" not in encoder.calls[0][1]


async def test_query_prompt_is_sent_when_configured() -> None:
    encoder = FakeEncoder()

    await make_provider(encoder, query_prompt_name="query").embed_query("query")

    assert encoder.calls[0][1]["prompt_name"] == "query"


async def test_documents_never_receive_the_query_prompt() -> None:
    encoder = FakeEncoder()

    await make_provider(encoder, query_prompt_name="query").embed_documents(["chunk"])

    assert "prompt_name" not in encoder.calls[0][1]


# --------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------
async def test_model_output_is_not_normalized() -> None:
    """Preserve raw output; the normalization decision is not locked yet."""
    encoder = FakeEncoder()

    await make_provider(encoder).embed_documents(["a"])

    assert encoder.calls[0][1]["normalize_embeddings"] is False


# --------------------------------------------------------------------------
# Event loop
# --------------------------------------------------------------------------
async def test_encoding_runs_off_the_event_loop() -> None:
    """A compute-bound forward pass must not block concurrent requests."""
    encoder = FakeEncoder()

    await make_provider(encoder).embed_documents(["a"])

    assert encoder.threads[0] != threading.current_thread().name


async def test_concurrent_calls_do_not_serialize_on_the_loop() -> None:
    provider = make_provider()

    results = await asyncio.gather(*(provider.embed_query(f"q{n}") for n in range(4)))

    assert len(results) == 4


# --------------------------------------------------------------------------
# Failure mapping
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "error",
    [RuntimeError("cuda out of memory"), ValueError("bad prompt"), OSError("disk")],
)
async def test_inference_failures_become_embedding_inference_error(error: Exception) -> None:
    provider = make_provider(ExplodingEncoder(error))

    with pytest.raises(EmbeddingInferenceError, match="embedding inference failed"):
        await provider.embed_documents(["a"])


async def test_query_inference_failure_is_mapped() -> None:
    provider = make_provider(ExplodingEncoder(RuntimeError("boom")))

    with pytest.raises(EmbeddingInferenceError):
        await provider.embed_query("query")


async def test_failure_messages_leak_no_internals() -> None:
    """Doc 06, section 6: no paths, no repo ids, no raw exception text."""
    model_path = "/opt/models/qwen3/pytorch_model.bin"
    provider = make_provider(ExplodingEncoder(RuntimeError(model_path)))

    with pytest.raises(EmbeddingInferenceError) as raised:
        await provider.embed_documents(["a"])

    message = str(raised.value)
    assert model_path not in message
    assert DEFAULT_EMBEDDING_MODEL not in message


async def test_the_original_error_stays_available_as_the_cause() -> None:
    """Safe messages must not cost debuggability."""
    original = RuntimeError("cuda out of memory")
    provider = make_provider(ExplodingEncoder(original))

    with pytest.raises(EmbeddingInferenceError) as raised:
        await provider.embed_documents(["a"])

    assert raised.value.__cause__ is original


async def test_malformed_encoder_output_is_rejected() -> None:
    class NonsenseEncoder:
        def encode(self, sentences: list[str], **kwargs: object) -> object:
            return "not embeddings"

    with pytest.raises(InvalidEmbeddingError, match="expected a sequence"):
        await make_provider(NonsenseEncoder()).embed_documents(["a"])


async def test_malformed_row_is_rejected() -> None:
    class BadRowEncoder:
        def encode(self, sentences: list[str], **kwargs: object) -> object:
            return [None]

    with pytest.raises(InvalidEmbeddingError, match="row"):
        await make_provider(BadRowEncoder()).embed_documents(["a"])


def test_missing_runtime_reports_model_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Absent optional extra must surface as MODEL_UNAVAILABLE, not ImportError."""
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)

    with pytest.raises(ModelUnavailableError, match="not installed"):
        Qwen3EmbeddingProvider.load(settings())


# --------------------------------------------------------------------------
# Composition with the validating wrapper
# --------------------------------------------------------------------------
async def test_wrapped_provider_enforces_the_locked_dimension() -> None:
    provider = ValidatedEmbeddingProvider(make_provider(FakeEncoder(dimension=512)))

    with pytest.raises(InvalidEmbeddingError, match="dimension 512"):
        await provider.embed_documents(["a"])


async def test_wrapped_provider_passes_valid_output_through() -> None:
    provider = ValidatedEmbeddingProvider(make_provider())

    embeddings = await provider.embed_documents(["a", "b"])

    assert len(embeddings) == 2
    assert all(len(embedding) == EMBEDDING_DIMENSION for embedding in embeddings)
