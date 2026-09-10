"""Exact unit-length rescaling.

Qwen3-Embedding already normalizes internally, but it runs in bfloat16, so its
norms land around ``1.0 +/- 0.003`` -- outside the validator's 1e-3 tolerance.
Measured, passing ``normalize_embeddings=True`` to the encoder is a no-op,
because the rounding happens after the model's own Normalize layer. Rescaling
in float64 is what closes the gap.
"""

import math
from collections.abc import Sequence

import pytest

from memovo_ai.embeddings import (
    EMBEDDING_DIMENSION,
    Embedding,
    EmbeddingProvider,
    InvalidEmbeddingError,
    NormalizingEmbeddingProvider,
    ValidatedEmbeddingProvider,
    l2_norm,
)

pytestmark = pytest.mark.unit

#: The worst deviation measured from the real model in the Phase 06 benchmark.
OBSERVED_MODEL_DEVIATION = 0.003646


def vector(*, scale: float = 1.0, dimension: int = EMBEDDING_DIMENSION) -> list[float]:
    """A vector whose norm is `scale`, built from varied components."""
    raw = [math.sin(index) for index in range(dimension)]
    norm = l2_norm(raw)
    return [value * scale / norm for value in raw]


class FixedProvider:
    def __init__(self, documents: list[Embedding], query: Embedding | None = None) -> None:
        self._documents = documents
        self._query = query if query is not None else documents[0]

    async def embed_documents(self, texts: Sequence[str]) -> list[Embedding]:
        return [list(v) for v in self._documents[: len(texts)]]

    async def embed_query(self, query: str) -> Embedding:
        return list(self._query)


def normalized(provider: object) -> NormalizingEmbeddingProvider:
    return NormalizingEmbeddingProvider(provider)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# Exactness
# --------------------------------------------------------------------------
async def test_documents_come_back_at_exactly_unit_length() -> None:
    provider = normalized(FixedProvider([vector(scale=1.0 + OBSERVED_MODEL_DEVIATION)]))

    embedding = (await provider.embed_documents(["chunk"]))[0]

    assert l2_norm(embedding) == pytest.approx(1.0, abs=1e-12)


async def test_queries_come_back_at_exactly_unit_length() -> None:
    provider = normalized(FixedProvider([vector()], query=vector(scale=0.9967)))

    assert l2_norm(await provider.embed_query("q")) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("scale", [0.996874, 1.003646, 0.5, 2.0, 1e-6, 1e6])
async def test_any_magnitude_is_rescaled(scale: float) -> None:
    provider = normalized(FixedProvider([vector(scale=scale)]))

    embedding = (await provider.embed_documents(["chunk"]))[0]

    assert l2_norm(embedding) == pytest.approx(1.0, abs=1e-12)


async def test_an_already_unit_vector_is_left_at_unit_length() -> None:
    provider = normalized(FixedProvider([vector()]))

    assert l2_norm((await provider.embed_documents(["c"]))[0]) == pytest.approx(1.0, abs=1e-12)


# --------------------------------------------------------------------------
# Direction is untouched, so no similarity changes
# --------------------------------------------------------------------------
async def test_rescaling_preserves_direction() -> None:
    """Cosine similarity is scale-invariant; this must change no ranking."""
    original = vector(scale=1.0037)
    provider = normalized(FixedProvider([original]))

    rescaled = (await provider.embed_documents(["chunk"]))[0]

    scale = l2_norm(original)
    assert rescaled == pytest.approx([value / scale for value in original], abs=1e-12)


async def test_cosine_similarity_is_unchanged_by_rescaling() -> None:
    a, b = vector(scale=1.0031), [math.cos(i) for i in range(EMBEDDING_DIMENSION)]

    def cosine(x: Sequence[float], y: Sequence[float]) -> float:
        return sum(p * q for p, q in zip(x, y, strict=True)) / (l2_norm(x) * l2_norm(y))

    before = cosine(a, b)
    after = cosine((await normalized(FixedProvider([a])).embed_documents(["c"]))[0], b)

    assert after == pytest.approx(before, abs=1e-12)


# --------------------------------------------------------------------------
# The point of the exercise: the validator's contract now holds
# --------------------------------------------------------------------------
async def test_model_precision_output_fails_validation_without_rescaling() -> None:
    """This is the state the real model ships in."""
    provider = ValidatedEmbeddingProvider(
        FixedProvider([vector(scale=1.0 + OBSERVED_MODEL_DEVIATION)]),  # type: ignore[arg-type]
        expect_unit_norm=True,
    )

    with pytest.raises(InvalidEmbeddingError, match="unit-normalized"):
        await provider.embed_documents(["chunk"])


async def test_rescaling_first_makes_the_unit_norm_contract_hold() -> None:
    provider = ValidatedEmbeddingProvider(
        normalized(FixedProvider([vector(scale=1.0 + OBSERVED_MODEL_DEVIATION)])),
        expect_unit_norm=True,
    )

    assert len(await provider.embed_documents(["chunk"])) == 1


async def test_the_query_path_satisfies_the_contract_too() -> None:
    provider = ValidatedEmbeddingProvider(
        normalized(FixedProvider([vector()], query=vector(scale=1.0036))),
        expect_unit_norm=True,
    )

    assert len(await provider.embed_query("q")) == EMBEDDING_DIMENSION


# --------------------------------------------------------------------------
# Structure and degenerate input
# --------------------------------------------------------------------------
def test_the_wrapper_is_itself_a_provider() -> None:
    assert isinstance(normalized(FixedProvider([vector()])), EmbeddingProvider)


async def test_order_and_count_are_preserved() -> None:
    documents = [vector(scale=1.01), vector(scale=0.99), vector(scale=1.0)]
    provider = normalized(FixedProvider(documents))

    embeddings = await provider.embed_documents(["a", "b", "c"])

    assert len(embeddings) == 3
    assert all(l2_norm(e) == pytest.approx(1.0, abs=1e-12) for e in embeddings)


async def test_an_empty_batch_stays_empty() -> None:
    assert await normalized(FixedProvider([vector()])).embed_documents([]) == []


async def test_a_zero_vector_is_rejected_rather_than_dividing_by_zero() -> None:
    provider = normalized(FixedProvider([[0.0] * EMBEDDING_DIMENSION]))

    with pytest.raises(InvalidEmbeddingError, match="zero magnitude"):
        await provider.embed_documents(["chunk"])


async def test_a_zero_query_vector_is_rejected() -> None:
    provider = normalized(FixedProvider([vector()], query=[0.0] * EMBEDDING_DIMENSION))

    with pytest.raises(InvalidEmbeddingError, match="zero magnitude"):
        await provider.embed_query("q")


async def test_a_bad_row_is_located_by_position() -> None:
    provider = normalized(FixedProvider([vector(), [0.0] * EMBEDDING_DIMENSION]))

    with pytest.raises(InvalidEmbeddingError, match="at position 1"):
        await provider.embed_documents(["a", "b"])


# --------------------------------------------------------------------------
# The composition root wires it in the right order
# --------------------------------------------------------------------------
def test_the_composition_root_normalizes_then_validates() -> None:
    """Order matters: validating first would reject the model's own output."""
    import ast
    from pathlib import Path

    from memovo_ai.api import dependencies

    source = Path(dependencies.__file__ or "").read_text(encoding="utf-8")
    tree = ast.parse(source)

    builder = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "build_embedding_provider"
    )
    call = next(
        node
        for node in ast.walk(builder)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "ValidatedEmbeddingProvider"
    )

    # Validation wraps normalization, not the other way round.
    inner = call.args[0]
    assert isinstance(inner, ast.Call)
    assert isinstance(inner.func, ast.Name)
    assert inner.func.id == "NormalizingEmbeddingProvider"

    # ...and the unit-norm contract is actually switched on.
    assert any(
        kw.arg == "expect_unit_norm" and getattr(kw.value, "value", None) is True
        for kw in call.keywords
    )
