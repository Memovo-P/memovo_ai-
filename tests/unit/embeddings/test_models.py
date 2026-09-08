"""Embedding validation invariants."""

import math

import pytest

from memovo_ai.embeddings import (
    EMBEDDING_DIMENSION,
    EmbeddingError,
    InvalidEmbeddingError,
    validate_embedding,
    validate_embeddings,
)

pytestmark = pytest.mark.unit


def vector(dimension: int = EMBEDDING_DIMENSION, value: float = 0.1) -> list[float]:
    return [value] * dimension


# --------------------------------------------------------------------------
# Locked dimension
# --------------------------------------------------------------------------
def test_dimension_is_locked_at_1024() -> None:
    """Doc 01, decision 8."""
    assert EMBEDDING_DIMENSION == 1024


def test_a_correct_embedding_is_accepted() -> None:
    assert validate_embedding(vector()) == vector()


def test_returns_floats_not_the_original_container() -> None:
    result = validate_embedding(tuple(vector()))

    assert isinstance(result, list)
    assert all(isinstance(entry, float) for entry in result)


@pytest.mark.parametrize("dimension", [1, 2, 512, 768, 1023, 1025, 2048])
def test_wrong_dimension_is_rejected(dimension: int) -> None:
    with pytest.raises(InvalidEmbeddingError, match="dimension"):
        validate_embedding(vector(dimension))


def test_dimension_is_overridable_for_tests() -> None:
    assert validate_embedding([0.1, 0.2], dimension=2) == [0.1, 0.2]


# --------------------------------------------------------------------------
# Empty and malformed output
# --------------------------------------------------------------------------
def test_empty_embedding_is_rejected() -> None:
    with pytest.raises(InvalidEmbeddingError, match="empty"):
        validate_embedding([])


@pytest.mark.parametrize(
    "value",
    [None, "not an embedding", b"bytes", 0.5, {"a": 1}, {0.1, 0.2}, object()],
)
def test_non_sequence_output_is_rejected(value: object) -> None:
    with pytest.raises(InvalidEmbeddingError, match="sequence"):
        validate_embedding(value)


@pytest.mark.parametrize("entry", ["0.1", None, [0.1], {"x": 1}, complex(1, 2), True, False])
def test_non_numeric_entries_are_rejected(entry: object) -> None:
    candidate = [*vector(EMBEDDING_DIMENSION - 1), entry]

    with pytest.raises(InvalidEmbeddingError, match="non-numeric"):
        validate_embedding(candidate)


def test_integer_entries_are_widened_to_float() -> None:
    result = validate_embedding([0, 1, -1], dimension=3)

    assert result == [0.0, 1.0, -1.0]
    assert all(isinstance(entry, float) for entry in result)


# --------------------------------------------------------------------------
# Finiteness
# --------------------------------------------------------------------------
def test_nan_is_rejected() -> None:
    with pytest.raises(InvalidEmbeddingError, match="NaN"):
        validate_embedding([*vector(EMBEDDING_DIMENSION - 1), math.nan])


@pytest.mark.parametrize("value", [math.inf, -math.inf])
def test_infinity_is_rejected(value: float) -> None:
    with pytest.raises(InvalidEmbeddingError, match="infinity"):
        validate_embedding([*vector(EMBEDDING_DIMENSION - 1), value])


def test_non_finite_entry_is_located_by_index() -> None:
    with pytest.raises(InvalidEmbeddingError, match="index 0"):
        validate_embedding([math.nan, *vector(EMBEDDING_DIMENSION - 1)])


@pytest.mark.parametrize("value", [0.0, -0.0, 1e-300, 1e300, -1e300])
def test_finite_extremes_are_accepted(value: float) -> None:
    assert validate_embedding([value] * 3, dimension=3) == [value] * 3


# --------------------------------------------------------------------------
# Unit norm (opt-in)
# --------------------------------------------------------------------------
def test_unit_norm_is_not_checked_by_default() -> None:
    """Cosine normalization is recommended, not locked (doc 01, section 7)."""
    assert validate_embedding(vector(value=5.0))


def test_unit_norm_is_enforced_when_requested() -> None:
    with pytest.raises(InvalidEmbeddingError, match="unit-normalized"):
        validate_embedding(vector(value=5.0), expect_unit_norm=True)


def test_a_normalized_vector_passes_the_norm_check() -> None:
    value = 1.0 / math.sqrt(EMBEDDING_DIMENSION)

    assert validate_embedding(vector(value=value), expect_unit_norm=True)


# --------------------------------------------------------------------------
# Batches
# --------------------------------------------------------------------------
def test_a_valid_batch_is_accepted() -> None:
    assert validate_embeddings([vector(), vector()], expected_count=2) == [vector(), vector()]


def test_batch_count_mismatch_is_rejected() -> None:
    """Misaligned rows would attach the wrong embedding to a chunk."""
    with pytest.raises(InvalidEmbeddingError, match="expected 3 embeddings, got 2"):
        validate_embeddings([vector(), vector()], expected_count=3)


def test_empty_batch_matching_zero_inputs_is_accepted() -> None:
    assert validate_embeddings([], expected_count=0) == []


def test_empty_batch_without_an_expected_count_is_rejected() -> None:
    with pytest.raises(InvalidEmbeddingError, match="empty"):
        validate_embeddings([])


@pytest.mark.parametrize("value", [None, "text", 0.5, {"a": 1}])
def test_non_sequence_batch_is_rejected(value: object) -> None:
    with pytest.raises(InvalidEmbeddingError, match="sequence"):
        validate_embeddings(value)


def test_a_bad_row_is_located_by_position() -> None:
    batch = [vector(), vector(512), vector()]

    with pytest.raises(InvalidEmbeddingError, match="at position 1"):
        validate_embeddings(batch, expected_count=3)


def test_a_nan_row_is_located_by_position() -> None:
    batch = [vector(), [math.nan, *vector(EMBEDDING_DIMENSION - 1)]]

    with pytest.raises(InvalidEmbeddingError, match="at position 1"):
        validate_embeddings(batch, expected_count=2)


# --------------------------------------------------------------------------
# Privacy of error messages
# --------------------------------------------------------------------------
def test_error_messages_never_contain_embedding_values() -> None:
    """CLAUDE.md forbids logging embeddings; these messages reach logs."""
    marker = 0.123456789

    with pytest.raises(InvalidEmbeddingError) as raised:
        validate_embedding([marker, marker], dimension=1024)

    assert "0.123456789" not in str(raised.value)
    assert "dimension 2" in str(raised.value)


def test_non_numeric_error_names_the_type_not_the_value() -> None:
    with pytest.raises(InvalidEmbeddingError) as raised:
        validate_embedding(["secret-token"], dimension=1)

    assert "secret-token" not in str(raised.value)
    assert "str" in str(raised.value)


# --------------------------------------------------------------------------
# Exception hierarchy
# --------------------------------------------------------------------------
def test_invalid_embedding_error_is_an_embedding_error() -> None:
    """Phase 15 maps EmbeddingError onto the public error contract."""
    assert issubclass(InvalidEmbeddingError, EmbeddingError)
    assert issubclass(EmbeddingError, Exception)
