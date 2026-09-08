"""Embedding value type, its invariants, and the error raised when they break.

Every embedding leaving a provider is validated here (doc 03, Phase 05):
non-empty, exactly :data:`EMBEDDING_DIMENSION` entries, every entry a real
finite number -- no NaN, no Infinity.

Privacy
-------
Error messages describe *structure* only: dimensions, positions, counts and
type names. They never contain embedding values, because these messages reach
logs and CLAUDE.md forbids logging embeddings. The same rule keeps them safe to
surface through the public error contract.
"""

import math
from collections.abc import Sequence

__all__ = [
    "EMBEDDING_DIMENSION",
    "Embedding",
    "EmbeddingError",
    "InvalidEmbeddingError",
    "validate_embedding",
    "validate_embeddings",
]

#: Locked by doc 01, decision 8. Not configurable: a different dimension is a
#: different index and a different model, not a runtime setting.
EMBEDDING_DIMENSION = 1024

#: Default tolerance when unit-norm checking is explicitly requested.
_NORM_TOLERANCE = 1e-3

type Embedding = list[float]


class EmbeddingError(Exception):
    """Base class for embedding failures.

    Phase 15 maps these onto the public error contract. Nothing here decides
    an HTTP status or an error code.
    """


class InvalidEmbeddingError(EmbeddingError):
    """A provider returned something that is not a usable embedding."""


def _describe(value: object) -> str:
    """Name a value's type without revealing the value itself."""
    return type(value).__name__


def validate_embedding(
    embedding: object,
    *,
    dimension: int = EMBEDDING_DIMENSION,
    expect_unit_norm: bool = False,
    position: int | None = None,
) -> Embedding:
    """Return ``embedding`` as a validated list of floats.

    Args:
        embedding: The candidate vector, straight from a provider.
        dimension: Required entry count.
        expect_unit_norm: Also require an L2 norm of 1. Off by default -- doc
            01, section 7 recommends cosine similarity with normalized vectors
            but does not lock it, so enforcing it would harden a decision the
            architecture has not made.
        position: Index within a batch, used only to make errors locatable.

    Raises:
        InvalidEmbeddingError: on any violation.
    """
    where = "" if position is None else f" at position {position}"

    if isinstance(embedding, str | bytes | dict) or not isinstance(embedding, Sequence):
        message = f"embedding{where} must be a sequence of numbers, got {_describe(embedding)}"
        raise InvalidEmbeddingError(message)

    if not embedding:
        message = f"embedding{where} is empty"
        raise InvalidEmbeddingError(message)

    if len(embedding) != dimension:
        message = f"embedding{where} has dimension {len(embedding)}, expected {dimension}"
        raise InvalidEmbeddingError(message)

    values: Embedding = []
    for entry_index, entry in enumerate(embedding):
        # bool is a subclass of int, but a boolean is not an embedding value.
        if isinstance(entry, bool) or not isinstance(entry, int | float):
            message = (
                f"embedding{where} has a non-numeric entry at index {entry_index}: "
                f"{_describe(entry)}"
            )
            raise InvalidEmbeddingError(message)

        value = float(entry)
        if not math.isfinite(value):
            kind = "NaN" if math.isnan(value) else "infinity"
            message = f"embedding{where} has a non-finite entry ({kind}) at index {entry_index}"
            raise InvalidEmbeddingError(message)

        values.append(value)

    if expect_unit_norm:
        norm = math.sqrt(sum(value * value for value in values))
        if abs(norm - 1.0) > _NORM_TOLERANCE:
            message = f"embedding{where} is not unit-normalized"
            raise InvalidEmbeddingError(message)

    return values


def validate_embeddings(
    embeddings: object,
    *,
    expected_count: int | None = None,
    dimension: int = EMBEDDING_DIMENSION,
    expect_unit_norm: bool = False,
) -> list[Embedding]:
    """Validate a batch of embeddings.

    Args:
        embeddings: The candidate batch, straight from a provider.
        expected_count: Number of inputs embedded. A provider that silently
            drops or duplicates rows would otherwise misalign embeddings with
            their chunks, which is far worse than an outright failure.
        dimension: Required entry count per embedding.
        expect_unit_norm: See :func:`validate_embedding`.

    Raises:
        InvalidEmbeddingError: on any violation.
    """
    if isinstance(embeddings, str | bytes | dict) or not isinstance(embeddings, Sequence):
        message = f"embeddings must be a sequence, got {_describe(embeddings)}"
        raise InvalidEmbeddingError(message)

    if expected_count is not None and len(embeddings) != expected_count:
        message = f"expected {expected_count} embeddings, got {len(embeddings)}"
        raise InvalidEmbeddingError(message)

    if expected_count is None and not embeddings:
        message = "embeddings batch is empty"
        raise InvalidEmbeddingError(message)

    return [
        validate_embedding(
            embedding,
            dimension=dimension,
            expect_unit_norm=expect_unit_norm,
            position=position,
        )
        for position, embedding in enumerate(embeddings)
    ]
