"""Behaviour of deterministic chunk ID derivation."""

import re

import pytest

from memovo_ai.chunking import ChunkingConfig, HybridChunker
from memovo_ai.core import CHUNK_ID_PREFIX, chunk_id

pytestmark = pytest.mark.unit

BASE = {"memory_id": "memory_456", "chunk_index": 0, "content": "MongoDB Vector Search..."}

_ID_PATTERN = re.compile(rf"^{re.escape(CHUNK_ID_PREFIX)}[0-9a-f]{{64}}$")


# --------------------------------------------------------------------------
# The documented guarantee
# --------------------------------------------------------------------------
def test_identical_inputs_produce_identical_ids() -> None:
    assert chunk_id(**BASE) == chunk_id(**BASE)


def test_repeated_retries_stay_stable() -> None:
    """Reprocessing an unchanged Memory must reproduce the same IDs."""
    assert len({chunk_id(**BASE) for _ in range(100)}) == 1


def test_changed_content_changes_the_id() -> None:
    assert chunk_id(**{**BASE, "content": "MongoDB Vector Search!"}) != chunk_id(**BASE)


def test_changed_index_changes_the_id() -> None:
    assert chunk_id(**{**BASE, "chunk_index": 1}) != chunk_id(**BASE)


def test_changed_memory_id_changes_the_id() -> None:
    assert chunk_id(**{**BASE, "memory_id": "memory_457"}) != chunk_id(**BASE)


def test_a_single_character_change_changes_the_id() -> None:
    assert chunk_id(**{**BASE, "content": BASE["content"] + " "}) != chunk_id(**BASE)


def test_whitespace_only_difference_changes_the_id() -> None:
    """Final content is hashed verbatim; normalization happened in Phase 02."""
    assert chunk_id(memory_id="m", chunk_index=0, content="a b") != chunk_id(
        memory_id="m", chunk_index=0, content="a  b"
    )


# --------------------------------------------------------------------------
# Format
# --------------------------------------------------------------------------
def test_id_has_the_documented_shape() -> None:
    assert _ID_PATTERN.match(chunk_id(**BASE))


def test_id_is_a_full_untruncated_sha256() -> None:
    """Chunk IDs are reconciliation keys; truncating trades collision safety
    for nothing the contract asks for."""
    assert len(chunk_id(**BASE)) == len(CHUNK_ID_PREFIX) + 64


@pytest.mark.parametrize(
    "content",
    ["", " ", "بحث المتجهات", "日本語の検索", "emoji 🧠", "a" * 10_000, "line\nbreak"],
)
def test_shape_holds_for_any_content(content: str) -> None:
    assert _ID_PATTERN.match(chunk_id(memory_id="m", chunk_index=0, content=content))


# --------------------------------------------------------------------------
# Golden vectors: these pin the algorithm itself
# --------------------------------------------------------------------------
def test_golden_vector_ascii() -> None:
    """An accidental change to framing, encoding or digest breaks this."""
    assert (
        chunk_id(memory_id="memory_456", chunk_index=0, content="MongoDB Vector Search...")
        == "chunk_c6eedb1f3e6d22a8a3d69b3b008002a41e82b7fe88d55530ea7fff92324fe1d2"
    )


def test_golden_vector_arabic() -> None:
    """Pins UTF-8 encoding, so IDs do not drift with platform or locale."""
    assert (
        chunk_id(memory_id="memory_456", chunk_index=0, content="بحث المتجهات")
        == "chunk_982a073382969fdfeeb0a1bf65216cf5cd0b79832ebd10b8611b0e9990440349"
    )


# --------------------------------------------------------------------------
# Field framing
# --------------------------------------------------------------------------
def test_separator_ambiguity_does_not_collide() -> None:
    """The case that plain separator concatenation gets wrong.

    ``("memory", 1, "2:2:text")`` and ``("memory:1", 2, "2:text")`` both render
    to ``memory:1:2:2:text``. They are different chunks of different Memories
    and must not share an ID.
    """
    assert chunk_id(memory_id="memory", chunk_index=1, content="2:2:text") != chunk_id(
        memory_id="memory:1", chunk_index=2, content="2:text"
    )


def test_content_cannot_impersonate_a_memory_id() -> None:
    assert chunk_id(memory_id="ab", chunk_index=0, content="c") != chunk_id(
        memory_id="a", chunk_index=0, content="bc"
    )


def test_index_boundary_does_not_leak_into_content() -> None:
    assert chunk_id(memory_id="m", chunk_index=1, content="0x") != chunk_id(
        memory_id="m", chunk_index=10, content="x"
    )


@pytest.mark.parametrize("separator", [":", "|", "\x00", "\n", "-"])
def test_memory_ids_containing_separators_stay_distinct(separator: str) -> None:
    first = chunk_id(memory_id=f"a{separator}b", chunk_index=0, content="x")
    second = chunk_id(memory_id="a", chunk_index=0, content=f"{separator}bx")

    assert first != second


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
@pytest.mark.parametrize("index", [-1, -10])
def test_negative_index_is_rejected(index: int) -> None:
    with pytest.raises(ValueError, match="must not be negative"):
        chunk_id(memory_id="m", chunk_index=index, content="c")


def test_index_zero_is_accepted() -> None:
    assert chunk_id(memory_id="m", chunk_index=0, content="c")


def test_large_index_is_accepted() -> None:
    assert _ID_PATTERN.match(chunk_id(memory_id="m", chunk_index=10_000, content="c"))


# --------------------------------------------------------------------------
# Uniqueness across a chunk set
# --------------------------------------------------------------------------
def test_indices_of_one_memory_produce_distinct_ids() -> None:
    ids = {chunk_id(memory_id="m", chunk_index=i, content="same content") for i in range(200)}

    assert len(ids) == 200


def test_same_content_across_memories_produces_distinct_ids() -> None:
    ids = {chunk_id(memory_id=f"memory_{n}", chunk_index=0, content="shared") for n in range(200)}

    assert len(ids) == 200


def test_ids_are_independent_of_call_order() -> None:
    forward = [chunk_id(memory_id="m", chunk_index=i, content=f"c{i}") for i in range(10)]
    backward = [
        chunk_id(memory_id="m", chunk_index=i, content=f"c{i}") for i in reversed(range(10))
    ]

    assert forward == list(reversed(backward))


# --------------------------------------------------------------------------
# Over real chunks (Phase 03 output)
#
# Phase 04's own guarantee only. The full reprocessing contract -- returning
# the complete chunk set for A / B2 / C -- is Phase 16.
# --------------------------------------------------------------------------
def _ids_for(text: str) -> list[str]:
    chunker = HybridChunker(config=ChunkingConfig(target_tokens=40, overlap_tokens=10))
    return [
        chunk_id(memory_id="memory_456", chunk_index=chunk.index, content=chunk.content)
        for chunk in chunker.chunk(text)
    ]


def _paragraph(label: str) -> str:
    """A paragraph just under the 40-token target, so it becomes its own chunk."""
    return f"{label}. " + " ".join(f"{label} sentence {n} about vectors." for n in range(4))


def _document(third_label: str) -> str:
    return "\n\n".join([_paragraph("Alpha"), _paragraph("Beta"), _paragraph(third_label)])


def test_rechunking_unchanged_content_reproduces_every_id() -> None:
    text = _document("Gamma")

    assert _ids_for(text) == _ids_for(text)


def test_chunk_ids_within_one_memory_are_unique() -> None:
    ids = _ids_for(" ".join(f"Sentence number {n} about vectors." for n in range(60)))

    assert len(ids) > 1
    assert len(set(ids)) == len(ids)


def test_editing_a_later_paragraph_leaves_earlier_ids_untouched() -> None:
    """Only chunks whose final content actually changed get new IDs."""
    original = _ids_for(_document("Gamma"))
    edited = _ids_for(_document("Delta"))

    assert len(original) == len(edited) == 3
    assert original[:2] == edited[:2]
    assert original[2] != edited[2]
