"""The reprocessing contract (doc 03 Phase 16, doc 05 section 9).

The scenario both documents specify::

    original    A  B   C      retry -> identical IDs
    edited      A  B2  C      response is the COMPLETE set, not just B2

Two properties matter to the Backend, and they are different things:

1. **Determinism.** An unchanged Memory reprocesses to identical chunk IDs, so
   a retry is a no-op rather than a duplicate insert.
2. **Completeness.** Every reprocess returns the whole current chunk set, so
   the Backend can reconcile by comparing ID sets instead of diffing text.

This module adds no production code. It pins behaviour the earlier phases
already implement.
"""

from collections.abc import Sequence

import pytest

from memovo_ai.chunking import ChunkingConfig, HybridChunker
from memovo_ai.embeddings import EMBEDDING_DIMENSION, Embedding
from memovo_ai.schemas import ProcessMemoryRequest, ProcessMemoryResponse
from memovo_ai.services import MemoryProcessorService

pytestmark = pytest.mark.contract

MEMORY_ID = "memory_abc"


class StubEmbeddings:
    async def embed_documents(self, texts: Sequence[str]) -> list[Embedding]:
        return [[0.01] * EMBEDDING_DIMENSION for _ in texts]

    async def embed_query(self, query: str) -> Embedding:
        return [0.01] * EMBEDDING_DIMENSION


def paragraph(stem: str, count: int = 24) -> str:
    """A paragraph of ~24 one-token words.

    Sized so two cannot share a 40-token chunk, which makes each paragraph a
    chunk and lets these tests talk about "chunk A" meaningfully.
    """
    return " ".join(f"{stem}{index:03d}" for index in range(count))


A = paragraph("a")
B = paragraph("b")
B2 = paragraph("d")
C = paragraph("c")


def document(*paragraphs: str) -> str:
    return "\n\n".join(paragraphs)


async def process(
    description: str,
    *,
    memory_id: str = MEMORY_ID,
    target: int = 40,
    overlap: int = 0,
    title: str = "T",
    why_saved: str = "",
    tags: list[str] | None = None,
    about: str | None = None,
    source: dict[str, object] | None = None,
) -> ProcessMemoryResponse:
    service = MemoryProcessorService(
        embedding_provider=StubEmbeddings(),
        chunker=HybridChunker(config=ChunkingConfig(target_tokens=target, overlap_tokens=overlap)),
    )
    payload: dict[str, object] = {
        "memoryId": memory_id,
        "title": title,
        "description": description,
        "whySaved": why_saved,
        "tags": tags if tags is not None else [],
    }
    if about is not None:
        payload["about"] = about
    if source is not None:
        payload["source"] = source

    return await service.process(ProcessMemoryRequest.model_validate(payload))


def ids(response: ProcessMemoryResponse) -> list[str]:
    return [chunk.chunk_id for chunk in response.chunks]


# --------------------------------------------------------------------------
# The documented scenario, without overlap
# --------------------------------------------------------------------------
async def test_retrying_an_unchanged_memory_reproduces_every_id() -> None:
    """A B C -> A B C, all IDs identical."""
    first = await process(document(A, B, C))
    second = await process(document(A, B, C))

    assert ids(first) == ids(second)
    assert len(first.chunks) == 3


async def test_editing_the_middle_chunk_changes_only_that_id() -> None:
    """The doc 03 Phase 16 expectation, verified without overlap.

    A keeps its ID, B2 gets a new one, C keeps its ID.
    """
    original = await process(document(A, B, C))
    edited = await process(document(A, B2, C))

    assert original.chunks[0].chunk_id == edited.chunks[0].chunk_id
    assert original.chunks[1].chunk_id != edited.chunks[1].chunk_id
    assert original.chunks[2].chunk_id == edited.chunks[2].chunk_id


async def test_the_edited_chunk_carries_the_new_content() -> None:
    edited = await process(document(A, B2, C))

    assert "d000" in edited.chunks[1].content
    assert "b000" not in edited.chunks[1].content


# --------------------------------------------------------------------------
# Completeness: never a diff
# --------------------------------------------------------------------------
async def test_the_complete_set_is_returned_not_only_the_changed_chunk() -> None:
    """The response is A B2 C, never just B2 (doc 01, decision 25)."""
    edited = await process(document(A, B2, C))

    assert len(edited.chunks) == 3
    assert [chunk.chunk_index for chunk in edited.chunks] == [0, 1, 2]


async def test_unchanged_chunks_still_carry_their_embeddings() -> None:
    """A partial response would leave the Backend unable to re-index."""
    edited = await process(document(A, B2, C))

    for chunk in edited.chunks:
        assert len(chunk.embedding) == EMBEDDING_DIMENSION


async def test_the_response_has_no_delta_vocabulary() -> None:
    edited = await process(document(A, B2, C))
    payload = edited.model_dump()

    assert set(payload) == {"intent", "content", "chunks"}
    for absent in ("changedChunks", "removedChunkIds", "delta", "partial", "diff"):
        assert absent not in payload


# --------------------------------------------------------------------------
# Overlap couples adjacent chunks -- the Backend must expect this
# --------------------------------------------------------------------------
async def test_with_overlap_the_following_chunk_also_gets_a_new_id() -> None:
    """Contract-compliant, but easy to be surprised by.

    Doc 03 says C keeps its ID *only if content and index are unchanged*. With
    overlap enabled, chunk 2 begins with the tail of chunk 1, so editing B
    genuinely changes C's content -- and therefore its ID. One middle edit
    yields two changed IDs at the production overlap of 50.
    """
    original = await process(document(A, B, C), overlap=10)
    edited = await process(document(A, B2, C), overlap=10)

    assert original.chunks[0].chunk_id == edited.chunks[0].chunk_id
    assert original.chunks[1].chunk_id != edited.chunks[1].chunk_id
    assert original.chunks[2].chunk_id != edited.chunks[2].chunk_id


async def test_overlap_does_not_break_retry_determinism() -> None:
    """Coupling changes how many IDs move on an edit, never on a retry."""
    first = await process(document(A, B, C), overlap=10)
    second = await process(document(A, B, C), overlap=10)

    assert ids(first) == ids(second)


# --------------------------------------------------------------------------
# The Backend's reconciliation algorithm (contract section 2.1)
# --------------------------------------------------------------------------
def reconcile(old: Sequence[str], new: Sequence[str]) -> tuple[set[str], set[str], set[str]]:
    """The contract's pseudocode, verbatim.

    Comparing ID sets is only sufficient because chunk IDs are deterministic;
    that is the whole point of section 2.2.
    """
    old_ids, new_ids = set(old), set(new)

    return (new_ids - old_ids, old_ids - new_ids, new_ids & old_ids)


async def test_a_retry_reconciles_to_no_writes_at_all() -> None:
    """Idempotency: the Backend upserts nothing on a duplicate job."""
    stored = ids(await process(document(A, B, C)))
    replayed = ids(await process(document(A, B, C)))

    inserts, deletes, unchanged = reconcile(stored, replayed)

    assert inserts == set()
    assert deletes == set()
    assert len(unchanged) == 3


async def test_a_middle_edit_reconciles_to_one_insert_and_one_delete() -> None:
    stored = ids(await process(document(A, B, C)))
    updated = ids(await process(document(A, B2, C)))

    inserts, deletes, unchanged = reconcile(stored, updated)

    assert len(inserts) == 1
    assert len(deletes) == 1
    assert len(unchanged) == 2


async def test_a_middle_edit_with_overlap_reconciles_to_two_of_each() -> None:
    """The churn the production overlap actually produces."""
    stored = ids(await process(document(A, B, C), overlap=10))
    updated = ids(await process(document(A, B2, C), overlap=10))

    inserts, deletes, unchanged = reconcile(stored, updated)

    assert len(inserts) == 2
    assert len(deletes) == 2
    assert len(unchanged) == 1


async def test_reconciliation_never_needs_text_comparison() -> None:
    """ID sets alone decide every operation (contract section 2.1)."""
    stored = ids(await process(document(A, B, C)))
    updated = ids(await process(document(A, B2, C)))

    inserts, deletes, _ = reconcile(stored, updated)

    assert inserts.isdisjoint(deletes)
    assert all(chunk_id.startswith("chunk_") for chunk_id in inserts | deletes)


# --------------------------------------------------------------------------
# Other edit shapes
# --------------------------------------------------------------------------
async def test_appending_leaves_the_earlier_chunks_untouched() -> None:
    original = await process(document(A, B, C))
    extended = await process(document(A, B, C, paragraph("e")))

    assert len(extended.chunks) == 4
    assert ids(original) == ids(extended)[:3]


async def test_deleting_a_paragraph_drops_exactly_one_chunk() -> None:
    original = await process(document(A, B, C))
    shortened = await process(document(A, C))

    inserts, deletes, unchanged = reconcile(ids(original), ids(shortened))

    assert len(shortened.chunks) == 2
    # A stays; the chunk that held B disappears; C moves to index 1 and so
    # gets a new ID -- chunkIndex is part of the hash.
    assert len(unchanged) == 1
    assert len(deletes) == 2
    assert len(inserts) == 1


async def test_prepending_shifts_indices_and_therefore_ids() -> None:
    """chunkIndex is hashed, so an insertion at the front renumbers the rest."""
    original = await process(document(A, B, C))
    prepended = await process(document(paragraph("z"), A, B, C))

    assert set(ids(original)).isdisjoint(set(ids(prepended)))


async def test_reordering_paragraphs_changes_the_affected_ids() -> None:
    original = await process(document(A, B, C))
    reordered = await process(document(A, C, B))

    assert ids(original)[0] == ids(reordered)[0]
    assert ids(original)[1:] != ids(reordered)[1:]


async def test_editing_a_non_description_field_also_reprocesses_cleanly() -> None:
    """whySaved is embedding input, so changing it changes content."""
    original = await process(document(A, B, C), why_saved="")
    edited = await process(document(A, B, C), why_saved="Because it matters")

    assert len(edited.chunks) >= len(original.chunks)
    assert ids(original) != ids(edited)


# --------------------------------------------------------------------------
# Stability guarantees
# --------------------------------------------------------------------------
async def test_ten_consecutive_retries_are_all_identical() -> None:
    runs = [ids(await process(document(A, B, C), overlap=10)) for _ in range(10)]

    assert len({tuple(run) for run in runs}) == 1


async def test_a_different_memory_id_shares_no_chunk_id() -> None:
    """Two Memories with identical content must never collide."""
    first = await process(document(A, B, C), memory_id="memory_1")
    second = await process(document(A, B, C), memory_id="memory_2")

    assert [c.content for c in first.chunks] == [c.content for c in second.chunks]
    assert set(ids(first)).isdisjoint(set(ids(second)))


async def test_determinism_holds_at_the_locked_production_settings() -> None:
    """Target 500 / overlap 50, the values the Backend will actually see.

    Eight 150-token paragraphs is ~1200 tokens, enough to span several chunks
    at the real target rather than collapsing into one.
    """
    body = document(*(paragraph(stem, 150) for stem in "abcdefgh"))

    first = await process(body, target=500, overlap=50)
    second = await process(body, target=500, overlap=50)

    assert len(first.chunks) > 1
    assert ids(first) == ids(second)


async def test_a_no_op_reprocess_returns_a_byte_identical_response() -> None:
    first = await process(document(A, B, C), overlap=10)
    second = await process(document(A, B, C), overlap=10)

    assert first.model_dump_json() == second.model_dump_json()


# --------------------------------------------------------------------------
# Links reprocess under the same contract (Phase 17)
# --------------------------------------------------------------------------
LINK_SOURCE: dict[str, object] = {"siteName": "MongoDB Docs", "publishedAt": "2024-01-15"}


async def test_reprocessing_a_link_reproduces_every_id() -> None:
    first = await process(document(A, B, C), about="My note", source=LINK_SOURCE)
    second = await process(document(A, B, C), about="My note", source=LINK_SOURCE)

    assert ids(first) == ids(second)


async def test_editing_link_about_returns_the_complete_set() -> None:
    original = await process(document(A, B, C), about="First note", source=LINK_SOURCE)
    edited = await process(document(A, B, C), about="Second note", source=LINK_SOURCE)

    assert len(edited.chunks) == len(original.chunks)
    assert [chunk.chunk_index for chunk in edited.chunks] == list(range(len(edited.chunks)))
    assert ids(original) != ids(edited)


async def test_changing_a_non_embedded_source_field_reconciles_to_nothing() -> None:
    """favicon never reaches the content, so it cannot move an ID."""
    stored = ids(await process(document(A, B, C), source=LINK_SOURCE))
    replayed = ids(
        await process(document(A, B, C), source={**LINK_SOURCE, "favicon": "https://cdn/x.ico"})
    )

    inserts, deletes, _ = reconcile(stored, replayed)

    assert inserts == set()
    assert deletes == set()
