"""The memory processor service.

Covers the process matrix with a stub embedding provider. No model is loaded.
"""

from collections.abc import Sequence

import pytest

from memovo_ai.chunking import ChunkingConfig, HybridChunker
from memovo_ai.core.errors import AiServiceError
from memovo_ai.core.hashing import CHUNK_ID_PREFIX, chunk_id
from memovo_ai.embeddings import EMBEDDING_DIMENSION, Embedding
from memovo_ai.schemas import (
    ErrorCode,
    LinkProcessRequest,
    NoteProcessRequest,
    ProcessMemoryRequestAdapter,
)
from memovo_ai.services import MemoryProcessorService

pytestmark = pytest.mark.unit

SOURCE: dict[str, object] = {
    "sourceTitle": "Tech Blog",
    "sourceDescription": "Articles about AI and ML",
    "authorName": "Jane Doe",
    "publicationDate": "2026-09-01",
}

NULL_SOURCE: dict[str, object] = dict.fromkeys(SOURCE)


class StubEmbeddings:
    """Returns one distinct vector per text and records the batches."""

    def __init__(self) -> None:
        self.batches: list[list[str]] = []
        self.query_calls: list[str] = []

    async def embed_documents(self, texts: Sequence[str]) -> list[Embedding]:
        self.batches.append(list(texts))
        return [[0.001 * (index + 1)] * EMBEDDING_DIMENSION for index in range(len(texts))]

    async def embed_query(self, query: str) -> Embedding:
        self.query_calls.append(query)
        return [0.1] * EMBEDDING_DIMENSION


class MiscountingEmbeddings(StubEmbeddings):
    async def embed_documents(self, texts: Sequence[str]) -> list[Embedding]:
        return [[0.1] * EMBEDDING_DIMENSION]


class ExplodingChunker(HybridChunker):
    def chunk(self, content: str) -> list[object]:  # type: ignore[override]
        message = "boom"
        raise RuntimeError(message)


def build(
    *,
    embeddings: StubEmbeddings | None = None,
    target: int = 500,
    overlap: int = 50,
    chunker: HybridChunker | None = None,
) -> tuple[MemoryProcessorService, StubEmbeddings]:
    stub = embeddings if embeddings is not None else StubEmbeddings()
    service = MemoryProcessorService(
        embedding_provider=stub,
        chunker=chunker
        or HybridChunker(config=ChunkingConfig(target_tokens=target, overlap_tokens=overlap)),
    )
    return service, stub


def note(
    *,
    memory_id: str = "memory_456",
    title: str = "MongoDB Vector Search",
    content: str = "How Atlas Vector Search works...",
    tags: list[str] | None = None,
) -> NoteProcessRequest:
    request = ProcessMemoryRequestAdapter.validate_python(
        {
            "type": "note",
            "memoryId": memory_id,
            "title": title,
            "content": content,
            "tags": ["mongodb", "vector-search", "ai"] if tags is None else tags,
        }
    )
    assert isinstance(request, NoteProcessRequest)
    return request


def link(
    *,
    memory_id: str = "memory_link_1",
    url: str = "https://example.com/article",
    title: str = "Understanding Vector Embeddings",
    content: str | None = "Good reference for understanding embeddings",
    tags: list[str] | None = None,
    source: dict[str, object] | None = None,
    extracted_content: str | None = "Vector embeddings are numerical representations...",
) -> LinkProcessRequest:
    request = ProcessMemoryRequestAdapter.validate_python(
        {
            "type": "link",
            "memoryId": memory_id,
            "url": url,
            "title": title,
            "content": content,
            "tags": ["ai", "embeddings"] if tags is None else tags,
            "source": SOURCE if source is None else source,
            "extractedContent": extracted_content,
        }
    )
    assert isinstance(request, LinkProcessRequest)
    return request


# --------------------------------------------------------------------------
# The documented pipeline
# --------------------------------------------------------------------------
async def test_a_note_is_chunked_and_embedded() -> None:
    service, _ = build()

    response = await service.process(note())

    assert response.memory_id == "memory_456"
    assert len(response.chunks) == 1
    assert len(response.chunks[0].embedding) == EMBEDDING_DIMENSION


async def test_the_response_echoes_the_request_memory_id() -> None:
    service, _ = build()

    assert (await service.process(note(memory_id="abc"))).memory_id == "abc"
    assert (await service.process(link(memory_id="xyz"))).memory_id == "xyz"


async def test_every_note_embedding_input_field_reaches_the_chunks() -> None:
    """Embedding input = title + content + tags (contract 8.4)."""
    service, embeddings = build()

    await service.process(note())

    combined = " ".join(embeddings.batches[0])
    assert combined.startswith("Title:\nMongoDB Vector Search")
    assert "Content:\nHow Atlas Vector Search works..." in combined
    assert "Tags:\nmongodb, vector-search, ai" in combined


async def test_every_link_embedding_input_field_reaches_the_chunks() -> None:
    """A Link adds source metadata and extracted content (contract 8.4)."""
    service, embeddings = build()

    await service.process(link())

    combined = " ".join(embeddings.batches[0])
    assert "Title:\nUnderstanding Vector Embeddings" in combined
    assert "Content:\nGood reference for understanding embeddings" in combined
    assert "Source:\nTech Blog, Articles about AI and ML, Jane Doe, 2026-09-01" in combined
    assert "Extracted Content:\nVector embeddings are numerical representations..." in combined
    assert "Tags:\nai, embeddings" in combined


async def test_the_url_is_context_not_embedding_input() -> None:
    """Deliberate: section 8.4 lists what is embedded, and ``url`` is not on
    that list. Changing this changes every Link chunk ID."""
    service, embeddings = build()

    await service.process(link(url="https://unique-host.example/path"))

    assert "unique-host.example" not in " ".join(embeddings.batches[0])


async def test_null_source_fields_never_render_as_text() -> None:
    service, embeddings = build()

    await service.process(link(source=NULL_SOURCE))

    combined = " ".join(embeddings.batches[0])
    assert "Source:" not in combined
    assert "None" not in combined
    assert "null" not in combined


async def test_chunks_are_embedded_in_one_batch() -> None:
    """Batching is the provider's job; the service issues a single call."""
    service, embeddings = build(target=40, overlap=0)

    response = await service.process(note(content=" ".join(f"w{n:03d}" for n in range(300))))

    assert len(response.chunks) > 1
    assert len(embeddings.batches) == 1
    assert len(embeddings.batches[0]) == len(response.chunks)


async def test_embeddings_are_paired_with_their_own_chunk() -> None:
    service, _ = build(target=40, overlap=0)

    response = await service.process(note(content=" ".join(f"w{n:03d}" for n in range(200))))

    assert len({tuple(chunk.embedding) for chunk in response.chunks}) == len(response.chunks)


async def test_the_embedded_text_is_the_final_chunk_content() -> None:
    """Hashing and embedding must see the same string, overlap included."""
    service, embeddings = build(target=40, overlap=10)

    response = await service.process(note(content=" ".join(f"w{n:03d}" for n in range(200))))

    assert embeddings.batches[0] == [chunk.content for chunk in response.chunks]


async def test_no_query_is_ever_embedded() -> None:
    service, embeddings = build()

    await service.process(note())
    await service.process(link())

    assert embeddings.query_calls == []


# --------------------------------------------------------------------------
# Chunk indices and IDs
# --------------------------------------------------------------------------
async def test_chunk_indices_are_contiguous_from_zero() -> None:
    service, _ = build(target=40, overlap=0)

    response = await service.process(note(content=" ".join(f"w{n:03d}" for n in range(300))))

    assert [chunk.chunk_index for chunk in response.chunks] == list(range(len(response.chunks)))


async def test_chunk_ids_are_derived_from_the_documented_inputs() -> None:
    service, _ = build()

    response = await service.process(note())
    chunk = response.chunks[0]

    assert chunk.chunk_id == chunk_id(
        memory_id="memory_456", chunk_index=chunk.chunk_index, content=chunk.content
    )
    assert chunk.chunk_id.startswith(CHUNK_ID_PREFIX)


async def test_chunk_ids_are_unique_within_a_memory() -> None:
    service, _ = build(target=40, overlap=0)

    response = await service.process(note(content=" ".join(f"w{n:03d}" for n in range(300))))

    assert len({chunk.chunk_id for chunk in response.chunks}) == len(response.chunks)


async def test_a_different_memory_id_changes_every_chunk_id() -> None:
    service, _ = build()

    first = await service.process(note(memory_id="memory_1"))
    second = await service.process(note(memory_id="memory_2"))

    assert first.chunks[0].chunk_id != second.chunks[0].chunk_id
    assert first.chunks[0].content == second.chunks[0].content


# --------------------------------------------------------------------------
# Reprocessing (contract section 13)
# --------------------------------------------------------------------------
async def test_reprocessing_unchanged_input_reproduces_identical_chunk_ids() -> None:
    service, _ = build()

    first = await service.process(note())
    second = await service.process(note())

    assert [c.chunk_id for c in first.chunks] == [c.chunk_id for c in second.chunks]
    assert [c.content for c in first.chunks] == [c.content for c in second.chunks]


async def test_reprocessing_returns_the_complete_chunk_set_not_a_diff() -> None:
    service, _ = build(target=40, overlap=0)
    long_text = " ".join(f"w{n:03d}" for n in range(300))

    original = await service.process(note(content=long_text))
    edited = await service.process(note(content=long_text + " extra"))

    assert len(edited.chunks) >= len(original.chunks) - 1
    assert len(edited.chunks) > 1


async def test_a_changed_tag_changes_the_affected_chunk_id() -> None:
    service, _ = build()

    first = await service.process(note(tags=["one"]))
    second = await service.process(note(tags=["two"]))

    assert first.chunks[0].chunk_id != second.chunks[0].chunk_id


async def test_a_changed_source_field_changes_the_affected_chunk_id() -> None:
    service, _ = build()

    first = await service.process(link(source={**SOURCE, "authorName": "Jane Doe"}))
    second = await service.process(link(source={**SOURCE, "authorName": "John Roe"}))

    assert first.chunks[0].chunk_id != second.chunks[0].chunk_id


async def test_a_changed_url_alone_keeps_every_chunk_id() -> None:
    """The url is not embedded, so it cannot move an ID."""
    service, _ = build()

    first = await service.process(link(url="https://example.com/a"))
    second = await service.process(link(url="https://example.com/b"))

    assert [c.chunk_id for c in first.chunks] == [c.chunk_id for c in second.chunks]


async def test_the_service_holds_no_state_between_calls() -> None:
    """Stateless with respect to previous versions of a Memory."""
    service, _ = build()

    first = await service.process(note())
    await service.process(note(title="Something else"))
    third = await service.process(note())

    assert first.model_dump_json() == third.model_dump_json()


# --------------------------------------------------------------------------
# Content variations
# --------------------------------------------------------------------------
async def test_empty_tags_are_accepted() -> None:
    service, embeddings = build()

    response = await service.process(note(tags=[]))

    assert "Tags:" not in embeddings.batches[0][0]
    assert len(response.chunks) == 1


async def test_null_tags_are_accepted() -> None:
    service, embeddings = build()

    request = ProcessMemoryRequestAdapter.validate_python(
        {"type": "note", "memoryId": "m", "title": "T", "content": "c", "tags": None}
    )
    response = await service.process(request)

    assert "Tags:" not in embeddings.batches[0][0]
    assert len(response.chunks) == 1


async def test_an_empty_note_title_is_accepted() -> None:
    service, embeddings = build()

    response = await service.process(note(title=""))

    assert embeddings.batches[0][0].startswith("Content:")
    assert len(response.chunks) == 1


async def test_a_link_with_null_content_and_no_extraction_is_accepted() -> None:
    service, embeddings = build()

    response = await service.process(link(content=None, extracted_content=None, tags=[]))

    assert embeddings.batches[0][0] == (
        "Title:\nUnderstanding Vector Embeddings\n\n"
        "Source:\nTech Blog, Articles about AI and ML, Jane Doe, 2026-09-01"
    )
    assert len(response.chunks) == 1


async def test_arabic_content_is_processed() -> None:
    service, embeddings = build()

    response = await service.process(note(title="بحث المتجهات", content="كيف يعمل البحث الدلالي"))

    assert "بحث المتجهات" in embeddings.batches[0][0]
    assert len(response.chunks) >= 1


async def test_mixed_language_content_is_processed() -> None:
    service, embeddings = build()

    await service.process(note(title="MongoDB بحث المتجهات"))

    assert "MongoDB بحث المتجهات" in embeddings.batches[0][0]


async def test_a_long_memory_produces_many_chunks() -> None:
    service, _ = build()

    # 1,000 nine-character words: ~9,000 characters, inside the 10,000 limit
    # and well past one 500-token chunk.
    response = await service.process(note(content=" ".join(f"word{n:04d}" for n in range(1000))))

    assert len(response.chunks) > 1


async def test_a_long_extracted_page_produces_many_chunks() -> None:
    """No AI-side cap on extractedContent; it simply chunks."""
    service, _ = build()

    response = await service.process(
        link(extracted_content=" ".join(f"word{n:04d}" for n in range(4000)))
    )

    assert len(response.chunks) > 1


async def test_a_memory_that_normalizes_to_nothing_yields_no_chunks() -> None:
    """The contract permits an empty chunk set; rejecting would invent a rule."""
    service, embeddings = build()

    response = await service.process(note(title="", content="   \n\t ", tags=[]))

    assert response.chunks == []
    assert response.memory_id == "memory_456"
    assert embeddings.batches == []


# --------------------------------------------------------------------------
# Failures
# --------------------------------------------------------------------------
async def test_a_chunker_failure_maps_to_chunking_failed() -> None:
    service, _ = build(chunker=ExplodingChunker())

    with pytest.raises(AiServiceError) as raised:
        await service.process(note())

    assert raised.value.code is ErrorCode.CHUNKING_FAILED


async def test_a_miscounted_embedding_batch_is_rejected() -> None:
    """A provider that drops rows would misalign embeddings with chunks."""
    service, _ = build(embeddings=MiscountingEmbeddings(), target=40, overlap=0)

    with pytest.raises(AiServiceError) as raised:
        await service.process(note(content=" ".join(f"w{n:03d}" for n in range(300))))

    assert raised.value.code is ErrorCode.EMBEDDING_FAILED


async def test_an_embedding_failure_is_not_swallowed() -> None:
    class FailingEmbeddings(StubEmbeddings):
        async def embed_documents(self, texts: Sequence[str]) -> list[Embedding]:
            message = "inference failed"
            raise RuntimeError(message)

    service, _ = build(embeddings=FailingEmbeddings())

    with pytest.raises(RuntimeError, match="inference failed"):
        await service.process(note())


# --------------------------------------------------------------------------
# Scope
# --------------------------------------------------------------------------
def test_the_service_imports_no_persistence_auth_or_network_library() -> None:
    """No Mongo or vector writes, no auth, no fetching."""
    import ast
    from pathlib import Path

    from memovo_ai.services import memory_processor

    tree = ast.parse(Path(memory_processor.__file__ or "").read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    forbidden = (
        "pymongo",
        "motor",
        "jwt",
        "jose",
        "qdrant",
        "pinecone",
        "boto3",
        "httpx",
        "urllib",
    )
    assert not [name for name in imported if any(bad in name for bad in forbidden)]
    assert not any(name.startswith("memovo_ai.providers") for name in imported)


def test_the_request_carries_no_user_identity() -> None:
    """Processing needs no user context; the Backend owns identity."""
    assert "user_id" not in NoteProcessRequest.model_fields
    assert "user_id" not in LinkProcessRequest.model_fields
