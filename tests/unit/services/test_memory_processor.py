"""The memory processor service.

Covers the doc 05 section 6 process matrix with a stub embedding provider.
No model is loaded.
"""

from collections.abc import Sequence

import pytest

from memovo_ai.chunking import ChunkingConfig, HybridChunker
from memovo_ai.core.errors import AiServiceError
from memovo_ai.core.hashing import CHUNK_ID_PREFIX, chunk_id
from memovo_ai.embeddings import EMBEDDING_DIMENSION, Embedding
from memovo_ai.schemas import ErrorCode, ProcessMemoryRequest
from memovo_ai.services import MemoryProcessorService
from memovo_ai.understanding import compose_canonical_content

pytestmark = pytest.mark.unit


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


def request(
    *,
    memory_id: str = "memory_456",
    title: str = "MongoDB Vector Search",
    description: str = "How Atlas Vector Search works...",
    why_saved: str = "Useful for the memory project",
    tags: list[str] | None = None,
) -> ProcessMemoryRequest:
    return ProcessMemoryRequest.model_validate(
        {
            "memoryId": memory_id,
            "title": title,
            "description": description,
            "whySaved": why_saved,
            "tags": ["mongodb", "vector-search", "ai"] if tags is None else tags,
        }
    )


# --------------------------------------------------------------------------
# The documented pipeline
# --------------------------------------------------------------------------
async def test_a_memory_is_chunked_and_embedded() -> None:
    service, _ = build()

    response = await service.process(request())

    assert response.intent == "save_memory"
    assert len(response.chunks) == 1
    assert len(response.chunks[0].embedding) == EMBEDDING_DIMENSION


async def test_response_content_is_the_canonical_labelled_text() -> None:
    """The prepared content, not the raw description."""
    service, _ = build()

    response = await service.process(request())

    assert response.content == compose_canonical_content(
        title="MongoDB Vector Search",
        description="How Atlas Vector Search works...",
        why_saved="Useful for the memory project",
        tags=["mongodb", "vector-search", "ai"],
    )
    assert response.content.startswith("Title:\nMongoDB Vector Search")
    assert "Why Saved:" in response.content
    assert "Tags:\nmongodb, vector-search, ai" in response.content


async def test_every_embedding_input_field_reaches_the_chunks() -> None:
    """Embedding input = Title + Content + WhySaved + Tags (doc 01, 1-2)."""
    service, embeddings = build()

    await service.process(request())

    combined = " ".join(embeddings.batches[0])
    assert "MongoDB Vector Search" in combined
    assert "How Atlas Vector Search works..." in combined
    assert "Useful for the memory project" in combined
    assert "mongodb, vector-search, ai" in combined


async def test_chunks_are_embedded_in_one_batch() -> None:
    """Batching is the provider's job; the service issues a single call."""
    service, embeddings = build(target=40, overlap=0)

    response = await service.process(request(description=" ".join(f"w{n:03d}" for n in range(300))))

    assert len(response.chunks) > 1
    assert len(embeddings.batches) == 1
    assert len(embeddings.batches[0]) == len(response.chunks)


async def test_embeddings_are_paired_with_their_own_chunk() -> None:
    service, _ = build(target=40, overlap=0)

    response = await service.process(request(description=" ".join(f"w{n:03d}" for n in range(200))))

    assert len({tuple(chunk.embedding) for chunk in response.chunks}) == len(response.chunks)


async def test_the_embedded_text_is_the_final_chunk_content() -> None:
    """Hashing and embedding must see the same string, overlap included."""
    service, embeddings = build(target=40, overlap=10)

    response = await service.process(request(description=" ".join(f"w{n:03d}" for n in range(200))))

    assert embeddings.batches[0] == [chunk.content for chunk in response.chunks]


async def test_no_query_is_ever_embedded() -> None:
    service, embeddings = build()

    await service.process(request())

    assert embeddings.query_calls == []


# --------------------------------------------------------------------------
# Chunk indices and IDs
# --------------------------------------------------------------------------
async def test_chunk_indices_are_contiguous_from_zero() -> None:
    service, _ = build(target=40, overlap=0)

    response = await service.process(request(description=" ".join(f"w{n:03d}" for n in range(300))))

    assert [chunk.chunk_index for chunk in response.chunks] == list(range(len(response.chunks)))


async def test_chunk_ids_are_derived_from_the_documented_inputs() -> None:
    service, _ = build()

    response = await service.process(request())
    chunk = response.chunks[0]

    assert chunk.chunk_id == chunk_id(
        memory_id="memory_456", chunk_index=chunk.chunk_index, content=chunk.content
    )
    assert chunk.chunk_id.startswith(CHUNK_ID_PREFIX)


async def test_chunk_ids_are_unique_within_a_memory() -> None:
    service, _ = build(target=40, overlap=0)

    response = await service.process(request(description=" ".join(f"w{n:03d}" for n in range(300))))

    assert len({chunk.chunk_id for chunk in response.chunks}) == len(response.chunks)


async def test_a_different_memory_id_changes_every_chunk_id() -> None:
    service, _ = build()

    first = await service.process(request(memory_id="memory_1"))
    second = await service.process(request(memory_id="memory_2"))

    assert first.chunks[0].chunk_id != second.chunks[0].chunk_id
    assert first.chunks[0].content == second.chunks[0].content


# --------------------------------------------------------------------------
# Reprocessing (doc 01, decision 25)
# --------------------------------------------------------------------------
async def test_reprocessing_unchanged_input_reproduces_identical_chunk_ids() -> None:
    service, _ = build()

    first = await service.process(request())
    second = await service.process(request())

    assert [c.chunk_id for c in first.chunks] == [c.chunk_id for c in second.chunks]
    assert [c.content for c in first.chunks] == [c.content for c in second.chunks]


async def test_reprocessing_returns_the_complete_chunk_set_not_a_diff() -> None:
    service, _ = build(target=40, overlap=0)
    long_text = " ".join(f"w{n:03d}" for n in range(300))

    original = await service.process(request(description=long_text))
    edited = await service.process(request(description=long_text + " extra"))

    assert len(edited.chunks) >= len(original.chunks) - 1
    assert len(edited.chunks) > 1


async def test_a_changed_field_changes_the_affected_chunk_id() -> None:
    service, _ = build()

    first = await service.process(request(why_saved="Original reason"))
    second = await service.process(request(why_saved="Different reason"))

    assert first.chunks[0].chunk_id != second.chunks[0].chunk_id


async def test_the_service_holds_no_state_between_calls() -> None:
    """Stateless with respect to previous versions of a Memory."""
    service, _ = build()

    first = await service.process(request())
    await service.process(request(title="Something else"))
    third = await service.process(request())

    assert first.model_dump_json() == third.model_dump_json()


# --------------------------------------------------------------------------
# Content variations (doc 05, section 2)
# --------------------------------------------------------------------------
async def test_empty_tags_are_accepted() -> None:
    service, _ = build()

    response = await service.process(request(tags=[]))

    assert "Tags:" not in response.content
    assert len(response.chunks) == 1


async def test_empty_optional_values_are_accepted() -> None:
    service, _ = build()

    response = await service.process(request(why_saved="", description=""))

    assert "Content:" not in response.content
    assert "Why Saved:" not in response.content
    assert response.content.startswith("Title:\nMongoDB Vector Search")
    assert response.content.endswith("Tags:\nmongodb, vector-search, ai")
    assert len(response.chunks) == 1


async def test_arabic_content_is_processed() -> None:
    service, _ = build()

    response = await service.process(
        request(title="بحث المتجهات", description="كيف يعمل البحث الدلالي", why_saved="مفيد")
    )

    assert "بحث المتجهات" in response.content
    assert len(response.chunks) >= 1


async def test_mixed_language_content_is_processed() -> None:
    service, _ = build()

    response = await service.process(request(title="MongoDB بحث المتجهات"))

    assert "MongoDB بحث المتجهات" in response.content


async def test_a_long_memory_produces_many_chunks() -> None:
    service, _ = build()

    response = await service.process(
        request(description=" ".join(f"word{n:04d}" for n in range(4000)))
    )

    assert len(response.chunks) > 1


async def test_an_entirely_empty_memory_yields_no_chunks() -> None:
    """The contract permits an empty chunk set; rejecting would invent a rule."""
    service, _ = build()

    response = await service.process(request(title="", description="", why_saved="", tags=[]))

    assert response.content == ""
    assert response.chunks == []
    assert response.intent == "save_memory"


# --------------------------------------------------------------------------
# Failures
# --------------------------------------------------------------------------
async def test_a_chunker_failure_maps_to_chunking_failed() -> None:
    service, _ = build(chunker=ExplodingChunker())

    with pytest.raises(AiServiceError) as raised:
        await service.process(request())

    assert raised.value.code is ErrorCode.CHUNKING_FAILED
    assert raised.value.retryable is False


async def test_a_miscounted_embedding_batch_is_rejected() -> None:
    """A provider that drops rows would misalign embeddings with chunks."""
    service, _ = build(embeddings=MiscountingEmbeddings(), target=40, overlap=0)

    with pytest.raises(AiServiceError) as raised:
        await service.process(request(description=" ".join(f"w{n:03d}" for n in range(300))))

    assert raised.value.code is ErrorCode.EMBEDDING_FAILED


async def test_an_embedding_failure_is_not_swallowed() -> None:
    class FailingEmbeddings(StubEmbeddings):
        async def embed_documents(self, texts: Sequence[str]) -> list[Embedding]:
            message = "inference failed"
            raise RuntimeError(message)

    service, _ = build(embeddings=FailingEmbeddings())

    with pytest.raises(RuntimeError, match="inference failed"):
        await service.process(request())


# --------------------------------------------------------------------------
# Scope
# --------------------------------------------------------------------------
def test_the_service_imports_no_persistence_or_auth_library() -> None:
    """Doc 03 Phase 07: no Mongo or vector writes, no auth."""
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

    forbidden = ("pymongo", "motor", "jwt", "jose", "qdrant", "pinecone", "boto3")
    assert not [name for name in imported if any(bad in name for bad in forbidden)]
    assert not any(name.startswith("memovo_ai.providers") for name in imported)


async def test_the_request_carries_no_user_identity() -> None:
    """Processing needs no user context; the Backend owns identity."""
    assert "user_id" not in ProcessMemoryRequest.model_fields
