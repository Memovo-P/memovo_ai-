"""Memory ingestion orchestration.

The pipeline::

    ProcessMemoryRequest (Note or Link)
      -> canonical content composition
      -> hybrid chunker
      -> deterministic chunk IDs
      -> batch embeddings
      -> embedding validation
      -> ProcessMemoryResponse

Like the search service this layer only composes; every rule lives in the
component that owns it.

Reprocessing returns the COMPLETE current chunk set, never a diff and never
only the changed chunks (contract section 13). This service is stateless with
respect to previous versions of a Memory: it holds no prior chunk set and
computes no delta. The Backend reconciles old against new.

Notes and Links follow the same path once the canonical text exists. For a
Link the page content was extracted by the **Backend** and arrives in
``extractedContent``; this service never fetches a URL, scrapes a page or
crawls anything (contract section 19).

What this service must never do: insert or update MongoDB, insert, update or
delete vectors, perform authentication or authorization, or reach out to the
network. It returns chunks and embeddings; the Backend persists them.
"""

import logging

from memovo_ai.chunking.hybrid_chunker import HybridChunker
from memovo_ai.chunking.models import Chunk
from memovo_ai.core.errors import AiServiceError
from memovo_ai.core.hashing import chunk_id
from memovo_ai.core.logging import log_event, safe_text, timed
from memovo_ai.embeddings.base import EmbeddingProvider
from memovo_ai.schemas.errors import ErrorCode
from memovo_ai.schemas.process import (
    LinkProcessRequest,
    LinkSource,
    ProcessedChunk,
    ProcessMemoryRequest,
    ProcessMemoryResponse,
)
from memovo_ai.understanding.content import compose_canonical_content

__all__ = ["MemoryProcessorService"]

_logger = logging.getLogger(__name__)


def _embeddable_source(source: LinkSource) -> list[str]:
    """The source metadata worth putting in front of the embedder.

    All four AI-facing fields (contract section 8.3), in a fixed order: they
    are natural-language-ish and help retrieval ("what did I save from the
    MongoDB blog?", "the article by Jane Doe"). ``null`` and blank values are
    skipped, never rendered as ``None``.
    """
    return [
        value
        for value in (
            source.source_title,
            source.source_description,
            source.author_name,
            source.publication_date,
        )
        if value
    ]


def _canonical_content(request: ProcessMemoryRequest) -> str:
    """Compose the text to chunk, per memory type.

    A Note contributes title, content and tags. A Link adds the Backend's
    source metadata and extracted page text (contract section 8.4). The
    ``url`` is not embedded -- see :mod:`memovo_ai.understanding.content`.
    """
    if isinstance(request, LinkProcessRequest):
        return compose_canonical_content(
            title=request.title,
            content=request.content,
            tags=request.tags,
            source=_embeddable_source(request.source),
            extracted_content=request.extracted_content,
        )

    return compose_canonical_content(
        title=request.title,
        content=request.content,
        tags=request.tags,
    )


class MemoryProcessorService:
    """Answers ``POST /ai/memories/process``."""

    __slots__ = ("_chunker", "_embeddings")

    def __init__(
        self,
        *,
        embedding_provider: EmbeddingProvider,
        chunker: HybridChunker | None = None,
    ) -> None:
        self._embeddings = embedding_provider
        self._chunker = chunker if chunker is not None else HybridChunker()

    async def process(self, request: ProcessMemoryRequest) -> ProcessMemoryResponse:
        """Chunk and embed a Memory, returning its complete current chunk set."""
        content = _canonical_content(request)

        with timed() as chunking_elapsed:
            chunks = self._chunk(content)

        if not chunks:
            # Every field normalized to nothing, so there is nothing to
            # index. The contract permits an empty chunk set (section 11),
            # and rejecting the request would invent a validation rule.
            self._log_processed(
                request.memory_id,
                content_chars=len(content),
                chunk_count=0,
                chunking_ms=chunking_elapsed.ms,
                embedding_ms=0.0,
            )
            return ProcessMemoryResponse(memoryId=request.memory_id, chunks=[])

        # One batched call: the provider decides how to split it into forward
        # passes.
        with timed() as embedding_elapsed:
            embeddings = await self._embeddings.embed_documents([chunk.content for chunk in chunks])

        self._log_processed(
            request.memory_id,
            content_chars=len(content),
            chunk_count=len(chunks),
            chunking_ms=chunking_elapsed.ms,
            embedding_ms=embedding_elapsed.ms,
        )

        if len(embeddings) != len(chunks):
            message = f"expected {len(chunks)} embeddings, got {len(embeddings)}"
            raise AiServiceError(ErrorCode.EMBEDDING_FAILED, message=message)

        return ProcessMemoryResponse(
            memoryId=request.memory_id,
            chunks=[
                ProcessedChunk(
                    chunkId=chunk_id(
                        memory_id=request.memory_id,
                        chunk_index=chunk.index,
                        # The final chunk content, exactly as returned below.
                        # Hashing anything else would break the guarantee that
                        # changed content changes the ID.
                        content=chunk.content,
                    ),
                    chunkIndex=chunk.index,
                    content=chunk.content,
                    embedding=embedding,
                )
                for chunk, embedding in zip(chunks, embeddings, strict=True)
            ],
        )

    def _log_processed(
        self,
        memory_id: str,
        *,
        content_chars: int,
        chunk_count: int,
        chunking_ms: float,
        embedding_ms: float,
    ) -> None:
        """Record the shape and cost of one ingestion.

        Sizes and durations only. The canonical content and the chunk text
        are precisely what the privacy rules forbid logging; their character
        count is what explains a slow request without revealing it.

        ``memoryId`` is an opaque Backend identifier rather than user
        content, so it is logged as-is -- but truncated, because the contract
        sets no length limit on it and an over-long value must not be able to
        turn a successful ingestion into a logging failure.
        """
        log_event(
            _logger,
            "memory.processed",
            memory_id=safe_text(memory_id),
            content_chars=content_chars,
            chunk_count=chunk_count,
            chunking_ms=chunking_ms,
            embedding_ms=embedding_ms,
        )

    def _chunk(self, content: str) -> list[Chunk]:
        try:
            return self._chunker.chunk(content)
        except AiServiceError:
            raise
        except Exception as error:
            raise AiServiceError(ErrorCode.CHUNKING_FAILED) from error
