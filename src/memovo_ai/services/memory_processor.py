"""Memory ingestion orchestration.

The pipeline from doc 03, Phase 07::

    ProcessMemoryRequest
      -> canonical content composition
      -> hybrid chunker
      -> deterministic chunk IDs
      -> batch embeddings
      -> embedding validation
      -> ProcessMemoryResponse

Like the search service this layer only composes; every rule lives in the
component that owns it.

Reprocessing returns the COMPLETE current chunk set, never a diff and never
only the changed chunks (doc 01, decision 25). This service is stateless with
respect to previous versions of a Memory: it holds no prior chunk set and
computes no delta. The Backend reconciles old against new.

Notes and Links follow the same path. For a Link the page content was
extracted by the **Backend** and arrives in ``description``; this service
never fetches a URL, scrapes a page or crawls anything.

What this service must never do (doc 03, Phase 07): insert or update MongoDB,
insert, update or delete vectors, perform authentication or authorization, or
reach out to the network.
It returns chunks and embeddings; the Backend persists them.
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
    LinkSource,
    ProcessedChunk,
    ProcessMemoryRequest,
    ProcessMemoryResponse,
)
from memovo_ai.understanding.content import compose_canonical_content

__all__ = ["MemoryProcessorService"]

_logger = logging.getLogger(__name__)


def _embeddable_source(source: LinkSource | None) -> list[str]:
    """Pick the provenance fields worth putting in front of the embedder.

    ``siteName`` and ``publishedAt`` are natural-language-ish and help
    retrieval ("what did I save from the MongoDB blog?").

    ``favicon`` and ``ogImage`` are deliberately excluded. They are URLs to
    binary assets: they carry no semantic meaning, and embedding them would
    dilute the chunk with tokens that can never match a user's query.

    Missing and ``null`` fields are skipped, never rendered as "None".
    """
    if source is None:
        return []

    return [value for value in (source.site_name, source.published_at) if value]


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
        content = compose_canonical_content(
            title=request.title,
            description=request.description,
            why_saved=request.why_saved,
            tags=request.tags,
            about=request.about,
            source=_embeddable_source(request.source),
        )

        with timed() as chunking_elapsed:
            chunks = self._chunk(content)

        if not chunks:
            # Every field was empty, so there is nothing to index. The
            # contract permits an empty chunk set, and rejecting the request
            # would invent a validation rule no source document states.
            self._log_processed(
                request.memory_id,
                content_chars=len(content),
                chunk_count=0,
                chunking_ms=chunking_elapsed.ms,
                embedding_ms=0.0,
            )
            return ProcessMemoryResponse(content=content, chunks=[])

        # One batched call: the provider decides how to split it into forward
        # passes (doc 02, section 7).
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
            content=content,
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
        are precisely what doc 06 section 5 forbids logging; their character
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
