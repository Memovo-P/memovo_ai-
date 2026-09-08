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

What this service must never do (doc 03, Phase 07): insert or update MongoDB,
insert, update or delete vectors, or perform authentication or authorization.
It returns chunks and embeddings; the Backend persists them.
"""

from memovo_ai.chunking.hybrid_chunker import HybridChunker
from memovo_ai.chunking.models import Chunk
from memovo_ai.core.errors import AiServiceError
from memovo_ai.core.hashing import chunk_id
from memovo_ai.embeddings.base import EmbeddingProvider
from memovo_ai.schemas.errors import ErrorCode
from memovo_ai.schemas.process import (
    ProcessedChunk,
    ProcessMemoryRequest,
    ProcessMemoryResponse,
)
from memovo_ai.understanding.content import compose_canonical_content

__all__ = ["MemoryProcessorService"]


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
        )

        chunks = self._chunk(content)
        if not chunks:
            # Every field was empty, so there is nothing to index. The
            # contract permits an empty chunk set, and rejecting the request
            # would invent a validation rule no source document states.
            return ProcessMemoryResponse(content=content, chunks=[])

        # One batched call: the provider decides how to split it into forward
        # passes (doc 02, section 7).
        embeddings = await self._embeddings.embed_documents([chunk.content for chunk in chunks])

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

    def _chunk(self, content: str) -> list[Chunk]:
        try:
            return self._chunker.chunk(content)
        except AiServiceError:
            raise
        except Exception as error:
            raise AiServiceError(ErrorCode.CHUNKING_FAILED) from error
