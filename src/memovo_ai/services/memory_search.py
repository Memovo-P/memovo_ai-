"""Memory retrieval orchestration.

The pipeline from doc 03, Phase 13::

    query
      -> embed_query
      -> VectorSearchProvider.search(user_id, top_k)
      -> threshold >= 0.75
      -> group by memoryId
      -> memory score = MAX(chunk score)
      -> descending ranking
      -> SearchMemoryResponse

This layer only composes; every rule lives in the component that owns it.
Chunking never happens at search time -- retrieval runs over chunks indexed
during ingestion (doc 02, section 3).

What this service must never do (doc 03, Phase 07 and doc 06): write to
MongoDB or the vector database, decode a JWT, authenticate, or authorize.
``userId`` is trusted service-to-service input from the Backend.
"""

import logging

from memovo_ai.core.config import LoggingSettings, SearchSettings
from memovo_ai.core.logging import hash_user_id, log_event, timed
from memovo_ai.embeddings.base import EmbeddingProvider
from memovo_ai.providers.vector_search.base import (
    UserScopedVectorSearchProvider,
    VectorSearchProvider,
)
from memovo_ai.retrieval.filtering import filter_by_threshold
from memovo_ai.retrieval.models import RankedMemory
from memovo_ai.retrieval.ranking import rank_memories
from memovo_ai.schemas.search import (
    SearchMemoryNoMatchResponse,
    SearchMemoryRequest,
    SearchMemoryResponse,
    SearchMemorySuccessResponse,
    SearchResult,
    SearchResultChunk,
)

__all__ = ["MemorySearchService"]

_logger = logging.getLogger(__name__)


class MemorySearchService:
    """Answers ``POST /ai/memories/search``."""

    __slots__ = ("_embeddings", "_settings", "_user_salt", "_vector_search")

    def __init__(
        self,
        *,
        embedding_provider: EmbeddingProvider,
        vector_search: VectorSearchProvider,
        settings: SearchSettings | None = None,
        logging_settings: LoggingSettings | None = None,
    ) -> None:
        self._embeddings = embedding_provider
        self._settings = settings if settings is not None else SearchSettings()

        resolved_logging = logging_settings if logging_settings is not None else LoggingSettings()
        self._user_salt = resolved_logging.user_salt.get_secret_value()

        # Isolation is wrapped on here rather than left to the caller. Doc 03
        # Phase 10 requires *every* vector query to be user-scoped, and a
        # guarantee that depends on remembering to opt in is not a guarantee.
        # Wrapping an already-wrapped provider merely verifies twice.
        self._vector_search = UserScopedVectorSearchProvider(vector_search)

    async def search(self, request: SearchMemoryRequest) -> SearchMemoryResponse:
        """Retrieve the Memories relevant to ``request.query`` for its user."""
        with timed() as embedding_elapsed:
            query_embedding = await self._embeddings.embed_query(request.query)

        with timed() as search_elapsed:
            hits = await self._vector_search.search(
                user_id=request.user_id,
                query_embedding=query_embedding,
                top_k=self._settings.top_k,
            )

        relevant = filter_by_threshold(hits, threshold=self._settings.similarity_threshold)
        ranked = rank_memories(relevant)

        # Counts and durations only. The query, the hits and the chunk text
        # they carry are exactly what doc 06 section 5 forbids logging; the
        # query's length is kept because it explains a slow embedding call
        # without revealing what was asked.
        log_event(
            _logger,
            "memory.searched",
            user=hash_user_id(request.user_id, salt=self._user_salt),
            query_chars=len(request.query),
            top_k=self._settings.top_k,
            threshold=self._settings.similarity_threshold,
            embedding_ms=embedding_elapsed.ms,
            vector_search_ms=search_elapsed.ms,
            hit_count=len(hits),
            result_count=len(ranked),
            matched=bool(ranked),
        )

        if not ranked:
            return SearchMemoryNoMatchResponse()

        return SearchMemorySuccessResponse(results=[_to_result(memory) for memory in ranked])


def _to_result(memory: RankedMemory) -> SearchResult:
    """Map a ranked Memory onto the public contract.

    The mapping is explicit field by field, which is what keeps internal
    fields -- ``user_id`` on a hit above all -- out of the response.
    """
    return SearchResult(
        memoryId=memory.memory_id,
        score=memory.score,
        title=memory.title,
        tags=list(memory.tags),
        chunks=[
            SearchResultChunk(
                chunkId=chunk.chunk_id,
                chunkIndex=chunk.chunk_index,
                content=chunk.content,
            )
            for chunk in memory.chunks
        ],
    )
