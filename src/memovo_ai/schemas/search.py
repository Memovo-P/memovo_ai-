"""Schemas for ``POST /ai/memories/search`` (contract v1.9, sections 14 and 16).

The request is ``{userId, query}`` and nothing else: the memory-type filter
and the public ``limit`` are Backend concerns applied after this response
(sections 14.1, 14.2 and 16.5), so neither is accepted here.

The success and no-match responses are modelled as two separate types joined by
a union discriminated on ``found``. ``found`` is already part of the wire
format, so this adds no discriminator field and does not alter the JSON shape.
Modelling them separately makes a contradictory payload -- ``found=false``
with a populated ``results`` list -- unrepresentable.

The per-result and per-chunk constraints are the ones section 16.4 tells the
Backend to validate. Enforcing them on the way out means a defect surfaces
here as a 500 rather than as a well-formed 200 carrying malformed data.
"""

from typing import Annotated, Literal

from pydantic import Field, TypeAdapter

from memovo_ai.schemas.common import MemovoBaseModel

__all__ = [
    "SearchMemoryNoMatchResponse",
    "SearchMemoryRequest",
    "SearchMemoryResponse",
    "SearchMemoryResponseAdapter",
    "SearchMemorySuccessResponse",
    "SearchResult",
    "SearchResultChunk",
]


class SearchMemoryRequest(MemovoBaseModel):
    """A retrieval request from the authenticated Backend (section 14.1).

    ``userId`` is trusted service-to-service input. This service does not
    decode JWTs, authenticate, or infer identity; the schema only validates the
    shape.
    """

    query: str
    user_id: str = Field(alias="userId")


class SearchResultChunk(MemovoBaseModel):
    """A matching chunk returned to the Backend (section 16.1).

    ``chunkId`` and ``content`` only. Embeddings are never exposed through
    search results, and ``chunkIndex`` stays internal: it still orders the
    chunks within a result, but is not part of this response.
    """

    chunk_id: str = Field(alias="chunkId", min_length=1)
    content: str = Field(min_length=1)


class SearchResult(MemovoBaseModel):
    """One unique Memory in the retrieval result set.

    ``score`` is the maximum score across the Memory's matching chunks. No
    numeric range is enforced: the contract describes it as *typically*
    0.0-1.0, and bounding it here would hard-code an assumption about the
    similarity metric.
    """

    memory_id: str = Field(alias="memoryId", min_length=1)
    score: float = Field(allow_inf_nan=False)
    title: str
    tags: list[str]
    chunks: list[SearchResultChunk]


class SearchMemorySuccessResponse(MemovoBaseModel):
    """Retrieval response carrying matched Memories."""

    found: Literal[True] = True
    results: list[SearchResult]


class SearchMemoryNoMatchResponse(MemovoBaseModel):
    """Retrieval response when no chunk meets the similarity threshold.

    Exactly ``{"found": false, "results": []}`` (section 16.1). ``results`` is
    constrained to empty, so the only valid instance of this model is the
    contract's no-match payload.
    """

    found: Literal[False] = False
    results: list[SearchResult] = Field(default_factory=list, max_length=0)


type SearchMemoryResponse = Annotated[
    SearchMemorySuccessResponse | SearchMemoryNoMatchResponse,
    Field(discriminator="found"),
]

#: Validates either arm of the response union from raw JSON or a mapping.
SearchMemoryResponseAdapter: TypeAdapter[SearchMemoryResponse] = TypeAdapter(SearchMemoryResponse)
