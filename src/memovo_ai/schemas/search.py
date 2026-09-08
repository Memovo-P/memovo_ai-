"""Schemas for ``POST /ai/memories/search``.

The success and no-match responses are modelled as two separate types joined by
a union discriminated on ``found``. ``found`` is already part of the wire
format, so this adds no discriminator field and does not alter the JSON shape.

Modelling them separately makes contradictory payloads unrepresentable:
``found=false`` with a populated ``results`` list, or ``found=true`` carrying
the no-match ``message``, both fail validation.
"""

from typing import Annotated, Literal

from pydantic import Field, TypeAdapter

from memovo_ai.schemas.common import BaseChunk, MemovoBaseModel

__all__ = [
    "NO_MATCH_MESSAGE",
    "SearchMemoryNoMatchResponse",
    "SearchMemoryRequest",
    "SearchMemoryResponse",
    "SearchMemoryResponseAdapter",
    "SearchMemorySuccessResponse",
    "SearchResult",
    "SearchResultChunk",
]

#: Locked by doc 01, decision 20. Kept as a constant so the value is defined
#: in exactly one place.
NO_MATCH_MESSAGE: Literal["I couldn't find a relevant memory"] = "I couldn't find a relevant memory"


class SearchMemoryRequest(MemovoBaseModel):
    """A retrieval request from the authenticated Backend.

    ``userId`` is trusted service-to-service input. This service does not
    decode JWTs, authenticate, or infer identity; the schema only validates the
    shape (doc 06, sections 1-2).
    """

    query: str
    user_id: str = Field(alias="userId")


class SearchResultChunk(BaseChunk):
    """A matching chunk returned to the Backend.

    Inherits ``chunkId``, ``chunkIndex`` and ``content`` only. Embeddings are
    never exposed through search results.
    """


class SearchResult(MemovoBaseModel):
    """One unique Memory in the retrieval result set.

    ``score`` is the maximum score across the Memory's matching chunks (doc 01,
    locked decision 18). No numeric range is enforced: the similarity metric is
    recommended-but-not-locked (doc 01, section 7), so bounding the score here
    would hard-code an assumption the architecture has not made.
    """

    memory_id: str = Field(alias="memoryId")
    score: float
    title: str
    tags: list[str]
    chunks: list[SearchResultChunk]


class SearchMemorySuccessResponse(MemovoBaseModel):
    """Retrieval response carrying matched Memories."""

    found: Literal[True] = True
    results: list[SearchResult]


class SearchMemoryNoMatchResponse(MemovoBaseModel):
    """Retrieval response when no chunk meets the similarity threshold.

    ``results`` is constrained to empty and ``message`` to the locked contract
    string, so the only valid instance of this model is the exact no-match
    payload defined by the contract.
    """

    found: Literal[False] = False
    results: list[SearchResult] = Field(default_factory=list, max_length=0)
    message: Literal["I couldn't find a relevant memory"] = NO_MATCH_MESSAGE


type SearchMemoryResponse = Annotated[
    SearchMemorySuccessResponse | SearchMemoryNoMatchResponse,
    Field(discriminator="found"),
]

#: Validates either arm of the response union from raw JSON or a mapping.
SearchMemoryResponseAdapter: TypeAdapter[SearchMemoryResponse] = TypeAdapter(SearchMemoryResponse)
