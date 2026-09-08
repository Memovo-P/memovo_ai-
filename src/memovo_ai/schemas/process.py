"""Schemas for ``POST /ai/memories/process``.

Sprint 1 covers the Note request shape only. Link-specific fields (``about``
and the ``source`` object) are Phase 17 and are deliberately absent here.
"""

from typing import Literal

from pydantic import Field

from memovo_ai.schemas.common import BaseChunk, MemovoBaseModel

__all__ = [
    "PROCESS_INTENT",
    "ProcessMemoryRequest",
    "ProcessMemoryResponse",
    "ProcessedChunk",
]

#: The endpoint identifies the operation, so the intent is deterministic.
#: There is no classifier in Sprint 1 (see doc 01, section 6).
PROCESS_INTENT: Literal["save_memory"] = "save_memory"


class ProcessMemoryRequest(MemovoBaseModel):
    """A Memory submitted by the Backend for chunking and embedding.

    ``userId`` is intentionally absent: processing needs no user context, and
    the Backend owns identity. ``embeddingVersion``, ``jobId`` and timestamps
    are not part of the Sprint 1 contract (doc 01, locked decision 30).
    """

    memory_id: str = Field(alias="memoryId")
    title: str
    description: str
    why_saved: str = Field(alias="whySaved")
    tags: list[str]


class ProcessedChunk(BaseChunk):
    """One chunk of the Memory, with its document embedding.

    The embedding is typed only as a list of floats. Dimension, finiteness and
    NaN/Infinity checks belong to the embedding provider boundary (Phase 05),
    not to the transport schema -- see ``docs`` note in the phase report.
    """

    embedding: list[float]


class ProcessMemoryResponse(MemovoBaseModel):
    """The COMPLETE current chunk set for the Memory.

    Reprocessing always returns every chunk, never a diff and never only the
    changed chunks (doc 01, locked decision 25). The Backend reconciles.
    """

    intent: Literal["save_memory"] = PROCESS_INTENT
    content: str
    chunks: list[ProcessedChunk]
