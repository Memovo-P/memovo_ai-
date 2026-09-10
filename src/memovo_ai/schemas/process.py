"""Schemas for ``POST /ai/memories/process``.

One request shape covers both Notes and Links. A Link is a request that
carries a ``source`` object; the contract requires that object to be present
on every Link request, so its presence is the discriminator and no ``type``
field is introduced.

Link page content is **extracted by the Backend** and arrives in
``description``. This service never fetches a URL, scrapes a page or crawls
anything -- it only processes the text it is given.
"""

from typing import Literal

from pydantic import Field

from memovo_ai.schemas.common import BaseChunk, MemovoBaseModel

__all__ = [
    "PROCESS_INTENT",
    "LinkSource",
    "ProcessMemoryRequest",
    "ProcessMemoryResponse",
    "ProcessedChunk",
]

#: The endpoint identifies the operation, so the intent is deterministic.
#: There is no classifier in Sprint 1 (see doc 01, section 6).
PROCESS_INTENT: Literal["save_memory"] = "save_memory"


class LinkSource(MemovoBaseModel):
    """Provenance metadata for a saved Link.

    Every field is optional and may be ``null``: the Backend fills in what it
    could extract at save time and nothing more (doc 01, locked decision 29).
    Missing values are handled gracefully and never cause an error.

    Only the four fields named by the contract exist. Undocumented extras are
    rejected like everywhere else in this contract, so adding a fifth is a
    deliberate contract change rather than a silent one.
    """

    site_name: str | None = Field(default=None, alias="siteName")
    favicon: str | None = None
    og_image: str | None = Field(default=None, alias="ogImage")
    published_at: str | None = Field(default=None, alias="publishedAt")


class ProcessMemoryRequest(MemovoBaseModel):
    """A Memory submitted by the Backend for chunking and embedding.

    Notes and Links share this shape. For a Link, ``description`` holds the
    page content the **Backend** already extracted, and ``source`` is present.

    ``userId`` is intentionally absent: processing needs no user context, and
    the Backend owns identity. ``embeddingVersion``, ``jobId`` and timestamps
    are not part of the Sprint 1 contract (doc 01, locked decision 30).
    """

    memory_id: str = Field(alias="memoryId")
    title: str
    description: str
    why_saved: str = Field(alias="whySaved")
    tags: list[str]

    #: Link context written by the user. Optional, and may be absent, empty or
    #: ``null`` (doc 01, locked decision 28).
    about: str | None = None

    #: Present on Link requests, absent on Notes.
    source: LinkSource | None = None

    @property
    def is_link(self) -> bool:
        """A Link is a request carrying a ``source`` object."""
        return self.source is not None


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
