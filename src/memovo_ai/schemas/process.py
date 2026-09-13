"""Schemas for ``POST /ai/memories/process`` (contract v1.9, sections 8, 9, 11).

The request is a union discriminated on ``type``: a Note (section 8.1) or a
Link (section 8.2). Every limit below is copied from the contract's tables and
none is invented here; where the contract sets no limit -- ``memoryId``,
``extractedContent`` -- none is enforced.

Link page content is **extracted by the Backend** and arrives in
``extractedContent``. This service never fetches a URL, scrapes a page or
crawls anything -- it only processes the text it is given. ``url`` is context,
not an instruction to fetch (sections 8.4 and 19).

``whySaved`` is permanently removed (section 8.5). It is not accepted, not
embedded and not mapped onto any other field; with ``extra="forbid"`` a request
still carrying it is rejected with 422, as is any other legacy field.
"""

from typing import Annotated, Literal

from pydantic import Field, TypeAdapter, field_validator

from memovo_ai.schemas.common import MemovoBaseModel

__all__ = [
    "LINK_CONTENT_MAX_LENGTH",
    "LINK_TITLE_MAX_LENGTH",
    "LINK_TITLE_MIN_LENGTH",
    "LINK_URL_MAX_LENGTH",
    "MAX_TAGS",
    "MEMORY_TYPE_LINK",
    "MEMORY_TYPE_NOTE",
    "NOTE_CONTENT_MAX_LENGTH",
    "NOTE_CONTENT_MIN_LENGTH",
    "NOTE_TITLE_MAX_LENGTH",
    "TAG_MAX_LENGTH",
    "LinkProcessRequest",
    "LinkSource",
    "NoteProcessRequest",
    "ProcessMemoryRequest",
    "ProcessMemoryRequestAdapter",
    "ProcessMemoryResponse",
    "ProcessedChunk",
]

#: The discriminator values (contract section 7).
MEMORY_TYPE_NOTE: Literal["note"] = "note"
MEMORY_TYPE_LINK: Literal["link"] = "link"

#: Contract section 8.1.
NOTE_TITLE_MAX_LENGTH = 200
NOTE_CONTENT_MIN_LENGTH = 1
NOTE_CONTENT_MAX_LENGTH = 10_000

#: Contract section 8.2.
LINK_URL_MAX_LENGTH = 2048
LINK_TITLE_MIN_LENGTH = 1
LINK_TITLE_MAX_LENGTH = 500
LINK_CONTENT_MAX_LENGTH = 1000

#: Contract sections 8.1 and 8.2; the same rule for both types.
MAX_TAGS = 20
TAG_MAX_LENGTH = 50

#: The schemes the Backend's own URL validation admits (D0, final decisions).
#: Checked as a prefix only -- nothing here parses, resolves or opens a URL.
_WEB_SCHEMES = ("http://", "https://")

#: Optional and nullable on both types: absent and ``null`` both mean "no
#: tags". Bounded in count and per-tag length by the contract. The count
#: bound sits on the list, inside the optional, so ``null`` is not measured.
type _Tags = (
    Annotated[
        list[Annotated[str, Field(max_length=TAG_MAX_LENGTH)]],
        Field(max_length=MAX_TAGS),
    ]
    | None
)


class LinkSource(MemovoBaseModel):
    """The four AI-facing source fields (contract section 8.3).

    Each is **required and nullable**: the key must be present, and its value
    may be ``null``. ``{}`` is therefore invalid. So is any other key --
    ``platform``, ``contentType``, ``thumbnailUrl`` and ``canonicalUrl`` are
    Backend-only metadata that must not reach this service, and the legacy
    ``siteName``/``favicon``/``ogImage``/``publishedAt`` shape is gone.

    ``publicationDate`` is a string; the contract allows ISO 8601 *or another
    string representation*, so no date format is enforced.
    """

    source_title: str | None = Field(alias="sourceTitle")
    source_description: str | None = Field(alias="sourceDescription")
    author_name: str | None = Field(alias="authorName")
    publication_date: str | None = Field(alias="publicationDate")


class NoteProcessRequest(MemovoBaseModel):
    """A persisted Note submitted for chunking and embedding (section 8.1).

    ``userId`` is intentionally absent: processing needs no user context, and
    the Backend owns identity. ``jobId``, timestamps and ``embeddingVersion``
    are not part of the contract either.
    """

    memory_type: Literal["note"] = Field(alias="type")
    memory_id: str = Field(alias="memoryId", min_length=1)
    #: 0-200 characters: an empty title is valid.
    title: str = Field(max_length=NOTE_TITLE_MAX_LENGTH)
    #: 1-10,000 characters: a Note without content is not.
    content: str = Field(min_length=NOTE_CONTENT_MIN_LENGTH, max_length=NOTE_CONTENT_MAX_LENGTH)
    tags: _Tags = None


class LinkProcessRequest(MemovoBaseModel):
    """A persisted Link submitted for chunking and embedding (section 8.2).

    Three different texts arrive, and they are kept apart:

    - ``content`` is the **user's own** context about the Link -- optional,
      nullable, may be empty.
    - ``extractedContent`` is the page text the **Backend** extracted --
      optional, nullable, may be empty, and carries no AI-side length limit
      because the Backend controls extraction and payload size (section 8.4).
    - ``source`` is the Backend-extracted metadata (section 8.3).

    None of them is a replacement for the removed ``whySaved``.
    """

    memory_type: Literal["link"] = Field(alias="type")
    memory_id: str = Field(alias="memoryId", min_length=1)
    url: str = Field(max_length=LINK_URL_MAX_LENGTH)
    title: str = Field(min_length=LINK_TITLE_MIN_LENGTH, max_length=LINK_TITLE_MAX_LENGTH)
    content: str | None = Field(default=None, max_length=LINK_CONTENT_MAX_LENGTH)
    tags: _Tags = None
    source: LinkSource
    extracted_content: str | None = Field(default=None, alias="extractedContent")

    @field_validator("url")
    @classmethod
    def _require_a_web_scheme(cls, value: str) -> str:
        """Accept only what the Backend's URL validation admits.

        A prefix check and nothing more. The value is never parsed for a host,
        never resolved and never opened; it is stored as the opaque context
        string it is (section 19).
        """
        if not value.lower().startswith(_WEB_SCHEMES):
            message = "url must use the http or https scheme"
            raise ValueError(message)

        return value


#: The request body: exactly one of the two shapes, selected by ``type``.
type ProcessMemoryRequest = Annotated[
    NoteProcessRequest | LinkProcessRequest,
    Field(discriminator="memory_type"),
]

#: Validates either shape from raw JSON or a mapping.
ProcessMemoryRequestAdapter: TypeAdapter[ProcessMemoryRequest] = TypeAdapter(ProcessMemoryRequest)


class ProcessedChunk(MemovoBaseModel):
    """One chunk of the Memory, with its document embedding (section 11).

    The embedding is typed only as a list of floats. Dimension, finiteness and
    NaN/Infinity checks belong to the embedding provider boundary, not to the
    transport schema.
    """

    chunk_id: str = Field(alias="chunkId")
    chunk_index: int = Field(alias="chunkIndex", ge=0)
    content: str
    embedding: list[float]


class ProcessMemoryResponse(MemovoBaseModel):
    """The COMPLETE current chunk set for the Memory (section 9).

    ``memoryId`` echoes the request. Reprocessing always returns every chunk,
    never a diff and never only the changed chunks; the Backend reconciles by
    chunk ID (section 13).
    """

    memory_id: str = Field(alias="memoryId", min_length=1)
    chunks: list[ProcessedChunk]
