"""Schemas for ``POST /ai/chat/memories`` (contract v1.9, section 17).

Request: ``userId`` and ``message`` are required; ``conversationId`` and
``history`` are optional and, per the plan, optional does **not** mean
nullable -- an explicit ``null`` is rejected, absence is fine.

History is accepted as the Backend bounded it (section 17.2.1). The AI
Service does not trim it, does not partially truncate a message and does not
enforce the 30 / 6,000 / 30,000 limits: those are Backend responsibilities,
and enforcing them here would silently second-guess the owner of the
conversation. Only the *shape* is validated: ``role`` is ``user`` or
``assistant`` and ``content`` is a string.

Response: ``found``, ``answer``, ``sources`` and nothing else (sections 17.3,
17.4.1, 17.4.2). No ``sufficient`` or ``confidence`` field, no citation ids.
"""

from typing import Literal

from pydantic import Field, model_validator

from memovo_ai.schemas.common import MemovoBaseModel

__all__ = [
    "ChatMemoriesRequest",
    "ChatMemoriesResponse",
    "ChatSource",
    "ChatSourceChunk",
    "HistoryMessage",
]


class HistoryMessage(MemovoBaseModel):
    """One prior turn. ``system`` and every other role are rejected."""

    role: Literal["user", "assistant"]
    content: str


class ChatMemoriesRequest(MemovoBaseModel):
    """A Memory Chat turn from the authenticated Backend."""

    user_id: str = Field(alias="userId")
    message: str
    #: Identifier only: never used to load or store anything, never used to
    #: change retrieval (section 17.2.2).
    conversation_id: str | None = Field(default=None, alias="conversationId")
    history: list[HistoryMessage] | None = None

    @model_validator(mode="after")
    def _optional_is_not_nullable(self) -> "ChatMemoriesRequest":
        for field in ("conversation_id", "history"):
            if field in self.model_fields_set and getattr(self, field) is None:
                message = f"{field} may be omitted but must not be null"
                raise ValueError(message)

        return self


class ChatSourceChunk(MemovoBaseModel):
    chunk_id: str = Field(alias="chunkId", min_length=1)
    content: str


class ChatSource(MemovoBaseModel):
    """A memory actually used by the answer (section 17.4.2)."""

    memory_id: str = Field(alias="memoryId", min_length=1)
    score: float = Field(allow_inf_nan=False)
    title: str
    chunks: list[ChatSourceChunk]


class ChatMemoriesResponse(MemovoBaseModel):
    """``found`` / ``answer`` / ``sources``; the shape never grows."""

    found: bool
    answer: str
    sources: list[ChatSource]
