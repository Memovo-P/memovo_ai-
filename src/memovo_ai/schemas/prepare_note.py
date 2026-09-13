"""Schemas for ``POST /ai/memories/prepare-note`` (contract v1.9, section 6).

Request: the raw content the user explicitly chose to save. No
``conversationId``, no ``userId``: the operation is independent of the chat
and needs no identity.

Response: ``title`` and ``content`` only. No tags, identifiers or timestamps
(section 6.2); the Backend creates the Note with ``tags: []``.
"""

from memovo_ai.schemas.common import MemovoBaseModel

__all__ = ["PrepareNoteRequest", "PrepareNoteResponse"]


class PrepareNoteRequest(MemovoBaseModel):
    content: str


class PrepareNoteResponse(MemovoBaseModel):
    title: str
    content: str
