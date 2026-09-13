"""Public API contract schemas for the Memovo AI Service.

These models define the externally observable JSON shapes for
``POST /ai/memories/process`` and ``POST /ai/memories/search``, plus the
standardized error envelope. They are the contract boundary: field names,
required fields and literal values here are locked by the integration
contract (``docs/AI_CONTRACT_FINAL.md``) and must not drift.
"""

from memovo_ai.schemas.chat import (
    ChatMemoriesRequest,
    ChatMemoriesResponse,
    ChatSource,
    ChatSourceChunk,
    HistoryMessage,
)
from memovo_ai.schemas.common import MemovoBaseModel
from memovo_ai.schemas.errors import ErrorCode, ErrorDetail, ErrorResponse
from memovo_ai.schemas.prepare_note import PrepareNoteRequest, PrepareNoteResponse
from memovo_ai.schemas.process import (
    MEMORY_TYPE_LINK,
    MEMORY_TYPE_NOTE,
    LinkProcessRequest,
    LinkSource,
    NoteProcessRequest,
    ProcessedChunk,
    ProcessMemoryRequest,
    ProcessMemoryRequestAdapter,
    ProcessMemoryResponse,
)
from memovo_ai.schemas.search import (
    SearchMemoryNoMatchResponse,
    SearchMemoryRequest,
    SearchMemoryResponse,
    SearchMemoryResponseAdapter,
    SearchMemorySuccessResponse,
    SearchResult,
    SearchResultChunk,
)

__all__ = [
    "MEMORY_TYPE_LINK",
    "MEMORY_TYPE_NOTE",
    "ChatMemoriesRequest",
    "ChatMemoriesResponse",
    "ChatSource",
    "ChatSourceChunk",
    "ErrorCode",
    "ErrorDetail",
    "ErrorResponse",
    "HistoryMessage",
    "LinkProcessRequest",
    "LinkSource",
    "MemovoBaseModel",
    "NoteProcessRequest",
    "PrepareNoteRequest",
    "PrepareNoteResponse",
    "ProcessMemoryRequest",
    "ProcessMemoryRequestAdapter",
    "ProcessMemoryResponse",
    "ProcessedChunk",
    "SearchMemoryNoMatchResponse",
    "SearchMemoryRequest",
    "SearchMemoryResponse",
    "SearchMemoryResponseAdapter",
    "SearchMemorySuccessResponse",
    "SearchResult",
    "SearchResultChunk",
]
