"""Public API contract schemas for the Memovo AI Service.

These models define the externally observable JSON shapes for
``POST /ai/memories/process`` and ``POST /ai/memories/search``, plus the
standardized error envelope. They are the contract boundary: field names,
required fields and literal values here are locked by the integration
contract and must not drift.
"""

from memovo_ai.schemas.common import BaseChunk, MemovoBaseModel
from memovo_ai.schemas.errors import ErrorCode, ErrorDetail, ErrorResponse
from memovo_ai.schemas.process import (
    PROCESS_INTENT,
    LinkSource,
    ProcessedChunk,
    ProcessMemoryRequest,
    ProcessMemoryResponse,
)
from memovo_ai.schemas.search import (
    NO_MATCH_MESSAGE,
    SearchMemoryNoMatchResponse,
    SearchMemoryRequest,
    SearchMemoryResponse,
    SearchMemoryResponseAdapter,
    SearchMemorySuccessResponse,
    SearchResult,
    SearchResultChunk,
)

__all__ = [
    "NO_MATCH_MESSAGE",
    "PROCESS_INTENT",
    "BaseChunk",
    "ErrorCode",
    "ErrorDetail",
    "ErrorResponse",
    "LinkSource",
    "MemovoBaseModel",
    "ProcessMemoryRequest",
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
