"""``POST /ai/memories/process``.

Transport only::

    HTTP -> schema validation -> application service -> schema serialization

No chunking, hashing, embedding or persistence logic lives here. The route
validates, delegates, and returns.
"""

from typing import Annotated

from fastapi import APIRouter, Body, status

from memovo_ai.api.dependencies import ProcessorService
from memovo_ai.schemas.process import (
    LinkProcessRequest,
    NoteProcessRequest,
    ProcessMemoryResponse,
)

__all__ = ["router"]

router = APIRouter(prefix="/ai/memories", tags=["memories"])


@router.post("/process", status_code=status.HTTP_200_OK)
async def process_memory(
    # The body is one of two shapes, selected by ``type`` (contract section
    # 8). Spelling the discriminator here keeps the validation error for a
    # Note request scoped to the Note schema, instead of a union of both.
    request: Annotated[NoteProcessRequest | LinkProcessRequest, Body(discriminator="memory_type")],
    service: ProcessorService,
) -> ProcessMemoryResponse:
    """Chunk and embed a Memory, returning its complete current chunk set.

    Reprocessing returns every chunk, never a diff; the Backend reconciles
    against what it already stored (contract section 13).
    """
    return await service.process(request)
