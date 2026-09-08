"""``POST /ai/memories/process``.

Transport only::

    HTTP -> schema validation -> application service -> schema serialization

No chunking, hashing, embedding or persistence logic lives here (doc 03,
Phase 08). The route validates, delegates, and returns.
"""

from fastapi import APIRouter, status

from memovo_ai.api.dependencies import ProcessorService
from memovo_ai.schemas.process import ProcessMemoryRequest, ProcessMemoryResponse

__all__ = ["router"]

router = APIRouter(prefix="/ai/memories", tags=["memories"])


@router.post("/process", status_code=status.HTTP_200_OK)
async def process_memory(
    request: ProcessMemoryRequest,
    service: ProcessorService,
) -> ProcessMemoryResponse:
    """Chunk and embed a Memory, returning its complete current chunk set.

    Reprocessing returns every chunk, never a diff; the Backend reconciles
    against what it already stored (doc 01, decision 25).
    """
    return await service.process(request)
