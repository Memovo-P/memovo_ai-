"""``POST /ai/memories/prepare-note``.

Transport only: validate, delegate, return.
"""

from fastapi import APIRouter, status

from memovo_ai.api.dependencies import NotePreparation
from memovo_ai.schemas.prepare_note import PrepareNoteRequest, PrepareNoteResponse

__all__ = ["router"]

router = APIRouter(prefix="/ai/memories", tags=["memories"])


@router.post("/prepare-note", status_code=status.HTTP_200_OK)
async def prepare_note(
    request: PrepareNoteRequest, service: NotePreparation
) -> PrepareNoteResponse:
    """Prepare a title and cleaned content for an explicitly requested Note.

    Nothing is persisted here; the Backend validates the result and creates
    the Note itself (contract section 6.3).
    """
    return await service.prepare(request)
