"""``POST /ai/memories/search``.

Transport only::

    HTTP -> schema validation -> application service -> schema serialization

No retrieval algorithm lives here: no embedding call, no threshold, no
grouping, no ranking. The route validates, delegates, and returns.
"""

from fastapi import APIRouter, status

from memovo_ai.api.dependencies import SearchService
from memovo_ai.schemas.search import (
    SearchMemoryNoMatchResponse,
    SearchMemoryRequest,
    SearchMemorySuccessResponse,
)

__all__ = ["router"]

router = APIRouter(prefix="/ai/memories", tags=["memories"])


@router.post("/search", status_code=status.HTTP_200_OK)
async def search_memories(
    request: SearchMemoryRequest,
    service: SearchService,
) -> SearchMemorySuccessResponse | SearchMemoryNoMatchResponse:
    """Retrieve the Memories relevant to a query for its user.

    ``userId`` in the body is trusted service-to-service input from the
    authenticated Backend. This service does not decode JWTs, authenticate,
    or authorize (doc 06, sections 1-2).
    """
    return await service.search(request)
