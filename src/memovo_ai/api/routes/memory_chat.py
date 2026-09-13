"""``POST /ai/chat/memories``.

Transport only: validate, delegate, return. No retrieval, prompting or
generation logic lives here.
"""

from fastapi import APIRouter, status

from memovo_ai.api.dependencies import ChatService
from memovo_ai.schemas.chat import ChatMemoriesRequest, ChatMemoriesResponse

__all__ = ["router"]

router = APIRouter(prefix="/ai/chat", tags=["chat"])


@router.post("/memories", status_code=status.HTTP_200_OK)
async def chat_memories(request: ChatMemoriesRequest, service: ChatService) -> ChatMemoriesResponse:
    """Answer a question grounded only in the user's own memories.

    ``userId`` is trusted service-to-service input from the authenticated
    Backend; ``history`` arrives already bounded by the Backend and is not
    trimmed here; ``conversationId`` is an identifier and nothing more.
    """
    return await service.chat(request)
