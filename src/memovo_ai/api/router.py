"""Route registration.

Routers are aggregated here so the application object stays free of endpoint
detail.
"""

from fastapi import APIRouter

from memovo_ai.api.routes import (
    health,
    memory_chat,
    memory_process,
    memory_search,
    prepare_note,
)

__all__ = ["api_router"]

api_router = APIRouter()
api_router.include_router(memory_process.router)
api_router.include_router(memory_search.router)
api_router.include_router(memory_chat.router)
api_router.include_router(prepare_note.router)

# Operational probes, outside /ai/ and outside the Backend contract.
api_router.include_router(health.router)
