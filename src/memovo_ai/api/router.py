"""Route registration.

Routers are aggregated here so the application object stays free of endpoint
detail.
"""

from fastapi import APIRouter

from memovo_ai.api.routes import health, memory_process, memory_search

__all__ = ["api_router"]

api_router = APIRouter()
api_router.include_router(memory_process.router)
api_router.include_router(memory_search.router)

# Operational probes, outside /ai/ and outside the Backend contract.
api_router.include_router(health.router)
