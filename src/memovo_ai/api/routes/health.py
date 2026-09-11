"""Liveness and readiness probes (doc 06, sections 8 and 9).

Operational endpoints, deliberately **not** part of the AI <-> Backend
contract. They live outside ``/ai/`` so they cannot be mistaken for it, and
their response models stay here rather than in ``schemas/``, which is
reserved for the contract boundary.

Liveness and readiness answer different questions, and conflating them is the
classic way to turn a degraded service into a restart loop:

``GET /health``
    Is this process alive and serving? Always ``200`` while it is. It does
    **not** consult the embedding model: if a failed model load made the
    process look dead, an orchestrator would kill and recreate it forever,
    and every replacement would fail the same way.

``GET /ready``
    Should this instance receive traffic? ``200`` only when the services are
    built; otherwise ``503`` with the state that explains why. This is what
    keeps an instance out of the load balancer while the model loads, and
    what doc 06 section 9 means by "not treated as ready before the embedding
    provider is usable".

Neither endpoint takes input, reads a header, or reveals configuration: no
model name, no provider, no connection detail. A probe response is the one
thing in the service that an unauthenticated network scanner is most likely
to reach.
"""

from typing import Literal

from fastapi import APIRouter, Request, Response, status

from memovo_ai.core.readiness import ReadinessState, readiness_of
from memovo_ai.schemas.common import MemovoBaseModel

__all__ = ["HealthResponse", "ReadinessResponse", "router"]

router = APIRouter(tags=["operations"])


class HealthResponse(MemovoBaseModel):
    """Liveness. One field, one value."""

    status: Literal["alive"] = "alive"


class ReadinessResponse(MemovoBaseModel):
    """Readiness, carrying the lifecycle state behind the status code."""

    status: ReadinessState


@router.get("/health", status_code=status.HTTP_200_OK)
async def health() -> HealthResponse:
    """Report that the process is alive.

    Independent of the model by design -- see the module docstring.
    """
    return HealthResponse()


@router.get("/ready")
async def ready(request: Request, response: Response) -> ReadinessResponse:
    """Report whether this instance should receive traffic.

    ``503`` rather than the standardized error envelope: this is not an AI
    endpoint failure, and a probe consumer reads the status code, not an
    error code. The envelope stays reserved for ``/ai/*``.
    """
    state = readiness_of(request.app.state)

    response.status_code = (
        status.HTTP_200_OK if state is ReadinessState.READY else status.HTTP_503_SERVICE_UNAVAILABLE
    )

    return ReadinessResponse(status=state)
