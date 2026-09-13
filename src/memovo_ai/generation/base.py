"""The generation provider boundary.

Application services depend on :class:`GenerationProvider`, never on a
concrete API client. The OpenRouter adapter implements it; tests use the
deterministic fake.

:class:`BoundedGenerationProvider` is the only place concurrency and deadlines
are enforced, so a service cannot forget to apply them. It is itself a
provider, and therefore transparent to callers.

No retries, anywhere on this side. The Backend owns retry behaviour
(contract section 21.1); an automatic retry here would multiply its three
attempts into nine and double the cost of every failure.
"""

import asyncio
from typing import Protocol, runtime_checkable

from memovo_ai.generation.models import (
    GenerationRequest,
    GenerationResult,
    GenerationTimeoutError,
)

__all__ = ["BoundedGenerationProvider", "GenerationProvider"]


@runtime_checkable
class GenerationProvider(Protocol):
    """Completes one chat-style request."""

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """Return the completion, or raise a :class:`GenerationError`.

        Implementations make exactly one upstream attempt per call.
        """
        ...


class BoundedGenerationProvider:
    """Caps in-flight requests and bounds each one by a deadline.

    The deadline covers the wait for a slot as well as the call itself: a
    request queued behind a slow provider must not be allowed to wait
    forever and then take its full timeout on top. A timeout releases the
    slot, and so does cancellation -- the semaphore is held in a context
    manager, so nothing leaks on either path.
    """

    __slots__ = ("_provider", "_slots", "_timeout_seconds")

    def __init__(
        self,
        provider: GenerationProvider,
        *,
        max_concurrency: int,
        timeout_seconds: float,
    ) -> None:
        if max_concurrency < 1:
            message = f"max_concurrency must be at least 1, got {max_concurrency}"
            raise ValueError(message)
        if timeout_seconds <= 0:
            message = f"timeout_seconds must be positive, got {timeout_seconds}"
            raise ValueError(message)

        self._provider = provider
        self._slots = asyncio.Semaphore(max_concurrency)
        self._timeout_seconds = timeout_seconds

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        try:
            async with asyncio.timeout(self._timeout_seconds), self._slots:
                return await self._provider.generate(request)
        except TimeoutError as error:
            # Includes the provider's own timeout subclass; re-raised as the
            # generation type so the transport layer maps it uniformly.
            if isinstance(error, GenerationTimeoutError):
                raise
            message = "generation did not complete within the deadline"
            raise GenerationTimeoutError(message) from error

    async def aclose(self) -> None:
        close = getattr(self._provider, "aclose", None)
        if close is not None:
            await close()
