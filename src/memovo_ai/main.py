"""FastAPI application.

Process startup and service readiness are two different events here.

The HTTP process starts first. The lifespan validates configuration, starts
one background task that builds the services -- which loads the embedding
model once, and on a cold cache downloads it -- and yields immediately, so
Uvicorn binds its socket while the model is still loading. ``/health``
answers from that moment; ``/ready`` reports ``loading`` until the build
finishes, then ``ready``. An orchestrator's liveness probe must therefore
target ``/health``, never ``/ready``.

A failed build does not crash the process: it leaves the services unset and
the instance unready, and requests fail with a clear unavailable state rather
than the application refusing to start (doc 06, section 9). That also lets
the API be exercised without model weights present.

An invalid *configuration* is the exception to that tolerance. A missing model
or an unreachable cluster may recover on its own; a wrong setting will not,
and a service that keeps answering with one is more dangerous than one that
refuses to start. Configuration is validated before the build is scheduled,
so that refusal still happens in the lifespan and startup fails.

This service is internal. It must not be exposed publicly (doc 06, section 7);
the Backend is the only caller.
"""

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI

from memovo_ai.api.dependencies import Services, build_services, validate_configuration
from memovo_ai.api.errors import register_exception_handlers
from memovo_ai.api.middleware import RequestContextMiddleware
from memovo_ai.api.router import api_router
from memovo_ai.core.config import (
    GenerationSettings,
    InvalidConfigurationError,
    ProviderSettings,
    RuntimeSettings,
)
from memovo_ai.core.logging import configure_logging, log_event, safe_text
from memovo_ai.core.readiness import ReadinessState, set_readiness
from memovo_ai.embeddings.models import EmbeddingError
from memovo_ai.providers.vector_search.base import VectorSearchUnavailableError

__all__ = ["app", "create_app", "wait_for_initialization"]

_logger = logging.getLogger(__name__)


#: Set on the app when it must not build services at startup. Tests supply
#: their own through ``dependency_overrides``; loading a real model just to
#: replace it is slow and makes the suite depend on whether the optional
#: ``embeddings`` extra happens to be installed.
_SKIP_STARTUP_SERVICES = "memovo_skip_startup_services"

#: Where the lifecycle object below lives on ``app.state``. Private: it is
#: how shutdown and the test suite wait for the build, never an HTTP surface.
_INITIALIZATION_ATTRIBUTE = "memovo_initialization"

#: Build failures the process outlives. The dependencies raise
#: ``ModelUnavailableError`` per request while the services are unset, which
#: Phase 15 maps to a standardized 503 rather than an opaque crash loop.
#:
#: ``VectorSearchUnavailableError`` belongs here too: an unset or wrong Atlas
#: connection string is a configuration problem, and a crash loop tells an
#: operator far less than a service that starts and reports itself
#: unavailable (doc 06, section 9).
_TOLERATED_BUILD_FAILURES = (EmbeddingError, VectorSearchUnavailableError, ValueError)


def _install(app: FastAPI, services: Services | None) -> None:
    """Publish the whole service graph to ``app.state``, or clear it.

    All five references change together, with no await in between, so a
    request handled on this same event loop can never see some services set
    and others not. Readiness is switched by the caller only after this
    returns, so ``ready`` is never reported ahead of the services it certifies.
    """
    app.state.memory_search_service = services.memory_search if services else None
    app.state.memory_processor_service = services.memory_processor if services else None
    app.state.generation_provider = services.generation if services else None
    app.state.memory_chat_service = services.memory_chat if services else None
    app.state.note_preparation_service = services.note_preparation if services else None


class _ServiceInitialization:
    """Owns the one background build and the handoff of its result.

    Exactly one build per application instance, run off the event loop
    because it is synchronous and does CPU, filesystem and network work.
    Everything that touches ``app.state`` happens on the event loop, so the
    build's completion and the lifespan's shutdown are serialized by the loop
    itself and cannot interleave.

    Ownership of the built :class:`Services` moves exactly once: from the
    build to ``_services`` when it is installed, or straight to ``aclose``
    when it arrives after shutdown began. Shutdown takes ``_services`` out
    before closing it, so nothing is closed twice.
    """

    __slots__ = (
        "_app",
        "_build",
        "_closing",
        "_done",
        "_generation",
        "_services",
        "_vector_provider",
        "task",
    )

    def __init__(
        self,
        app: FastAPI,
        *,
        build: Callable[[], Services],
        vector_provider: str,
        generation: GenerationSettings,
    ) -> None:
        self._app = app
        self._build = build
        self._vector_provider = vector_provider
        self._generation = generation
        self._services: Services | None = None
        self._closing = False
        self._done = asyncio.Event()
        #: The background task, or ``None`` when no build was requested.
        self.task: asyncio.Task[None] | None = None

    def start(self) -> None:
        """Schedule the build. Returns at once; the lifespan yields next."""
        log_event(_logger, "service.initialization_started", vector_provider=self._vector_provider)
        self.task = asyncio.create_task(self._run(), name="memovo-service-initialization")

    def skip(self) -> None:
        """Complete immediately with no services (``load_services=False``)."""
        self._complete(None)
        self._done.set()

    async def wait(self) -> None:
        """Return once the build has reached a terminal state.

        An event rather than the task itself: a waiter that gets cancelled
        must not cancel the build along with it.
        """
        await self._done.wait()

    async def stop(self) -> None:
        """Withdraw readiness, wait out a running build, release the services.

        Readiness goes first, so a draining instance stops attracting traffic
        before anything is torn down.

        A build still in progress is waited for, not cancelled. Cancelling
        the task would only detach it from its worker thread: the thread keeps
        running, its result -- possibly a live Atlas client -- would be
        dropped unclosed, and the interpreter joins the executor thread at
        exit anyway, so nothing would finish sooner. The orchestrator's stop
        deadline bounds the wait. With ``_closing`` set, a late result is
        released by :meth:`_finish` instead of installed.
        """
        self._closing = True
        set_readiness(self._app.state, ReadinessState.UNAVAILABLE)
        await self._done.wait()

        services, self._services = self._services, None
        _install(self._app, None)
        if services is not None:
            await services.aclose()

    async def _run(self) -> None:
        services: Services | None = None
        try:
            services = await asyncio.to_thread(self._build)
        except InvalidConfigurationError as error:
            # Validation already ran before this task was scheduled, so this
            # is a setting that changed underneath the build or a caller that
            # bypassed the guard. Too late to fail startup -- the process is
            # already serving -- but never silent: CRITICAL, and unready.
            log_event(
                _logger,
                "service.startup_rejected",
                level=logging.CRITICAL,
                error_type=type(error).__name__,
                vector_provider=self._vector_provider,
            )
        except _TOLERATED_BUILD_FAILURES as error:
            # The exception type, never its message: a driver error can carry
            # a connection string (doc 06, section 6). The full traceback goes
            # to the operator's log because this one is worth diagnosing.
            log_event(
                _logger,
                "service.startup_failed",
                level=logging.ERROR,
                error_type=type(error).__name__,
                exc_info=True,
            )
        except Exception as error:
            # A defect, not a tolerated failure. It cannot fail startup any
            # more -- the HTTP process is up -- and it must not become an
            # unretrieved task exception that surfaces only at garbage
            # collection, so it is recorded here at the highest level and
            # leaves the instance unready.
            log_event(
                _logger,
                "service.startup_failed",
                level=logging.CRITICAL,
                error_type=type(error).__name__,
                exc_info=True,
            )
        await self._finish(services)

    async def _finish(self, services: Services | None) -> None:
        try:
            if self._closing:
                # Shutdown began while the build ran. Nobody will serve
                # this; release it here, the one place that still holds it.
                if services is not None:
                    await services.aclose()
                return
            self._complete(services)
        finally:
            self._done.set()

    def _complete(self, services: Services | None) -> None:
        """Install the finished graph and report the outcome. Loop-only."""
        self._services = services
        _install(self._app, services)

        # Readiness follows the services, not the process. Without usable
        # services the process stays alive and answers probes, but must be
        # kept out of the load balancer (doc 06, section 9).
        set_readiness(
            self._app.state,
            ReadinessState.READY if services is not None else ReadinessState.UNAVAILABLE,
        )

        generation = self._generation
        log_event(
            _logger,
            "service.started",
            services_available=services is not None,
            vector_provider=self._vector_provider,
            generation_enabled=generation.enabled,
            generation_provider=safe_text(generation.provider)
            if generation.enabled
            else "disabled",
            generation_available=services is not None and services.generation is not None,
        )


async def wait_for_initialization(app: FastAPI) -> None:
    """Wait until the background build has reached a terminal state.

    Terminal means the services were installed and ``/ready`` is ``ready``,
    or the build failed and ``/ready`` is ``unavailable``. Returns at once
    for an application that was told not to build services, and for one
    whose startup has not run: there is nothing to wait for either way.

    Shutdown relies on the same wait; the test suite uses it to observe the
    outcome of startup without timing-based sleeps. It is not an HTTP
    surface.
    """
    initialization: _ServiceInitialization | None = getattr(
        app.state, _INITIALIZATION_ATTRIBUTE, None
    )
    if initialization is not None:
        await initialization.wait()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Validate cheaply, schedule the one build, yield at once.

    Importing the composition root here is safe: ``qwen3`` defers the model
    runtime import to ``load()``, so nothing heavy is pulled in until a model
    is actually requested -- and that now happens on the build's own thread,
    after this function has yielded.
    """
    skip = getattr(app.state, _SKIP_STARTUP_SERVICES, False)
    set_readiness(app.state, ReadinessState.LOADING)
    _install(app, None)

    # The configured name, read here rather than taken from the built
    # services: an operator needs it in the log even when startup failed, and
    # it is what tells them whether a deployment is pointed at a real index.
    provider = ProviderSettings()
    vector_provider = safe_text(provider.vector_provider)

    if not skip:
        try:
            validate_configuration(provider, RuntimeSettings())
        except InvalidConfigurationError as error:
            # Not tolerated. A missing model or an unreachable cluster can fix
            # itself, so those start unready and say so; a wrong setting
            # cannot, and a process that keeps serving with one is worse than
            # one that refuses to start. Re-raised before any build is
            # scheduled, so startup fails, the model is never loaded, and
            # /ready is never reachable, let alone 200.
            set_readiness(app.state, ReadinessState.UNAVAILABLE)
            log_event(
                _logger,
                "service.startup_rejected",
                level=logging.CRITICAL,
                error_type=type(error).__name__,
                vector_provider=vector_provider,
            )
            raise

    initialization = _ServiceInitialization(
        app,
        build=build_services,
        vector_provider=vector_provider,
        generation=GenerationSettings(),
    )
    setattr(app.state, _INITIALIZATION_ATTRIBUTE, initialization)
    if skip:
        initialization.skip()
    else:
        initialization.start()

    yield

    await initialization.stop()
    log_event(_logger, "service.stopped")


def create_app(*, load_services: bool = True) -> FastAPI:
    """Build the application.

    Args:
        load_services: Build the services in the background after startup.
            Tests pass ``False`` and supply their own through
            ``dependency_overrides``, which keeps the suite fast and, more
            importantly, makes it behave identically whether or not the
            optional ``embeddings`` extra is installed. Production always
            uses the default.
    """
    configure_logging()

    application = FastAPI(
        title="Memovo AI Service",
        version="0.1.0",
        summary="Memory processing and retrieval for Memovo.",
        lifespan=lifespan,
    )
    setattr(application.state, _SKIP_STARTUP_SERVICES, not load_services)
    set_readiness(application.state, ReadinessState.STARTING)
    application.add_middleware(RequestContextMiddleware)
    register_exception_handlers(application)
    application.include_router(api_router)

    return application


app = create_app()
