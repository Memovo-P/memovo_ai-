"""Service readiness state (doc 06, section 9).

The lifecycle an operator needs to distinguish::

    starting -> loading -> ready
                       \\-> unavailable

``unavailable`` is the state a failed model load leaves behind. The service
keeps running and keeps answering probes, because a crash loop tells an
operator far less than a process that starts and reports itself unready --
the same reasoning the startup path in ``main`` already follows.

The state is stored on the application's own state object. This module takes
that object as a plain ``object`` and reaches it through ``getattr`` /
``setattr``, so ``core`` stays free of a framework import while the API layer
and the startup path share one definition.
"""

from enum import StrEnum

__all__ = ["READINESS_ATTRIBUTE", "ReadinessState", "readiness_of", "set_readiness"]

#: Attribute name used on the application state object.
READINESS_ATTRIBUTE = "memovo_readiness_state"


class ReadinessState(StrEnum):
    """Where the service is in its startup lifecycle."""

    #: The application object exists; startup has not run yet.
    STARTING = "starting"

    #: The HTTP process is up and the background build is loading the
    #: embedding model. Liveness already answers; readiness does not yet.
    LOADING = "loading"

    #: Every service is built. The AI endpoints can serve traffic.
    READY = "ready"

    #: Startup finished without usable services, or shutdown has begun.
    #: The process is alive and must not be restarted blindly; its AI
    #: endpoints answer with the standardized unavailable error.
    UNAVAILABLE = "unavailable"


def set_readiness(state: object, value: ReadinessState) -> None:
    """Record the current lifecycle state on an application state object."""
    setattr(state, READINESS_ATTRIBUTE, value)


def readiness_of(state: object) -> ReadinessState:
    """Read the lifecycle state, defaulting to ``STARTING``.

    The default matters: an application whose startup has not run yet must
    not be reported as ready merely because nothing set the attribute.
    """
    value = getattr(state, READINESS_ATTRIBUTE, None)

    return value if isinstance(value, ReadinessState) else ReadinessState.STARTING
