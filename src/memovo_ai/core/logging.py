"""Privacy-safe structured logging.

Doc 06 section 5 splits every field into two lists: things this service must
never log -- Memory title and content, ``WhySaved``, the raw query,
embeddings, retrieved chunk text, revealing tags -- and things it should log,
which are all counts, durations, identifiers and codes.

That distinction is enforced here rather than left to review. :func:`log_event`
accepts only the field names in :data:`SAFE_FIELDS`, and only short scalar
values; anything else raises. A developer cannot log a user's query by
accident, because the field name it would need does not exist. Every call site
passes literal keys, so a violation surfaces in the test suite rather than in
production.

Correlation
-----------
The Backend correlates a job with an AI call through the ``X-Request-Id``
header (contract, section 5). The id is held in a
:class:`~contextvars.ContextVar` so services can be logged against it without
threading a parameter through every signature -- an ``asyncio`` task inherits
the context of whoever created it, so concurrent requests cannot borrow each
other's id.

User identifiers
----------------
``userId`` is hashed, never logged raw (doc 03, Phase 18). A bare hash of a
low-entropy identifier is confirmable by anyone who can guess candidate ids,
so ``MEMOVO_LOG_USER_SALT`` exists for deployments that need the hash to
resist that. Unsalted is the default: it still keeps raw identifiers out of
log aggregation, which is what the requirement asks for.
"""

import hashlib
import json
import logging
import sys
import time
import traceback
import types
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Any

from memovo_ai.core.config import JSON_LOG_FORMAT, LoggingSettings

__all__ = [
    "CORRELATION_ID_HEADER",
    "MAX_VALUE_LENGTH",
    "SAFE_FIELDS",
    "JsonFormatter",
    "KeyValueFormatter",
    "Timer",
    "UnsafeLogFieldError",
    "bind_correlation_id",
    "configure_logging",
    "current_correlation_id",
    "format_safe_exception",
    "hash_user_id",
    "log_event",
    "new_correlation_id",
    "reset_correlation_id",
    "safe_text",
    "timed",
]

#: Correlation header the contract tells the Backend to send (section 5).
CORRELATION_ID_HEADER = "X-Request-Id"

#: The only field names that may be logged. Every one is a count, a duration,
#: an opaque identifier, a code or a boolean -- doc 06 section 5's "prefer"
#: list, extended only where a phase needed a measurement that list allows.
SAFE_FIELDS = frozenset(
    {
        # request context
        "correlation_id",
        "method",
        "route",
        "status",
        "duration_ms",
        # ingestion
        "memory_id",
        "chunk_count",
        "content_chars",
        "chunking_ms",
        "embedding_ms",
        # retrieval
        "user",
        "top_k",
        "threshold",
        "query_chars",
        "vector_search_ms",
        "hit_count",
        "result_count",
        "matched",
        # failures and lifecycle
        "error_code",
        "error_type",
        "services_available",
    }
)

#: Upper bound on a logged string. Long enough for a hash, a route or an
#: identifier; far too short to smuggle Memory content through.
MAX_VALUE_LENGTH = 96

#: Substituted for a value that is empty or unavailable, so a field is never
#: silently dropped.
UNSET = "unset"

_ALLOWED_VALUE_TYPES = (str, int, float, bool, type(None))

#: What ``logging`` hands a formatter for ``exc_info``. The all-``None`` arm
#: is what a record carries when no exception was active.
_ExcInfo = (
    tuple[type[BaseException], BaseException, types.TracebackType | None] | tuple[None, None, None]
)

#: Marks the handler this module installed, so reconfiguring replaces it
#: instead of stacking a second copy on every ``create_app()``.
_HANDLER_FLAG = "_memovo_handler"

_EVENT_KEY = "memovo_event"
_FIELDS_KEY = "memovo_fields"

_correlation_id: ContextVar[str] = ContextVar("memovo_correlation_id", default="")

#: Fields ``logging`` puts on every record. Anything outside this set was
#: attached by a caller, which is how the formatters surface extras from
#: third-party loggers without those callers having to cooperate.
_RESERVED_RECORD_FIELDS = frozenset(logging.LogRecord("", 0, "", 0, "", None, None).__dict__)


class UnsafeLogFieldError(ValueError):
    """Raised when a log call would emit something outside the safe list.

    A programming error rather than a runtime condition: every call site
    passes literal field names, so this fires in tests, not in production.
    """


def new_correlation_id() -> str:
    """Generate an id for a request that arrived without one."""
    return uuid.uuid4().hex


def safe_text(value: str | None, *, fallback: str = UNSET) -> str:
    """Reduce untrusted text to something safe to put in a log line.

    Control characters are removed -- a newline in an inbound header would
    otherwise let a caller forge whole log records -- and the result is
    truncated. Used on the ``X-Request-Id`` header and on request paths, both
    of which are caller-controlled.
    """
    if not value:
        return fallback

    cleaned = "".join(character for character in value if character.isprintable()).strip()

    return cleaned[:MAX_VALUE_LENGTH] or fallback


def bind_correlation_id(value: str | None = None) -> Token[str]:
    """Set the correlation id for the current context.

    Returns the token to hand back to :func:`reset_correlation_id`, so a
    request cannot leak its id into whatever runs next on the same task.
    """
    return _correlation_id.set(safe_text(value, fallback="") or new_correlation_id())


def reset_correlation_id(token: Token[str]) -> None:
    """Restore whatever correlation id was in place before binding."""
    _correlation_id.reset(token)


def current_correlation_id() -> str:
    """The id of the request in flight, or ``""`` outside a request."""
    return _correlation_id.get()


def hash_user_id(user_id: str, *, salt: str = "") -> str:
    """Render a trusted ``userId`` as a stable, non-reversible log token."""
    if not user_id:
        return f"u_{UNSET}"

    digest = hashlib.sha256(f"{salt}{user_id}".encode())

    return f"u_{digest.hexdigest()[:16]}"


@dataclass(slots=True)
class Timer:
    """Elapsed milliseconds, filled in when :func:`timed` exits."""

    ms: float = 0.0


@contextmanager
def timed() -> Iterator[Timer]:
    """Measure a block, including one that awaits.

    Uses a monotonic clock, so a wall-clock adjustment cannot produce a
    negative duration.
    """
    timer = Timer()
    start = time.perf_counter()
    try:
        yield timer
    finally:
        timer.ms = round((time.perf_counter() - start) * 1000, 3)


def _validate(name: str, value: object) -> None:
    if name not in SAFE_FIELDS:
        message = (
            f"{name!r} is not a loggable field; doc 06 section 5 allows only "
            f"operational metadata, and user content must never be logged"
        )
        raise UnsafeLogFieldError(message)

    if not isinstance(value, _ALLOWED_VALUE_TYPES):
        message = f"log field {name!r} must be a scalar, got {type(value).__name__}"
        raise UnsafeLogFieldError(message)

    if not isinstance(value, str):
        return

    if len(value) > MAX_VALUE_LENGTH:
        message = (
            f"log field {name!r} is {len(value)} characters, over the "
            f"{MAX_VALUE_LENGTH} limit; long strings are how content leaks"
        )
        raise UnsafeLogFieldError(message)

    if any(not character.isprintable() for character in value):
        message = f"log field {name!r} contains control characters"
        raise UnsafeLogFieldError(message)


def log_event(
    logger: logging.Logger,
    event: str,
    *,
    level: int = logging.INFO,
    exc_info: bool = False,
    **fields: object,
) -> None:
    """Emit one structured operational record.

    Args:
        logger: Logger of the emitting module.
        event: Stable machine-readable name, e.g. ``"http.request"``.
        level: Standard ``logging`` level.
        exc_info: Attach a traceback. Only for genuine defects -- a traceback
            can contain locals holding Memory content, so it belongs in an
            operator's log and never in a response.
        **fields: Must all be named in :data:`SAFE_FIELDS`.

    Raises:
        UnsafeLogFieldError: on an unknown field name or an unsafe value.
    """
    for name, value in fields.items():
        _validate(name, value)

    payload: dict[str, object] = {"correlation_id": current_correlation_id() or UNSET}
    payload.update(fields)

    logger.log(level, event, exc_info=exc_info, extra={_EVENT_KEY: event, _FIELDS_KEY: payload})


def format_safe_exception(exc_info: _ExcInfo) -> str:
    """Render a traceback with every exception *message* removed.

    A message is the one part of a traceback that can carry user content:
    ``RuntimeError(f"connect failed for {query}")`` is an ordinary thing for a
    library to raise, and doc 06 section 5 forbids the raw query reaching a
    log. Chained causes are dropped for the same reason.

    What remains is the exception type and the frames -- file, line, function
    and the *source* line, which is static code and therefore safe. That is
    where the defect is; the message is rarely what locates it, and
    ``error_type`` is logged as a field alongside.

    Applied by both formatters, so a traceback logged by Starlette or by any
    other library is sanitized too, not only the ones this service writes.
    """
    _, exception, traceback_object = exc_info

    frames = "".join(traceback.format_tb(traceback_object)).rstrip()
    name = type(exception).__name__ if exception is not None else "UnknownError"

    return f"{frames}\n{name}: [message omitted: may contain user content]".lstrip()


def _extras(record: logging.LogRecord) -> dict[str, Any]:
    fields = record.__dict__.get(_FIELDS_KEY)
    if isinstance(fields, dict):
        return dict(fields)

    return {
        key: value
        for key, value in record.__dict__.items()
        if key not in _RESERVED_RECORD_FIELDS and key not in {_EVENT_KEY, _FIELDS_KEY}
    }


class JsonFormatter(logging.Formatter):
    """One JSON object per line, for log aggregation."""

    def formatException(self, ei: _ExcInfo) -> str:  # noqa: N802 - logging's own name
        return format_safe_exception(ei)

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "event": record.__dict__.get(_EVENT_KEY, record.getMessage()),
        }
        payload.update(_extras(record))

        if record.exc_info:
            payload["traceback"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False, default=str)


class KeyValueFormatter(logging.Formatter):
    """Human-readable equivalent, for local development."""

    def formatException(self, ei: _ExcInfo) -> str:  # noqa: N802 - logging's own name
        return format_safe_exception(ei)

    def format(self, record: logging.LogRecord) -> str:
        event = record.__dict__.get(_EVENT_KEY, record.getMessage())
        pairs = " ".join(f"{key}={value}" for key, value in _extras(record).items())
        line = f"{record.levelname:<7} {event} {pairs}".rstrip()

        if record.exc_info:
            line = f"{line}\n{self.formatException(record.exc_info)}"

        return line


def configure_logging(settings: LoggingSettings | None = None) -> None:
    """Install the structured handler on the root logger.

    Idempotent: the handler is tagged, so repeated calls -- every
    ``create_app()`` in the test suite is one -- replace it rather than
    stacking duplicates. Only this module's handler is touched; one installed
    by pytest or by a host process is left alone.
    """
    resolved = settings if settings is not None else LoggingSettings()

    root = logging.getLogger()
    for existing in [item for item in root.handlers if getattr(item, _HANDLER_FLAG, False)]:
        root.removeHandler(existing)

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        JsonFormatter() if resolved.format == JSON_LOG_FORMAT else KeyValueFormatter()
    )
    setattr(handler, _HANDLER_FLAG, True)

    root.addHandler(handler)
    root.setLevel(resolved.level_number)
