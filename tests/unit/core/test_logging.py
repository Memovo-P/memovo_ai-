"""Privacy-safe structured logging (doc 03, Phase 18; doc 06, section 5).

The point of these tests is the *negative* guarantee: that the logging layer
structurally refuses to emit user content, rather than merely happening not
to today.
"""

import json
import logging

import pytest

from memovo_ai.core.config import LoggingSettings
from memovo_ai.core.logging import (
    CORRELATION_ID_HEADER,
    MAX_VALUE_LENGTH,
    SAFE_FIELDS,
    JsonFormatter,
    KeyValueFormatter,
    UnsafeLogFieldError,
    bind_correlation_id,
    configure_logging,
    current_correlation_id,
    format_safe_exception,
    hash_user_id,
    log_event,
    new_correlation_id,
    reset_correlation_id,
    safe_text,
    timed,
)

pytestmark = pytest.mark.unit

logger = logging.getLogger("memovo_ai.tests.logging")


def record_of(caplog: pytest.LogCaptureFixture) -> logging.LogRecord:
    assert len(caplog.records) == 1
    return caplog.records[0]


# ---------------------------------------------------------------------------
# The safe-field allowlist
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field",
    ["content", "query", "title", "description", "why_saved", "tags", "embedding", "chunk_text"],
)
def test_the_forbidden_fields_cannot_be_logged(field: str) -> None:
    """Doc 06 section 5's "do not log" list, by name.

    Each of these is a plausible field name a developer might reach for. None
    of them exists, so reaching for one fails loudly.
    """
    with pytest.raises(UnsafeLogFieldError):
        log_event(logger, "test.event", **{field: "user data"})


def test_an_unknown_field_is_refused_rather_than_dropped() -> None:
    with pytest.raises(UnsafeLogFieldError, match="not a loggable field"):
        log_event(logger, "test.event", something_new=1)


def test_the_allowlist_holds_only_operational_metadata() -> None:
    """A guard against someone widening the list to admit content.

    Anything content-shaped -- singular nouns for the user's own text --
    must never appear here.
    """
    forbidden = {"content", "query", "title", "description", "why_saved", "tags", "embedding"}

    assert SAFE_FIELDS & forbidden == set()


def test_a_long_string_value_is_refused() -> None:
    """The other way content could arrive: through an allowed field name."""
    with pytest.raises(UnsafeLogFieldError, match="over the"):
        log_event(logger, "test.event", memory_id="x" * (MAX_VALUE_LENGTH + 1))


def test_a_value_at_the_limit_is_accepted(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG):
        log_event(logger, "test.event", memory_id="x" * MAX_VALUE_LENGTH)

    assert len(caplog.records) == 1


def test_a_control_character_in_a_value_is_refused() -> None:
    """A newline would let a caller forge a second log record."""
    with pytest.raises(UnsafeLogFieldError, match="control characters"):
        log_event(logger, "test.event", memory_id="memory_1\ninjected event")


def test_a_non_scalar_value_is_refused() -> None:
    """Blocks the obvious way to smuggle a list of chunks through."""
    with pytest.raises(UnsafeLogFieldError, match="must be a scalar"):
        log_event(logger, "test.event", chunk_count=[1, 2, 3])


def test_nothing_is_emitted_when_a_field_is_refused(caplog: pytest.LogCaptureFixture) -> None:
    """Validation happens before emission, so no partial record escapes."""
    with caplog.at_level(logging.DEBUG), pytest.raises(UnsafeLogFieldError):
        log_event(logger, "test.event", chunk_count=1, query="what did I save")

    assert caplog.records == []


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------


def test_every_record_carries_a_correlation_id(caplog: pytest.LogCaptureFixture) -> None:
    token = bind_correlation_id("job-abc-1")
    try:
        with caplog.at_level(logging.DEBUG):
            log_event(logger, "test.event", chunk_count=2)
    finally:
        reset_correlation_id(token)

    assert record_of(caplog).memovo_fields["correlation_id"] == "job-abc-1"


def test_an_absent_correlation_id_is_generated() -> None:
    token = bind_correlation_id(None)
    try:
        assert len(current_correlation_id()) == len(new_correlation_id())
    finally:
        reset_correlation_id(token)


def test_a_blank_correlation_id_is_generated() -> None:
    """An empty header must not produce records that correlate to nothing."""
    token = bind_correlation_id("   ")
    try:
        assert current_correlation_id()
    finally:
        reset_correlation_id(token)


def test_a_hostile_correlation_id_is_sanitized() -> None:
    token = bind_correlation_id("job-1\nlevel=INFO event=forged")
    try:
        assert "\n" not in current_correlation_id()
    finally:
        reset_correlation_id(token)


def test_an_over_long_correlation_id_is_truncated() -> None:
    token = bind_correlation_id("j" * 500)
    try:
        assert len(current_correlation_id()) == MAX_VALUE_LENGTH
    finally:
        reset_correlation_id(token)


def test_resetting_restores_the_previous_id() -> None:
    outer = bind_correlation_id("outer")
    inner = bind_correlation_id("inner")
    reset_correlation_id(inner)

    try:
        assert current_correlation_id() == "outer"
    finally:
        reset_correlation_id(outer)


def test_the_correlation_header_is_the_one_the_contract_names() -> None:
    """The contract tells the Backend to send ``X-Request-Id`` (section 5)."""
    assert CORRELATION_ID_HEADER == "X-Request-Id"


# ---------------------------------------------------------------------------
# User identifiers
# ---------------------------------------------------------------------------


def test_a_user_id_is_never_logged_in_the_clear() -> None:
    assert "user_123" not in hash_user_id("user_123")


def test_hashing_a_user_id_is_stable() -> None:
    """Two requests from one user must be joinable in a log aggregator."""
    assert hash_user_id("user_123") == hash_user_id("user_123")


def test_different_users_hash_differently() -> None:
    assert hash_user_id("user_123") != hash_user_id("user_456")


def test_the_salt_changes_the_hash() -> None:
    """Without this, a guessed userId can be confirmed against a log."""
    assert hash_user_id("user_123", salt="pepper") != hash_user_id("user_123")


def test_a_missing_user_id_is_marked_rather_than_hashed() -> None:
    assert hash_user_id("") == "u_unset"


def test_a_hashed_user_id_fits_the_value_limit(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG):
        log_event(logger, "test.event", user=hash_user_id("user_123"))

    assert len(caplog.records) == 1


# ---------------------------------------------------------------------------
# safe_text
# ---------------------------------------------------------------------------


def test_safe_text_strips_control_characters() -> None:
    assert safe_text("a\r\nb") == "ab"


def test_safe_text_truncates() -> None:
    assert len(safe_text("x" * 1000)) == MAX_VALUE_LENGTH


def test_safe_text_falls_back_when_empty() -> None:
    assert safe_text("") == "unset"


def test_safe_text_falls_back_when_only_control_characters() -> None:
    assert safe_text("\n\t") == "unset"


def test_safe_text_keeps_non_latin_text() -> None:
    """Arabic identifiers must survive; ``isprintable`` is not ASCII-only."""
    assert safe_text("ذاكرة_1") == "ذاكرة_1"


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


def test_timed_reports_a_non_negative_duration() -> None:
    with timed() as elapsed:
        pass

    assert elapsed.ms >= 0.0


def test_timed_records_the_duration_even_when_the_block_raises() -> None:
    """The middleware logs the duration of a failed request from this."""
    with pytest.raises(RuntimeError), timed() as elapsed:
        raise RuntimeError

    assert elapsed.ms >= 0.0


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------


def test_json_output_is_one_parseable_object(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG):
        log_event(logger, "memory.processed", chunk_count=3, memory_id="memory_1")

    payload = json.loads(JsonFormatter().format(record_of(caplog)))

    assert payload["event"] == "memory.processed"
    assert payload["chunk_count"] == 3
    assert payload["level"] == "INFO"


def test_json_output_survives_non_latin_content(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG):
        log_event(logger, "memory.processed", memory_id="ذاكرة_1")

    assert json.loads(JsonFormatter().format(record_of(caplog)))["memory_id"] == "ذاكرة_1"


def test_a_third_party_log_line_still_formats(caplog: pytest.LogCaptureFixture) -> None:
    """The formatter is on the root logger, so it sees other libraries too."""
    with caplog.at_level(logging.DEBUG):
        logger.info("a plain message")

    assert json.loads(JsonFormatter().format(record_of(caplog)))["event"] == "a plain message"


def test_key_value_output_is_readable(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG):
        log_event(logger, "memory.processed", chunk_count=3)

    line = KeyValueFormatter().format(record_of(caplog))

    assert "memory.processed" in line
    assert "chunk_count=3" in line


def test_a_traceback_is_rendered_when_requested(caplog: pytest.LogCaptureFixture) -> None:
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        with caplog.at_level(logging.DEBUG):
            log_event(logger, "request.failed", level=logging.ERROR, exc_info=True)

    payload = json.loads(JsonFormatter().format(record_of(caplog)))

    assert "RuntimeError" in payload["traceback"]


# ---------------------------------------------------------------------------
# Traceback sanitizing
# ---------------------------------------------------------------------------


#: Never written as a literal inside a ``raise``: a frame keeps its source
#: line, and these tests are about the *message*, not the code.
PRIVATE_MESSAGE = "ZZQ-" + "the-users-query"


def exc_info_from(exception: BaseException) -> tuple[type[BaseException], BaseException, object]:
    return (type(exception), exception, exception.__traceback__)


def raised(message: str) -> BaseException:
    try:
        raise RuntimeError(message)
    except RuntimeError as error:
        return error


def test_an_exception_message_is_never_rendered() -> None:
    """``RuntimeError(f"failed for {query}")`` is an ordinary thing to raise."""
    rendered = format_safe_exception(exc_info_from(raised(PRIVATE_MESSAGE)))  # type: ignore[arg-type]

    assert PRIVATE_MESSAGE not in rendered


def test_the_exception_type_survives() -> None:
    """Dropping the message must not cost the operator the type."""
    rendered = format_safe_exception(exc_info_from(raised("boom")))  # type: ignore[arg-type]

    assert "RuntimeError" in rendered


def test_the_frames_survive() -> None:
    """Where it failed is the diagnostic value; the message rarely is."""
    rendered = format_safe_exception(exc_info_from(raised("boom")))  # type: ignore[arg-type]

    assert "test_logging.py" in rendered


def test_a_chained_cause_message_is_also_dropped() -> None:
    """``raise X from Y`` must not smuggle Y's message through."""
    try:
        try:
            raise ValueError("ZZQ-inner-content")
        except ValueError as inner:
            raise RuntimeError("outer") from inner
    except RuntimeError as outer:
        rendered = format_safe_exception(exc_info_from(outer))  # type: ignore[arg-type]

    assert "ZZQ-inner-content" not in rendered


def test_the_key_value_formatter_sanitizes_too(caplog: pytest.LogCaptureFixture) -> None:
    """Local development must not be the weak spot.

    The message is built through a variable so it appears only in the
    exception, not in this file's source -- frames keep their source lines,
    which is exactly the distinction being checked.
    """
    try:
        raise RuntimeError(PRIVATE_MESSAGE)
    except RuntimeError:
        with caplog.at_level(logging.DEBUG):
            log_event(logger, "request.failed", level=logging.ERROR, exc_info=True)

    assert PRIVATE_MESSAGE not in KeyValueFormatter().format(record_of(caplog))


# ---------------------------------------------------------------------------
# configure_logging
# ---------------------------------------------------------------------------


def json_settings() -> LoggingSettings:
    """Defaults pinned against the ambient environment.

    ``_env_file=None`` only disables the dotenv file; exported variables are
    still read. A shell with ``MEMOVO_LOG_FORMAT=text`` set must not change
    what these tests assert.
    """
    return LoggingSettings(_env_file=None, format="json", level="INFO")  # type: ignore[call-arg]


def installed_handlers() -> list[logging.Handler]:
    return [
        handler
        for handler in logging.getLogger().handlers
        if isinstance(handler.formatter, JsonFormatter | KeyValueFormatter)
    ]


def test_configuring_twice_does_not_duplicate_handlers() -> None:
    """``create_app()`` runs per test; handlers must not accumulate."""
    configure_logging(json_settings())
    configure_logging(json_settings())
    configure_logging(json_settings())

    assert len(installed_handlers()) == 1


def test_a_handler_installed_by_the_host_is_left_alone() -> None:
    """pytest's own capture handler must survive reconfiguration."""
    root = logging.getLogger()
    foreign = logging.NullHandler()
    root.addHandler(foreign)

    try:
        configure_logging(json_settings())

        assert foreign in root.handlers
    finally:
        root.removeHandler(foreign)


def test_configuring_installs_the_json_formatter_by_default() -> None:
    configure_logging(json_settings())
    installed = [
        handler
        for handler in logging.getLogger().handlers
        if isinstance(handler.formatter, JsonFormatter)
    ]

    assert installed


def test_the_text_format_is_selectable() -> None:
    configure_logging(LoggingSettings(_env_file=None, format="text"))  # type: ignore[call-arg]
    installed = [
        handler
        for handler in logging.getLogger().handlers
        if isinstance(handler.formatter, KeyValueFormatter)
    ]

    try:
        assert installed
    finally:
        configure_logging(json_settings())


def test_an_unknown_level_falls_back_to_info() -> None:
    """Losing log detail is recoverable; refusing to boot over a typo is not."""
    settings = LoggingSettings(_env_file=None, level="LOUD")  # type: ignore[call-arg]

    assert settings.level_number == logging.INFO


def test_a_known_level_is_honoured() -> None:
    settings = LoggingSettings(_env_file=None, level="warning")  # type: ignore[call-arg]

    assert settings.level_number == logging.WARNING


def test_the_user_salt_is_a_secret() -> None:
    """It must not surface in a repr, a traceback or a settings dump."""
    settings = LoggingSettings(_env_file=None, user_salt="pepper")  # type: ignore[call-arg]

    assert "pepper" not in repr(settings)
    assert "pepper" not in str(settings.model_dump())


def test_there_is_no_setting_that_enables_content_logging() -> None:
    """Doc 06 section 5 is absolute; a switch would make it configurable."""
    suspicious = {"log_content", "log_query", "verbose_bodies", "debug_payloads"}

    assert suspicious & set(LoggingSettings.model_fields) == set()
