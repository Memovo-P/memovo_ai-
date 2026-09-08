"""Error code tables and exception classification."""

import pytest

from memovo_ai.api.errors import classify
from memovo_ai.core.errors import (
    ERROR_HTTP_STATUS,
    ERROR_MESSAGE,
    ERROR_RETRYABLE,
    AiServiceError,
)
from memovo_ai.embeddings.models import (
    EmbeddingError,
    EmbeddingInferenceError,
    InvalidEmbeddingError,
    ModelUnavailableError,
)
from memovo_ai.providers.vector_search import InvalidUserScopeError, UserIsolationError
from memovo_ai.schemas import ErrorCode

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------
# The tables are complete
# --------------------------------------------------------------------------
@pytest.mark.parametrize("table", [ERROR_HTTP_STATUS, ERROR_RETRYABLE, ERROR_MESSAGE])
def test_every_code_is_covered(table: object) -> None:
    assert set(table) == set(ErrorCode)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("code", "status"),
    [
        (ErrorCode.INVALID_INPUT, 422),
        (ErrorCode.INVALID_REQUEST, 400),
        (ErrorCode.MODEL_UNAVAILABLE, 503),
        (ErrorCode.EMBEDDING_FAILED, 503),
        (ErrorCode.VECTOR_SEARCH_FAILED, 503),
        (ErrorCode.TIMEOUT, 504),
        (ErrorCode.INTERNAL_ERROR, 500),
    ],
)
def test_http_status_matches_the_recommended_table(code: ErrorCode, status: int) -> None:
    """Doc 03, Phase 15."""
    assert ERROR_HTTP_STATUS[code] == status


def test_chunking_failed_is_mapped_despite_the_gap_in_the_table() -> None:
    """Doc 03's table omits CHUNKING_FAILED; it is deterministic local work,
    so a failure is a service defect rather than a transient dependency."""
    assert ERROR_HTTP_STATUS[ErrorCode.CHUNKING_FAILED] == 500
    assert ERROR_RETRYABLE[ErrorCode.CHUNKING_FAILED] is False


# --------------------------------------------------------------------------
# Retryability
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "code",
    [
        ErrorCode.MODEL_UNAVAILABLE,
        ErrorCode.EMBEDDING_FAILED,
        ErrorCode.VECTOR_SEARCH_FAILED,
        ErrorCode.TIMEOUT,
    ],
)
def test_transient_failures_are_retryable(code: ErrorCode) -> None:
    assert ERROR_RETRYABLE[code] is True


@pytest.mark.parametrize(
    "code",
    [
        ErrorCode.INVALID_INPUT,
        ErrorCode.INVALID_REQUEST,
        ErrorCode.CHUNKING_FAILED,
        ErrorCode.INTERNAL_ERROR,
    ],
)
def test_deterministic_failures_are_not_retryable(code: ErrorCode) -> None:
    """Repeating an identical request could not plausibly succeed."""
    assert ERROR_RETRYABLE[code] is False


def test_internal_error_is_never_retryable() -> None:
    """It covers a broken user pre-filter; retrying would leak the same rows."""
    assert ERROR_RETRYABLE[ErrorCode.INTERNAL_ERROR] is False


# --------------------------------------------------------------------------
# Messages are safe by construction
# --------------------------------------------------------------------------
@pytest.mark.parametrize("code", list(ErrorCode))
def test_messages_are_non_empty_and_generic(code: ErrorCode) -> None:
    message = ERROR_MESSAGE[code]

    assert message
    assert "Traceback" not in message
    assert "/" not in message
    assert "\\" not in message


def test_the_documented_model_unavailable_message_is_used() -> None:
    """Doc 03's Phase 15 example."""
    assert ERROR_MESSAGE[ErrorCode.MODEL_UNAVAILABLE] == (
        "Embedding model is temporarily unavailable"
    )


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("exception", "expected"),
    [
        (ModelUnavailableError("x"), ErrorCode.MODEL_UNAVAILABLE),
        (EmbeddingInferenceError("x"), ErrorCode.EMBEDDING_FAILED),
        (InvalidEmbeddingError("x"), ErrorCode.EMBEDDING_FAILED),
        (EmbeddingError("x"), ErrorCode.EMBEDDING_FAILED),
        (InvalidUserScopeError("x"), ErrorCode.INVALID_INPUT),
        (UserIsolationError("x"), ErrorCode.INTERNAL_ERROR),
        (TimeoutError(), ErrorCode.TIMEOUT),
        (RuntimeError("x"), ErrorCode.INTERNAL_ERROR),
        (ValueError("x"), ErrorCode.INTERNAL_ERROR),
        (KeyError("x"), ErrorCode.INTERNAL_ERROR),
    ],
)
def test_exceptions_map_to_their_code(exception: Exception, expected: ErrorCode) -> None:
    assert classify(exception) is expected


def test_a_bare_value_error_is_not_treated_as_a_bad_request() -> None:
    """Only InvalidUserScopeError means bad input; a misconfigured provider
    name is a defect and must not be reported as the caller's fault."""
    assert classify(ValueError("unknown vector provider 'qdrant'")) is ErrorCode.INTERNAL_ERROR
    assert classify(InvalidUserScopeError("blank")) is ErrorCode.INVALID_INPUT


def test_invalid_user_scope_is_still_a_value_error() -> None:
    """Phase 10 callers catching ValueError keep working."""
    assert issubclass(InvalidUserScopeError, ValueError)


@pytest.mark.parametrize("code", list(ErrorCode))
def test_an_explicit_service_error_keeps_its_code(code: ErrorCode) -> None:
    assert classify(AiServiceError(code)) is code


# --------------------------------------------------------------------------
# AiServiceError
# --------------------------------------------------------------------------
def test_service_error_defaults_come_from_the_tables() -> None:
    error = AiServiceError(ErrorCode.TIMEOUT)

    assert error.public_message == ERROR_MESSAGE[ErrorCode.TIMEOUT]
    assert error.retryable is True
    assert error.http_status == 504


def test_service_error_accepts_overrides() -> None:
    error = AiServiceError(ErrorCode.TIMEOUT, message="Search timed out", retryable=False)

    assert error.public_message == "Search timed out"
    assert error.retryable is False
