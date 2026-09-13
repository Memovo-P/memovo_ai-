"""Error code tables and exception classification.

The tables follow contract v1.9 section 21 and its approved addendum
(section 21.9): the Backend classifies by HTTP status + code, and
``retryable`` is consistent metadata rather than an override.
"""

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
from memovo_ai.generation import (
    GenerationDisabledError,
    GenerationRateLimitedError,
    GenerationTimeoutError,
    GenerationUnavailableError,
    InvalidGenerationOutputError,
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
        (ErrorCode.CHUNKING_FAILED, 500),
        (ErrorCode.GENERATION_UNAVAILABLE, 503),
        (ErrorCode.RATE_LIMITED, 429),
        (ErrorCode.AI_INVALID_RESPONSE, 502),
    ],
)
def test_http_status_per_code(code: ErrorCode, status: int) -> None:
    assert ERROR_HTTP_STATUS[code] == status


def test_502_is_reserved_for_invalid_model_output() -> None:
    """The one 5xx the Backend must never retry has to be unambiguous."""
    at_502 = [code for code, status in ERROR_HTTP_STATUS.items() if status == 502]

    assert at_502 == [ErrorCode.AI_INVALID_RESPONSE]


# --------------------------------------------------------------------------
# Retryability is consistent with the canonical status classification
# --------------------------------------------------------------------------
def test_every_5xx_is_retryable_except_invalid_response() -> None:
    """Contract 21.2 lists 500, 502, 503, 504 and other 5xx as retryable;
    21.9 carves out exactly one exception."""
    for code, status in ERROR_HTTP_STATUS.items():
        if status >= 500:
            assert ERROR_RETRYABLE[code] is (code is not ErrorCode.AI_INVALID_RESPONSE), code


def test_every_4xx_is_non_retryable_except_rate_limiting() -> None:
    """Contract 21.3: deterministic 4xx; 21.2.1: 429 is the explicit exception."""
    for code, status in ERROR_HTTP_STATUS.items():
        if 400 <= status < 500:
            assert ERROR_RETRYABLE[code] is (code is ErrorCode.RATE_LIMITED), code


def test_invalid_response_is_never_retryable() -> None:
    assert ERROR_RETRYABLE[ErrorCode.AI_INVALID_RESPONSE] is False


def test_internal_and_chunking_errors_follow_the_5xx_rule() -> None:
    """Reviewed with the addendum: a 500 is retried by the Backend regardless,
    so the metadata must say so rather than contradict the status."""
    assert ERROR_RETRYABLE[ErrorCode.INTERNAL_ERROR] is True
    assert ERROR_RETRYABLE[ErrorCode.CHUNKING_FAILED] is True


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
    assert "openrouter" not in message.lower()
    assert "qwen" not in message.lower()


def test_the_documented_model_unavailable_message_is_used() -> None:
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
        # generation
        (InvalidGenerationOutputError("x"), ErrorCode.AI_INVALID_RESPONSE),
        (GenerationRateLimitedError("x"), ErrorCode.RATE_LIMITED),
        (GenerationTimeoutError("x"), ErrorCode.TIMEOUT),
        (GenerationUnavailableError("x"), ErrorCode.GENERATION_UNAVAILABLE),
        (GenerationDisabledError("x"), ErrorCode.GENERATION_UNAVAILABLE),
    ],
)
def test_exceptions_map_to_their_code(exception: Exception, expected: ErrorCode) -> None:
    assert classify(exception) is expected


def test_malformed_output_is_distinct_from_an_unavailable_provider() -> None:
    """The distinction the addendum exists for: one is retried, one is not."""
    assert classify(InvalidGenerationOutputError("x")) is not classify(
        GenerationUnavailableError("x")
    )
    assert ERROR_RETRYABLE[classify(InvalidGenerationOutputError("x"))] is False
    assert ERROR_RETRYABLE[classify(GenerationUnavailableError("x"))] is True


def test_a_bare_value_error_is_not_treated_as_a_bad_request() -> None:
    """Only InvalidUserScopeError means bad input; a misconfigured provider
    name is a defect and must not be reported as the caller's fault."""
    assert classify(ValueError("unknown vector provider 'qdrant'")) is ErrorCode.INTERNAL_ERROR
    assert classify(InvalidUserScopeError("blank")) is ErrorCode.INVALID_INPUT


def test_invalid_user_scope_is_still_a_value_error() -> None:
    assert issubclass(InvalidUserScopeError, ValueError)


def test_a_generation_timeout_is_a_timeout_error_too() -> None:
    """Callers handling the standard TimeoutError keep working."""
    assert issubclass(GenerationTimeoutError, TimeoutError)


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
    assert error.retry_after_seconds is None


def test_service_error_accepts_overrides() -> None:
    error = AiServiceError(
        ErrorCode.RATE_LIMITED, message="Slow down", retryable=True, retry_after_seconds=7
    )

    assert error.public_message == "Slow down"
    assert error.retry_after_seconds == 7
