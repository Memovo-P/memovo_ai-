"""Validation behaviour for the standardized public error envelope."""

import pytest
from pydantic import ValidationError

from memovo_ai.schemas import ErrorCode, ErrorDetail, ErrorResponse

pytestmark = pytest.mark.unit

SUPPORTED_CODES = [
    "INVALID_INPUT",
    "INVALID_REQUEST",
    "EMBEDDING_FAILED",
    "CHUNKING_FAILED",
    "VECTOR_SEARCH_FAILED",
    "MODEL_UNAVAILABLE",
    "TIMEOUT",
    "INTERNAL_ERROR",
]

VALID_DETAIL = {
    "code": "MODEL_UNAVAILABLE",
    "message": "Embedding model is temporarily unavailable",
    "retryable": True,
}


def test_enum_contains_exactly_the_supported_codes() -> None:
    assert [c.value for c in ErrorCode] == SUPPORTED_CODES


@pytest.mark.parametrize("code", SUPPORTED_CODES)
def test_every_supported_code_is_accepted(code: str) -> None:
    detail = ErrorDetail.model_validate({**VALID_DETAIL, "code": code})

    assert detail.code == ErrorCode(code)


@pytest.mark.parametrize(
    "code",
    [
        "UNKNOWN_ERROR",
        "invalid_input",
        "InvalidInput",
        "",
        "RATE_LIMITED",
        "NOT_FOUND",
        1,
        None,
    ],
)
def test_unsupported_code_is_rejected(code: object) -> None:
    with pytest.raises(ValidationError):
        ErrorDetail.model_validate({**VALID_DETAIL, "code": code})


@pytest.mark.parametrize("field", ["code", "message", "retryable"])
def test_missing_required_field_is_rejected(field: str) -> None:
    payload = {k: v for k, v in VALID_DETAIL.items() if k != field}

    with pytest.raises(ValidationError):
        ErrorDetail.model_validate(payload)


@pytest.mark.parametrize("value", [1, 0, "true", "false", None])
def test_non_boolean_retryable_is_rejected(value: object) -> None:
    """Strict mode: retryable must be a real boolean."""
    with pytest.raises(ValidationError):
        ErrorDetail.model_validate({**VALID_DETAIL, "retryable": value})


@pytest.mark.parametrize(
    "field",
    [
        "exceptionType",
        "stackTrace",
        "traceback",
        "internalError",
        "debug",
        "modelPath",
        "provider",
        "database",
        "connectionString",
    ],
)
def test_internal_detail_fields_are_rejected(field: str) -> None:
    """The envelope must not carry a channel for leaking internals."""
    with pytest.raises(ValidationError):
        ErrorDetail.model_validate({**VALID_DETAIL, field: "leaked"})


def test_error_detail_has_exactly_three_fields() -> None:
    assert set(ErrorDetail.model_fields) == {"code", "message", "retryable"}


def test_envelope_wraps_the_detail() -> None:
    response = ErrorResponse.model_validate({"error": VALID_DETAIL})

    assert response.error.code is ErrorCode.MODEL_UNAVAILABLE
    assert response.error.retryable is True


def test_envelope_has_exactly_one_field() -> None:
    assert set(ErrorResponse.model_fields) == {"error"}


def test_extra_field_on_the_envelope_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ErrorResponse.model_validate({"error": VALID_DETAIL, "status": 503})


def test_envelope_requires_the_error_object() -> None:
    with pytest.raises(ValidationError):
        ErrorResponse.model_validate({})


def test_retryable_has_no_default() -> None:
    """Each raise site must state retryability rather than inherit a guess."""
    assert ErrorDetail.model_fields["retryable"].is_required()
