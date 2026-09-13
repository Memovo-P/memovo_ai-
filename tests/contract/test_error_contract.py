"""Wire-format contract for the standardized public error envelope."""

import json

import pytest

from memovo_ai.schemas import ErrorCode, ErrorDetail, ErrorResponse

pytestmark = pytest.mark.contract

ERROR_JSON = """
{
  "error": {
    "code": "MODEL_UNAVAILABLE",
    "message": "Embedding model is temporarily unavailable",
    "retryable": true
  }
}
"""

SUPPORTED_CODES = [
    "INVALID_INPUT",
    "INVALID_REQUEST",
    "EMBEDDING_FAILED",
    "CHUNKING_FAILED",
    "VECTOR_SEARCH_FAILED",
    "MODEL_UNAVAILABLE",
    "TIMEOUT",
    "INTERNAL_ERROR",
    "GENERATION_UNAVAILABLE",
    "RATE_LIMITED",
    "AI_INVALID_RESPONSE",
]


def test_documented_error_validates() -> None:
    response = ErrorResponse.model_validate_json(ERROR_JSON)

    assert response.error.code is ErrorCode.MODEL_UNAVAILABLE
    assert response.error.message == "Embedding model is temporarily unavailable"
    assert response.error.retryable is True


def test_error_round_trips_to_the_documented_json() -> None:
    response = ErrorResponse.model_validate_json(ERROR_JSON)

    assert json.loads(response.model_dump_json()) == json.loads(ERROR_JSON)


def test_envelope_serializes_exactly_four_keys() -> None:
    response = ErrorResponse.model_validate_json(ERROR_JSON)
    payload = json.loads(response.model_dump_json())

    assert set(payload) == {"error"}
    assert set(payload["error"]) == {"code", "message", "retryable"}


@pytest.mark.parametrize("code", SUPPORTED_CODES)
def test_every_supported_code_serializes_as_its_contract_string(code: str) -> None:
    response = ErrorResponse(
        error=ErrorDetail(code=ErrorCode(code), message="failure", retryable=False)
    )
    payload = json.loads(response.model_dump_json())

    assert payload == {"error": {"code": code, "message": "failure", "retryable": False}}


@pytest.mark.parametrize("code", SUPPORTED_CODES)
def test_every_supported_code_parses_from_json(code: str) -> None:
    raw = json.dumps({"error": {"code": code, "message": "failure", "retryable": True}})

    assert ErrorResponse.model_validate_json(raw).error.code == ErrorCode(code)


def test_error_schema_exposes_only_the_supported_codes() -> None:
    schema = ErrorResponse.model_json_schema()
    enum_values = schema["$defs"]["ErrorCode"]["enum"]

    assert enum_values == SUPPORTED_CODES


def test_error_detail_required_fields_are_exact() -> None:
    schema = ErrorDetail.model_json_schema()

    assert set(schema["properties"]) == {"code", "message", "retryable"}
    assert set(schema["required"]) == {"code", "message", "retryable"}
