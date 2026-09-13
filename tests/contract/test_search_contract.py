"""Wire-format contract for ``POST /ai/memories/search``.

Payloads copied verbatim from ``docs/AI_CONTRACT_FINAL.md`` v1.9, sections
14.1 and 16.1.
"""

import json

import pytest

from memovo_ai.schemas import (
    SearchMemoryNoMatchResponse,
    SearchMemoryRequest,
    SearchMemoryResponseAdapter,
    SearchMemorySuccessResponse,
    SearchResult,
    SearchResultChunk,
)

pytestmark = pytest.mark.contract

REQUEST_JSON = """
{
  "userId": "user-id",
  "query": "What did I save about ...?"
}
"""

SUCCESS_JSON = """
{
  "found": true,
  "results": [
    {
      "memoryId": "507f1f77bcf86cd799439013",
      "score": 0.94,
      "title": "MongoDB Performance Guide",
      "tags": ["mongodb", "performance"],
      "chunks": [
        {
          "chunkId": "chunk_001",
          "content": "MongoDB indexing strategies improve query performance..."
        }
      ]
    }
  ]
}
"""

NO_MATCH_JSON = """
{
  "found": false,
  "results": []
}
"""


# --------------------------------------------------------------------------
# Request
# --------------------------------------------------------------------------
def test_documented_request_validates() -> None:
    request = SearchMemoryRequest.model_validate_json(REQUEST_JSON)

    assert request.query == "What did I save about ...?"
    assert request.user_id == "user-id"


def test_request_public_field_names_are_exact() -> None:
    schema = SearchMemoryRequest.model_json_schema()

    assert set(schema["properties"]) == {"query", "userId"}
    assert set(schema["required"]) == {"query", "userId"}


def test_request_round_trips_to_the_documented_json() -> None:
    request = SearchMemoryRequest.model_validate_json(REQUEST_JSON)

    assert json.loads(request.model_dump_json()) == json.loads(REQUEST_JSON)


# --------------------------------------------------------------------------
# Success response
# --------------------------------------------------------------------------
def test_documented_success_response_validates() -> None:
    response = SearchMemoryResponseAdapter.validate_json(SUCCESS_JSON)

    assert isinstance(response, SearchMemorySuccessResponse)
    assert response.results[0].memory_id == "507f1f77bcf86cd799439013"
    assert response.results[0].score == 0.94
    assert response.results[0].chunks[0].chunk_id == "chunk_001"


def test_success_response_round_trips_to_the_documented_json() -> None:
    response = SearchMemoryResponseAdapter.validate_json(SUCCESS_JSON)

    assert json.loads(response.model_dump_json()) == json.loads(SUCCESS_JSON)


def test_success_result_public_field_names_are_exact() -> None:
    schema = SearchResult.model_json_schema()

    assert set(schema["properties"]) == {"memoryId", "score", "title", "tags", "chunks"}
    assert set(schema["required"]) == {"memoryId", "score", "title", "tags", "chunks"}


def test_returned_chunk_public_field_names_are_exact() -> None:
    schema = SearchResultChunk.model_json_schema()

    assert set(schema["properties"]) == {"chunkId", "content"}
    assert set(schema["required"]) == {"chunkId", "content"}


def test_search_results_never_serialize_an_embedding_or_chunk_index() -> None:
    serialized = SearchMemoryResponseAdapter.validate_json(SUCCESS_JSON).model_dump_json()

    assert "embedding" not in serialized
    assert "chunkIndex" not in serialized


def test_success_response_never_emits_snake_case_keys() -> None:
    serialized = SearchMemoryResponseAdapter.validate_json(SUCCESS_JSON).model_dump_json()

    for snake in ("memory_id", "chunk_id", "chunk_index", "user_id"):
        assert snake not in serialized


# --------------------------------------------------------------------------
# No-match response
# --------------------------------------------------------------------------
def test_documented_no_match_response_validates() -> None:
    response = SearchMemoryResponseAdapter.validate_json(NO_MATCH_JSON)

    assert isinstance(response, SearchMemoryNoMatchResponse)
    assert response.found is False
    assert response.results == []


def test_no_match_response_round_trips_to_the_documented_json() -> None:
    response = SearchMemoryResponseAdapter.validate_json(NO_MATCH_JSON)

    assert json.loads(response.model_dump_json()) == json.loads(NO_MATCH_JSON)


def test_default_no_match_response_serializes_to_the_documented_json() -> None:
    """Building the no-match response from defaults yields the exact payload."""
    payload = json.loads(SearchMemoryNoMatchResponse().model_dump_json())

    assert payload == json.loads(NO_MATCH_JSON)


def test_no_response_arm_has_a_message_field() -> None:
    assert "message" not in SearchMemorySuccessResponse.model_fields
    assert "message" not in SearchMemoryNoMatchResponse.model_fields
