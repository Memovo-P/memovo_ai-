"""Wire-format contract for ``POST /ai/memories/search``.

The no-match payload is the exact shape locked by doc 01, decision 20.
"""

import json

import pytest

from memovo_ai.schemas import (
    NO_MATCH_MESSAGE,
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
  "query": "What did I save about MongoDB vector search?",
  "userId": "user_123"
}
"""

SUCCESS_JSON = """
{
  "found": true,
  "results": [
    {
      "memoryId": "memory_456",
      "score": 0.94,
      "title": "MongoDB Vector Search",
      "tags": ["mongodb", "vector-search"],
      "chunks": [
        {
          "chunkId": "chunk_001",
          "chunkIndex": 0,
          "content": "MongoDB Vector Search..."
        },
        {
          "chunkId": "chunk_002",
          "chunkIndex": 1,
          "content": "Vector indexes are used..."
        }
      ]
    }
  ]
}
"""

NO_MATCH_JSON = """
{
  "found": false,
  "results": [],
  "message": "I couldn't find a relevant memory"
}
"""


# --------------------------------------------------------------------------
# Request
# --------------------------------------------------------------------------
def test_documented_request_validates() -> None:
    request = SearchMemoryRequest.model_validate_json(REQUEST_JSON)

    assert request.query == "What did I save about MongoDB vector search?"
    assert request.user_id == "user_123"


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
    assert response.results[0].memory_id == "memory_456"
    assert response.results[0].score == 0.94
    assert len(response.results[0].chunks) == 2


def test_success_response_round_trips_to_the_documented_json() -> None:
    response = SearchMemoryResponseAdapter.validate_json(SUCCESS_JSON)

    assert json.loads(response.model_dump_json()) == json.loads(SUCCESS_JSON)


def test_success_result_public_field_names_are_exact() -> None:
    schema = SearchResult.model_json_schema()

    assert set(schema["properties"]) == {"memoryId", "score", "title", "tags", "chunks"}
    assert set(schema["required"]) == {"memoryId", "score", "title", "tags", "chunks"}


def test_returned_chunk_public_field_names_are_exact() -> None:
    schema = SearchResultChunk.model_json_schema()

    assert set(schema["properties"]) == {"chunkId", "chunkIndex", "content"}
    assert set(schema["required"]) == {"chunkId", "chunkIndex", "content"}


def test_search_results_never_serialize_an_embedding() -> None:
    response = SearchMemoryResponseAdapter.validate_json(SUCCESS_JSON)

    assert "embedding" not in response.model_dump_json()


def test_success_response_never_emits_snake_case_keys() -> None:
    response = SearchMemoryResponseAdapter.validate_json(SUCCESS_JSON)
    serialized = response.model_dump_json()

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
    assert response.message == NO_MATCH_MESSAGE


def test_no_match_response_round_trips_to_the_documented_json() -> None:
    response = SearchMemoryResponseAdapter.validate_json(NO_MATCH_JSON)

    assert json.loads(response.model_dump_json()) == json.loads(NO_MATCH_JSON)


def test_default_no_match_response_serializes_to_the_documented_json() -> None:
    """Building the no-match response from defaults yields the locked payload."""
    payload = json.loads(SearchMemoryNoMatchResponse().model_dump_json())

    assert payload == json.loads(NO_MATCH_JSON)


def test_locked_no_match_message_constant() -> None:
    assert NO_MATCH_MESSAGE == "I couldn't find a relevant memory"


def test_success_response_has_no_message_field() -> None:
    """`message` belongs to the no-match response only."""
    assert "message" not in SearchMemorySuccessResponse.model_fields
