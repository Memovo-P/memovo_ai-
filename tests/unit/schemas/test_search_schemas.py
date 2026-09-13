"""Validation behaviour for the ``/ai/memories/search`` schemas.

Contract v1.9, sections 14.1 and 16.1-16.4.
"""

import pytest
from pydantic import ValidationError

from memovo_ai.schemas import (
    SearchMemoryNoMatchResponse,
    SearchMemoryRequest,
    SearchMemoryResponseAdapter,
    SearchMemorySuccessResponse,
    SearchResult,
    SearchResultChunk,
)

pytestmark = pytest.mark.unit

VALID_REQUEST = {
    "query": "What did I save about MongoDB vector search?",
    "userId": "user_123",
}

VALID_CHUNK = {"chunkId": "chunk_001", "content": "MongoDB Vector Search..."}

VALID_RESULT = {
    "memoryId": "memory_456",
    "score": 0.94,
    "title": "MongoDB Vector Search",
    "tags": ["mongodb", "vector-search"],
    "chunks": [VALID_CHUNK, {"chunkId": "chunk_002", "content": "Vector indexes are used..."}],
}


# --------------------------------------------------------------------------
# SearchMemoryRequest: userId and query, nothing else
# --------------------------------------------------------------------------
def test_valid_request_is_accepted() -> None:
    request = SearchMemoryRequest.model_validate(VALID_REQUEST)

    assert request.query == "What did I save about MongoDB vector search?"
    assert request.user_id == "user_123"


@pytest.mark.parametrize("field", ["query", "userId"])
def test_missing_required_field_is_rejected(field: str) -> None:
    payload = {k: v for k, v in VALID_REQUEST.items() if k != field}

    with pytest.raises(ValidationError):
        SearchMemoryRequest.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "value"), [("type", "note"), ("memoryType", "link"), ("limit", 5), ("topK", 5)]
)
def test_backend_owned_filter_and_limit_fields_are_rejected(field: str, value: object) -> None:
    """Type filtering and the public limit are applied by the Backend after
    this response (sections 14.1, 14.2 and 16.5); the AI never receives them."""
    with pytest.raises(ValidationError):
        SearchMemoryRequest.model_validate({**VALID_REQUEST, field: value})


@pytest.mark.parametrize(
    "field", ["token", "jwt", "authorization", "intent", "conversationId", "threshold"]
)
def test_auth_and_routing_fields_are_rejected(field: str) -> None:
    with pytest.raises(ValidationError):
        SearchMemoryRequest.model_validate({**VALID_REQUEST, field: "x"})


def test_snake_case_user_id_is_rejected() -> None:
    with pytest.raises(ValidationError):
        SearchMemoryRequest.model_validate({"query": "q", "user_id": "user_123"})


@pytest.mark.parametrize(("field", "value"), [("query", 1), ("userId", 123), ("userId", None)])
def test_wrong_type_is_rejected(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        SearchMemoryRequest.model_validate({**VALID_REQUEST, field: value})


# --------------------------------------------------------------------------
# SearchResultChunk: chunkId and content only
# --------------------------------------------------------------------------
def test_a_chunk_exposes_only_chunk_id_and_content() -> None:
    assert set(SearchResultChunk.model_json_schema()["properties"]) == {"chunkId", "content"}


def test_chunk_index_is_not_part_of_the_search_response() -> None:
    """It still orders the chunks internally; it is just not on the wire."""
    with pytest.raises(ValidationError):
        SearchResultChunk.model_validate({**VALID_CHUNK, "chunkIndex": 0})


def test_search_result_chunk_does_not_accept_an_embedding() -> None:
    """Embeddings must never be exposed through retrieval results."""
    with pytest.raises(ValidationError):
        SearchResultChunk.model_validate({**VALID_CHUNK, "embedding": [0.1, 0.2]})


@pytest.mark.parametrize("field", ["chunkId", "content"])
def test_an_empty_chunk_field_is_rejected(field: str) -> None:
    """Section 16.4: the Backend validates both as non-empty; so do we."""
    with pytest.raises(ValidationError):
        SearchResultChunk.model_validate({**VALID_CHUNK, field: ""})


# --------------------------------------------------------------------------
# SearchResult
# --------------------------------------------------------------------------
def test_result_with_multiple_chunks_is_accepted() -> None:
    result = SearchResult.model_validate(VALID_RESULT)

    assert [c.chunk_id for c in result.chunks] == ["chunk_001", "chunk_002"]
    assert result.score == 0.94


def test_an_empty_chunk_list_is_accepted() -> None:
    assert SearchResult.model_validate({**VALID_RESULT, "chunks": []}).chunks == []


def test_an_empty_memory_id_is_rejected() -> None:
    with pytest.raises(ValidationError):
        SearchResult.model_validate({**VALID_RESULT, "memoryId": ""})


@pytest.mark.parametrize("field", ["whySaved", "type", "content", "url", "userId", "id"])
def test_backend_only_fields_are_never_part_of_a_result(field: str) -> None:
    """Section 16.2: the Backend fetches or maps these itself."""
    with pytest.raises(ValidationError):
        SearchResult.model_validate({**VALID_RESULT, field: "x"})


def test_integer_score_is_accepted_but_string_score_is_not() -> None:
    assert SearchResult.model_validate({**VALID_RESULT, "score": 1}).score == 1.0

    with pytest.raises(ValidationError):
        SearchResult.model_validate({**VALID_RESULT, "score": "0.94"})


@pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_score_is_rejected(score: float) -> None:
    """Section 16.4: finite, not NaN, not Infinity."""
    with pytest.raises(ValidationError):
        SearchResult.model_validate({**VALID_RESULT, "score": score})


@pytest.mark.parametrize("score", [-1.0, 0.0, 1.0, 1.5])
def test_score_range_is_not_constrained(score: float) -> None:
    """Typically 0.0-1.0, but bounding it would hard-code the metric."""
    assert SearchResult.model_validate({**VALID_RESULT, "score": score}).score == score


# --------------------------------------------------------------------------
# SearchMemorySuccessResponse
# --------------------------------------------------------------------------
def test_valid_success_response_is_accepted() -> None:
    response = SearchMemorySuccessResponse.model_validate(
        {"found": True, "results": [VALID_RESULT]}
    )

    assert response.found is True
    assert len(response.results) == 1


def test_success_response_rejects_found_false() -> None:
    with pytest.raises(ValidationError):
        SearchMemorySuccessResponse.model_validate({"found": False, "results": []})


def test_success_response_rejects_a_message_field() -> None:
    with pytest.raises(ValidationError):
        SearchMemorySuccessResponse.model_validate(
            {"found": True, "results": [VALID_RESULT], "message": "x"}
        )


def test_multiple_results_are_accepted_in_order() -> None:
    second = {**VALID_RESULT, "memoryId": "memory_789", "score": 0.81}
    response = SearchMemorySuccessResponse.model_validate(
        {"found": True, "results": [VALID_RESULT, second]}
    )

    assert [r.memory_id for r in response.results] == ["memory_456", "memory_789"]


# --------------------------------------------------------------------------
# SearchMemoryNoMatchResponse: exactly {"found": false, "results": []}
# --------------------------------------------------------------------------
def test_exact_no_match_shape_is_accepted() -> None:
    response = SearchMemoryNoMatchResponse.model_validate({"found": False, "results": []})

    assert response.found is False
    assert response.results == []


def test_no_match_response_is_constructible_from_the_defaults() -> None:
    assert SearchMemoryNoMatchResponse().model_dump() == {"found": False, "results": []}


def test_no_match_response_has_no_message_field() -> None:
    """The pre-v1.9 no-match carried a fixed message; section 16.1 does not."""
    assert "message" not in SearchMemoryNoMatchResponse.model_fields

    with pytest.raises(ValidationError):
        SearchMemoryNoMatchResponse.model_validate(
            {"found": False, "results": [], "message": "I couldn't find a relevant memory"}
        )


def test_no_match_response_rejects_populated_results() -> None:
    with pytest.raises(ValidationError):
        SearchMemoryNoMatchResponse.model_validate({"found": False, "results": [VALID_RESULT]})


def test_no_match_response_rejects_found_true() -> None:
    with pytest.raises(ValidationError):
        SearchMemoryNoMatchResponse.model_validate({"found": True, "results": []})


# --------------------------------------------------------------------------
# Response union
# --------------------------------------------------------------------------
def test_union_routes_to_the_success_arm() -> None:
    response = SearchMemoryResponseAdapter.validate_python(
        {"found": True, "results": [VALID_RESULT]}
    )

    assert isinstance(response, SearchMemorySuccessResponse)


def test_union_routes_to_the_no_match_arm() -> None:
    response = SearchMemoryResponseAdapter.validate_python({"found": False, "results": []})

    assert isinstance(response, SearchMemoryNoMatchResponse)


@pytest.mark.parametrize(
    "payload",
    [
        {"found": False, "results": [VALID_RESULT]},
        {"found": True, "results": [], "message": "x"},
        {"results": []},
        {"found": "true", "results": []},
        {"found": True},
    ],
)
def test_contradictory_response_states_are_rejected(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        SearchMemoryResponseAdapter.validate_python(payload)
