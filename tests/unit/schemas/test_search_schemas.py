"""Validation behaviour for the ``/ai/memories/search`` schemas."""

import pytest
from pydantic import ValidationError

from memovo_ai.schemas import (
    NO_MATCH_MESSAGE,
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

VALID_CHUNK = {"chunkId": "chunk_001", "chunkIndex": 0, "content": "MongoDB Vector Search..."}

VALID_RESULT = {
    "memoryId": "memory_456",
    "score": 0.94,
    "title": "MongoDB Vector Search",
    "tags": ["mongodb", "vector-search"],
    "chunks": [
        VALID_CHUNK,
        {"chunkId": "chunk_002", "chunkIndex": 1, "content": "Vector indexes are used..."},
    ],
}


# --------------------------------------------------------------------------
# SearchMemoryRequest
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


def test_extra_field_is_rejected() -> None:
    with pytest.raises(ValidationError):
        SearchMemoryRequest.model_validate({**VALID_REQUEST, "topK": 5})


@pytest.mark.parametrize(
    "field", ["token", "jwt", "authorization", "intent", "conversationId", "threshold"]
)
def test_auth_and_routing_fields_are_rejected(field: str) -> None:
    """No auth input and no routing input: both are out of Sprint 1 scope."""
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
# SearchResultChunk / SearchResult
# --------------------------------------------------------------------------
def test_search_result_chunk_does_not_accept_an_embedding() -> None:
    """Embeddings must never be exposed through retrieval results."""
    with pytest.raises(ValidationError):
        SearchResultChunk.model_validate({**VALID_CHUNK, "embedding": [0.1, 0.2]})


def test_embedding_is_not_a_field_of_the_search_result_chunk() -> None:
    assert "embedding" not in SearchResultChunk.model_fields
    assert set(SearchResultChunk.model_json_schema()["properties"]) == {
        "chunkId",
        "chunkIndex",
        "content",
    }


def test_negative_chunk_index_is_rejected() -> None:
    with pytest.raises(ValidationError):
        SearchResultChunk.model_validate({**VALID_CHUNK, "chunkIndex": -1})


def test_result_with_multiple_chunks_is_accepted() -> None:
    result = SearchResult.model_validate(VALID_RESULT)

    assert [c.chunk_index for c in result.chunks] == [0, 1]
    assert result.score == 0.94


def test_extra_result_field_is_rejected() -> None:
    with pytest.raises(ValidationError):
        SearchResult.model_validate({**VALID_RESULT, "whySaved": "leaked"})


def test_integer_score_is_accepted_but_string_score_is_not() -> None:
    assert SearchResult.model_validate({**VALID_RESULT, "score": 1}).score == 1.0

    with pytest.raises(ValidationError):
        SearchResult.model_validate({**VALID_RESULT, "score": "0.94"})


@pytest.mark.parametrize("score", [-1.0, 0.0, 1.0, 1.5])
def test_score_range_is_not_constrained(score: float) -> None:
    """The similarity metric is recommended, not locked (doc 01, section 7).

    Bounding the score here would hard-code an assumption the architecture has
    not made.
    """
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


def test_success_response_rejects_the_no_match_message() -> None:
    with pytest.raises(ValidationError):
        SearchMemorySuccessResponse.model_validate(
            {"found": True, "results": [], "message": NO_MATCH_MESSAGE}
        )


def test_multiple_results_are_accepted_in_order() -> None:
    second = {**VALID_RESULT, "memoryId": "memory_789", "score": 0.81}
    response = SearchMemorySuccessResponse.model_validate(
        {"found": True, "results": [VALID_RESULT, second]}
    )

    assert [r.memory_id for r in response.results] == ["memory_456", "memory_789"]


# --------------------------------------------------------------------------
# SearchMemoryNoMatchResponse
# --------------------------------------------------------------------------
def test_exact_no_match_shape_is_accepted() -> None:
    response = SearchMemoryNoMatchResponse.model_validate(
        {"found": False, "results": [], "message": NO_MATCH_MESSAGE}
    )

    assert response.found is False
    assert response.results == []
    assert response.message == NO_MATCH_MESSAGE


def test_no_match_response_is_constructible_from_the_locked_defaults() -> None:
    assert SearchMemoryNoMatchResponse().model_dump() == {
        "found": False,
        "results": [],
        "message": NO_MATCH_MESSAGE,
    }


@pytest.mark.parametrize(
    "message",
    [
        "",
        "No relevant memory found",
        NO_MATCH_MESSAGE.lower(),
        NO_MATCH_MESSAGE + ".",
        NO_MATCH_MESSAGE.replace("couldn't", "could not"),
    ],
)
def test_wrong_no_match_message_is_rejected(message: str) -> None:
    """The message is locked by doc 01, decision 20."""
    with pytest.raises(ValidationError):
        SearchMemoryNoMatchResponse.model_validate(
            {"found": False, "results": [], "message": message}
        )


def test_no_match_response_rejects_populated_results() -> None:
    with pytest.raises(ValidationError):
        SearchMemoryNoMatchResponse.model_validate(
            {"found": False, "results": [VALID_RESULT], "message": NO_MATCH_MESSAGE}
        )


def test_no_match_response_rejects_found_true() -> None:
    with pytest.raises(ValidationError):
        SearchMemoryNoMatchResponse.model_validate(
            {"found": True, "results": [], "message": NO_MATCH_MESSAGE}
        )


# --------------------------------------------------------------------------
# Response union
# --------------------------------------------------------------------------
def test_union_routes_to_the_success_arm() -> None:
    response = SearchMemoryResponseAdapter.validate_python(
        {"found": True, "results": [VALID_RESULT]}
    )

    assert isinstance(response, SearchMemorySuccessResponse)


def test_union_routes_to_the_no_match_arm() -> None:
    response = SearchMemoryResponseAdapter.validate_python(
        {"found": False, "results": [], "message": NO_MATCH_MESSAGE}
    )

    assert isinstance(response, SearchMemoryNoMatchResponse)


@pytest.mark.parametrize(
    "payload",
    [
        # found=false carrying results
        {"found": False, "results": [VALID_RESULT], "message": NO_MATCH_MESSAGE},
        # found=true carrying the no-match message
        {"found": True, "results": [], "message": NO_MATCH_MESSAGE},
        # found=true carrying results AND a message
        {"found": True, "results": [VALID_RESULT], "message": NO_MATCH_MESSAGE},
        # missing discriminator
        {"results": []},
        # non-boolean discriminator
        {"found": "true", "results": []},
    ],
)
def test_contradictory_response_states_are_rejected(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        SearchMemoryResponseAdapter.validate_python(payload)
