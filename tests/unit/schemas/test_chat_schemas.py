"""Validation behaviour for the ``/ai/chat/memories`` and prepare-note schemas.

Contract v1.9, sections 6 and 17. History limits are the Backend's to
enforce (17.2.1); only the shape is validated here.
"""

import pytest
from pydantic import ValidationError

from memovo_ai.schemas import (
    ChatMemoriesRequest,
    ChatMemoriesResponse,
    ChatSource,
    ChatSourceChunk,
    PrepareNoteRequest,
    PrepareNoteResponse,
)

pytestmark = pytest.mark.unit

REQUEST: dict[str, object] = {
    "userId": "user-id",
    "message": "User's question about their memories",
    "conversationId": "optional-conversation-id",
    "history": [
        {"role": "user", "content": "Previous message"},
        {"role": "assistant", "content": "Previous response"},
    ],
}


def rejected(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ChatMemoriesRequest.model_validate(payload)


# --------------------------------------------------------------------------
# Request
# --------------------------------------------------------------------------
def test_the_documented_request_is_accepted() -> None:
    request = ChatMemoriesRequest.model_validate(REQUEST)

    assert request.user_id == "user-id"
    assert request.conversation_id == "optional-conversation-id"
    assert request.history is not None
    assert [turn.role for turn in request.history] == ["user", "assistant"]


@pytest.mark.parametrize("field", ["userId", "message"])
def test_required_fields(field: str) -> None:
    rejected({key: value for key, value in REQUEST.items() if key != field})


def test_history_and_conversation_id_may_be_omitted() -> None:
    request = ChatMemoriesRequest.model_validate({"userId": "u", "message": "m"})

    assert request.history is None
    assert request.conversation_id is None


def test_an_empty_history_is_accepted() -> None:
    assert ChatMemoriesRequest.model_validate({**REQUEST, "history": []}).history == []


@pytest.mark.parametrize("field", ["history", "conversationId"])
def test_optional_is_not_nullable(field: str) -> None:
    """The plan: optional must not be read as nullable without evidence."""
    rejected({**REQUEST, field: None})


@pytest.mark.parametrize("role", ["system", "tool", "function", "developer", "User", ""])
def test_only_user_and_assistant_roles_are_allowed(role: str) -> None:
    rejected({**REQUEST, "history": [{"role": role, "content": "x"}]})


def test_history_entries_carry_only_role_and_content() -> None:
    rejected({**REQUEST, "history": [{"role": "user", "content": "x", "name": "n"}]})


def test_history_content_must_be_a_string() -> None:
    rejected({**REQUEST, "history": [{"role": "user", "content": None}]})


def test_the_ai_does_not_enforce_history_size_limits() -> None:
    """Thirty-one messages of 7,000 characters are the Backend's problem to
    bound before sending; this service accepts them whole and untrimmed."""
    oversized = [{"role": "user", "content": "x" * 7_000} for _ in range(31)]

    request = ChatMemoriesRequest.model_validate({**REQUEST, "history": oversized})

    assert request.history is not None
    assert len(request.history) == 31
    assert all(len(turn.content) == 7_000 for turn in request.history)


def test_no_current_message_length_cap_is_invented() -> None:
    assert len(
        ChatMemoriesRequest.model_validate({**REQUEST, "message": "m" * 50_000}).message
    ) == (50_000)


@pytest.mark.parametrize("field", ["type", "limit", "token", "authorization", "intent"])
def test_unknown_fields_are_rejected(field: str) -> None:
    rejected({**REQUEST, field: "x"})


def test_snake_case_is_rejected() -> None:
    rejected({"user_id": "u", "message": "m"})


# --------------------------------------------------------------------------
# Response
# --------------------------------------------------------------------------
def test_the_response_has_exactly_three_fields() -> None:
    assert set(ChatMemoriesResponse.model_json_schema()["properties"]) == {
        "found",
        "answer",
        "sources",
    }


@pytest.mark.parametrize("field", ["sufficient", "confidence", "citations", "status"])
def test_no_extra_response_field_can_be_added(field: str) -> None:
    with pytest.raises(ValidationError):
        ChatMemoriesResponse.model_validate(
            {"found": True, "answer": "a", "sources": [], field: True}
        )


def test_a_source_carries_memory_id_score_title_and_chunks() -> None:
    schema = ChatSource.model_json_schema()

    assert set(schema["properties"]) == {"memoryId", "score", "title", "chunks"}
    assert set(ChatSourceChunk.model_json_schema()["properties"]) == {"chunkId", "content"}


@pytest.mark.parametrize("score", [float("nan"), float("inf")])
def test_a_non_finite_source_score_is_rejected(score: float) -> None:
    with pytest.raises(ValidationError):
        ChatSource.model_validate({"memoryId": "m", "score": score, "title": "t", "chunks": []})


@pytest.mark.parametrize("field", ["tags", "chunkIndex", "index", "citationId"])
def test_sources_carry_no_index_or_citation_fields(field: str) -> None:
    with pytest.raises(ValidationError):
        ChatSource.model_validate(
            {"memoryId": "m", "score": 0.9, "title": "t", "chunks": [], field: "x"}
        )


# --------------------------------------------------------------------------
# Prepare note
# --------------------------------------------------------------------------
def test_prepare_note_request_is_content_only() -> None:
    assert PrepareNoteRequest.model_validate({"content": "c"}).content == "c"

    for extra in ("conversationId", "userId", "title", "tags"):
        with pytest.raises(ValidationError):
            PrepareNoteRequest.model_validate({"content": "c", extra: "x"})


def test_prepare_note_response_is_title_and_content_only() -> None:
    assert set(PrepareNoteResponse.model_json_schema()["properties"]) == {"title", "content"}

    for extra in ("tags", "userId", "createdAt", "updatedAt", "id", "memoryId"):
        with pytest.raises(ValidationError):
            PrepareNoteResponse.model_validate({"title": "t", "content": "c", extra: "x"})
