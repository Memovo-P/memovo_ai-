"""Wire-format contract for ``POST /ai/chat/memories`` and prepare-note.

Payloads copied verbatim from ``docs/AI_CONTRACT_FINAL.md`` v1.9, sections
6.1, 6.2, 17.2 and 17.3.
"""

import json

import pytest

from memovo_ai.schemas import (
    ChatMemoriesRequest,
    ChatMemoriesResponse,
    PrepareNoteRequest,
    PrepareNoteResponse,
)

pytestmark = pytest.mark.contract

CHAT_REQUEST_JSON = """
{
  "userId": "user-id",
  "message": "User's question about their memories",
  "conversationId": "optional-conversation-id",
  "history": [
    { "role": "user", "content": "Previous message" },
    { "role": "assistant", "content": "Previous response" }
  ]
}
"""

CHAT_RESPONSE_JSON = """
{
  "found": true,
  "answer": "Natural language answer grounded in retrieved memories",
  "sources": [
    {
      "memoryId": "memory-id",
      "score": 0.89,
      "title": "Memory title",
      "chunks": [
        {
          "chunkId": "chunk-id",
          "content": "Relevant chunk content"
        }
      ]
    }
  ]
}
"""

NO_MATCH_JSON = """
{
  "found": false,
  "answer": "I couldn't find any relevant memories for that.",
  "sources": []
}
"""

NOTE_REQUEST_JSON = """
{
  "content": "The content I want to save as a note."
}
"""

NOTE_RESPONSE_JSON = """
{
  "title": "Prepared note title",
  "content": "Cleaned and organized note content"
}
"""


def test_the_documented_chat_request_validates_and_round_trips() -> None:
    request = ChatMemoriesRequest.model_validate_json(CHAT_REQUEST_JSON)

    assert request.message == "User's question about their memories"
    assert json.loads(request.model_dump_json()) == json.loads(CHAT_REQUEST_JSON)


def test_a_minimal_chat_request_round_trips_without_optional_fields() -> None:
    request = ChatMemoriesRequest.model_validate_json('{"userId": "u", "message": "m"}')

    assert json.loads(request.model_dump_json(exclude_none=True)) == {"userId": "u", "message": "m"}


def test_chat_request_public_field_names_are_exact() -> None:
    schema = ChatMemoriesRequest.model_json_schema()

    assert set(schema["properties"]) == {"userId", "message", "conversationId", "history"}
    assert set(schema["required"]) == {"userId", "message"}


def test_the_documented_chat_response_validates_and_round_trips() -> None:
    response = ChatMemoriesResponse.model_validate_json(CHAT_RESPONSE_JSON)

    assert response.found is True
    assert response.sources[0].memory_id == "memory-id"
    assert json.loads(response.model_dump_json()) == json.loads(CHAT_RESPONSE_JSON)


def test_the_no_match_chat_response_round_trips() -> None:
    response = ChatMemoriesResponse.model_validate_json(NO_MATCH_JSON)

    assert response.found is False
    assert response.sources == []
    assert json.loads(response.model_dump_json()) == json.loads(NO_MATCH_JSON)


def test_chat_response_public_field_names_are_exact() -> None:
    schema = ChatMemoriesResponse.model_json_schema()

    assert set(schema["properties"]) == {"found", "answer", "sources"}
    assert set(schema["required"]) == {"found", "answer", "sources"}


def test_chat_response_never_emits_snake_case_keys() -> None:
    serialized = ChatMemoriesResponse.model_validate_json(CHAT_RESPONSE_JSON).model_dump_json()

    for snake in ("memory_id", "chunk_id", "chunk_index", "user_id"):
        assert snake not in serialized


def test_the_documented_note_payloads_validate_and_round_trip() -> None:
    request = PrepareNoteRequest.model_validate_json(NOTE_REQUEST_JSON)
    response = PrepareNoteResponse.model_validate_json(NOTE_RESPONSE_JSON)

    assert json.loads(request.model_dump_json()) == json.loads(NOTE_REQUEST_JSON)
    assert json.loads(response.model_dump_json()) == json.loads(NOTE_RESPONSE_JSON)


def test_note_public_field_names_are_exact() -> None:
    assert set(PrepareNoteRequest.model_json_schema()["required"]) == {"content"}
    assert set(PrepareNoteResponse.model_json_schema()["required"]) == {"title", "content"}
