"""Wire-format contract for ``POST /ai/memories/process``.

The payloads below are copied verbatim from
``memovo-ai-docs/03_IMPLEMENTATION_PLAN.md``, Phase 01. These tests fail if the
public JSON shape drifts, regardless of how the Python models are named.
"""

import json

import pytest

from memovo_ai.schemas import ProcessMemoryRequest, ProcessMemoryResponse

pytestmark = pytest.mark.contract

REQUEST_JSON = """
{
  "memoryId": "memory_456",
  "title": "MongoDB Vector Search",
  "description": "How Atlas Vector Search works...",
  "whySaved": "Useful for the memory project",
  "tags": ["mongodb", "vector-search", "ai"]
}
"""

RESPONSE_JSON = """
{
  "intent": "save_memory",
  "content": "How MongoDB Vector Search works...",
  "chunks": [
    {
      "chunkId": "chunk_001",
      "chunkIndex": 0,
      "content": "MongoDB Vector Search...",
      "embedding": [0.012, -0.034, 0.087]
    }
  ]
}
"""


def test_documented_request_validates() -> None:
    request = ProcessMemoryRequest.model_validate_json(REQUEST_JSON)

    assert request.memory_id == "memory_456"
    assert request.why_saved == "Useful for the memory project"


def test_request_public_field_names_are_exact() -> None:
    schema = ProcessMemoryRequest.model_json_schema()

    assert set(schema["properties"]) == {
        "memoryId",
        "title",
        "description",
        "whySaved",
        "tags",
    }
    assert set(schema["required"]) == {
        "memoryId",
        "title",
        "description",
        "whySaved",
        "tags",
    }


def test_request_round_trips_to_the_documented_json() -> None:
    request = ProcessMemoryRequest.model_validate_json(REQUEST_JSON)

    assert json.loads(request.model_dump_json()) == json.loads(REQUEST_JSON)


def test_documented_response_validates() -> None:
    response = ProcessMemoryResponse.model_validate_json(RESPONSE_JSON)

    assert response.intent == "save_memory"
    assert response.chunks[0].chunk_id == "chunk_001"
    assert response.chunks[0].chunk_index == 0
    assert response.chunks[0].embedding == [0.012, -0.034, 0.087]


def test_response_round_trips_to_the_documented_json() -> None:
    response = ProcessMemoryResponse.model_validate_json(RESPONSE_JSON)

    assert json.loads(response.model_dump_json()) == json.loads(RESPONSE_JSON)


def test_response_serializes_with_exact_public_field_names() -> None:
    response = ProcessMemoryResponse.model_validate_json(RESPONSE_JSON)
    payload = json.loads(response.model_dump_json())

    assert set(payload) == {"intent", "content", "chunks"}
    assert set(payload["chunks"][0]) == {"chunkId", "chunkIndex", "content", "embedding"}


def test_response_never_emits_snake_case_keys() -> None:
    response = ProcessMemoryResponse.model_validate_json(RESPONSE_JSON)
    serialized = response.model_dump_json()

    for snake in ("chunk_id", "chunk_index", "memory_id", "why_saved"):
        assert snake not in serialized


def test_process_response_carries_the_complete_chunk_set() -> None:
    """Reprocessing returns every chunk, never a diff (doc 01, decision 25).

    The schema shape supports that: `chunks` is the full list and there is no
    field for signalling a partial or delta response.
    """
    assert set(ProcessMemoryResponse.model_fields) == {"intent", "content", "chunks"}
    for absent in ("changedChunks", "removedChunkIds", "delta", "partial"):
        assert absent not in ProcessMemoryResponse.model_fields


def test_process_request_carries_no_user_identity() -> None:
    """Processing needs no user context; the Backend owns identity."""
    assert "user_id" not in ProcessMemoryRequest.model_fields
    assert "userId" not in ProcessMemoryRequest.model_json_schema()["properties"]
