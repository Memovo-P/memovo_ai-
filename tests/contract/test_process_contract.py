"""Wire-format contract for ``POST /ai/memories/process``.

The payloads below are copied verbatim from ``docs/AI_CONTRACT_FINAL.md``
v1.9, sections 8.1, 8.2 and 9. These tests fail if the public JSON shape
drifts, regardless of how the Python models are named.
"""

import json

import pytest

from memovo_ai.schemas import (
    LinkProcessRequest,
    LinkSource,
    NoteProcessRequest,
    ProcessedChunk,
    ProcessMemoryRequestAdapter,
    ProcessMemoryResponse,
)

pytestmark = pytest.mark.contract

NOTE_JSON = """
{
  "type": "note",
  "memoryId": "67890abcdef1234567890abc",
  "title": "MongoDB Performance Tips",
  "content": "Use indexes for frequently queried fields...",
  "tags": ["database", "performance"]
}
"""

LINK_JSON = """
{
  "type": "link",
  "memoryId": "12345abcdef6789012345abc",
  "url": "https://example.com/article",
  "title": "Understanding Vector Embeddings",
  "content": "Good reference for understanding embeddings",
  "tags": ["ai", "embeddings"],
  "source": {
    "sourceTitle": "Tech Blog",
    "sourceDescription": "Articles about AI and ML",
    "authorName": "Jane Doe",
    "publicationDate": "2026-09-01"
  },
  "extractedContent": "Vector embeddings are numerical representations..."
}
"""

RESPONSE_JSON = """
{
  "memoryId": "memory-id",
  "chunks": [
    {
      "chunkId": "unique-chunk-id",
      "chunkIndex": 0,
      "content": "Chunk content",
      "embedding": [0.0123, -0.0456]
    }
  ]
}
"""


# --------------------------------------------------------------------------
# Requests
# --------------------------------------------------------------------------
def test_the_documented_note_validates() -> None:
    request = ProcessMemoryRequestAdapter.validate_json(NOTE_JSON)

    assert isinstance(request, NoteProcessRequest)
    assert request.memory_id == "67890abcdef1234567890abc"
    assert request.tags == ["database", "performance"]


def test_the_documented_link_validates() -> None:
    request = ProcessMemoryRequestAdapter.validate_json(LINK_JSON)

    assert isinstance(request, LinkProcessRequest)
    assert request.source.source_title == "Tech Blog"
    assert request.extracted_content == "Vector embeddings are numerical representations..."


def test_note_public_field_names_are_exact() -> None:
    schema = NoteProcessRequest.model_json_schema()

    assert set(schema["properties"]) == {"type", "memoryId", "title", "content", "tags"}
    assert set(schema["required"]) == {"type", "memoryId", "title", "content"}


def test_link_public_field_names_are_exact() -> None:
    schema = LinkProcessRequest.model_json_schema()

    assert set(schema["properties"]) == {
        "type",
        "memoryId",
        "url",
        "title",
        "content",
        "tags",
        "source",
        "extractedContent",
    }
    assert set(schema["required"]) == {"type", "memoryId", "url", "title", "source"}


def test_source_public_field_names_are_exact_and_all_required() -> None:
    schema = LinkSource.model_json_schema()
    names = {"sourceTitle", "sourceDescription", "authorName", "publicationDate"}

    assert set(schema["properties"]) == names
    assert set(schema["required"]) == names


def test_requests_round_trip_to_the_documented_json() -> None:
    for document in (NOTE_JSON, LINK_JSON):
        request = ProcessMemoryRequestAdapter.validate_json(document)

        assert json.loads(request.model_dump_json()) == json.loads(document)


def test_no_request_field_is_named_why_saved() -> None:
    """Contract section 8.5: permanently removed."""
    for model in (NoteProcessRequest, LinkProcessRequest, LinkSource):
        assert "whySaved" not in model.model_json_schema()["properties"]
        assert "why_saved" not in model.model_fields


def test_process_requests_carry_no_user_identity() -> None:
    for model in (NoteProcessRequest, LinkProcessRequest):
        assert "user_id" not in model.model_fields
        assert "userId" not in model.model_json_schema()["properties"]


# --------------------------------------------------------------------------
# Response
# --------------------------------------------------------------------------
def test_the_documented_response_validates() -> None:
    response = ProcessMemoryResponse.model_validate_json(RESPONSE_JSON)

    assert response.memory_id == "memory-id"
    assert response.chunks[0].chunk_id == "unique-chunk-id"
    assert response.chunks[0].chunk_index == 0
    assert response.chunks[0].embedding == [0.0123, -0.0456]


def test_the_response_round_trips_to_the_documented_json() -> None:
    response = ProcessMemoryResponse.model_validate_json(RESPONSE_JSON)

    assert json.loads(response.model_dump_json()) == json.loads(RESPONSE_JSON)


def test_the_response_serializes_with_exact_public_field_names() -> None:
    response = ProcessMemoryResponse.model_validate_json(RESPONSE_JSON)
    payload = json.loads(response.model_dump_json())

    assert set(payload) == {"memoryId", "chunks"}
    assert set(payload["chunks"][0]) == {"chunkId", "chunkIndex", "content", "embedding"}


def test_chunk_public_field_names_are_exact() -> None:
    schema = ProcessedChunk.model_json_schema()

    assert set(schema["properties"]) == {"chunkId", "chunkIndex", "content", "embedding"}
    assert set(schema["required"]) == {"chunkId", "chunkIndex", "content", "embedding"}


def test_the_response_never_emits_snake_case_keys() -> None:
    serialized = ProcessMemoryResponse.model_validate_json(RESPONSE_JSON).model_dump_json()

    for snake in ("chunk_id", "chunk_index", "memory_id"):
        assert snake not in serialized


def test_the_response_carries_the_complete_chunk_set_and_nothing_legacy() -> None:
    """Reprocessing returns every chunk, never a diff (contract section 13).

    The schema shape supports that: ``chunks`` is the full list and there is
    no field for a partial or delta response. ``intent`` and top-level
    ``content`` from the pre-v1.9 response are gone.
    """
    assert set(ProcessMemoryResponse.model_fields) == {"memory_id", "chunks"}
    for absent in ("intent", "content", "changedChunks", "removedChunkIds", "delta", "partial"):
        assert absent not in ProcessMemoryResponse.model_fields
