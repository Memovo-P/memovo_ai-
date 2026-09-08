"""Validation behaviour for the ``/ai/memories/process`` schemas."""

import pytest
from pydantic import ValidationError

from memovo_ai.schemas import ProcessedChunk, ProcessMemoryRequest, ProcessMemoryResponse

pytestmark = pytest.mark.unit

VALID_REQUEST = {
    "memoryId": "memory_456",
    "title": "MongoDB Vector Search",
    "description": "How Atlas Vector Search works...",
    "whySaved": "Useful for the memory project",
    "tags": ["mongodb", "vector-search", "ai"],
}

VALID_CHUNK = {
    "chunkId": "chunk_001",
    "chunkIndex": 0,
    "content": "MongoDB Vector Search...",
    "embedding": [0.012, -0.034, 0.087],
}


# --------------------------------------------------------------------------
# ProcessMemoryRequest
# --------------------------------------------------------------------------
def test_valid_request_is_accepted() -> None:
    request = ProcessMemoryRequest.model_validate(VALID_REQUEST)

    assert request.memory_id == "memory_456"
    assert request.why_saved == "Useful for the memory project"
    assert request.tags == ["mongodb", "vector-search", "ai"]


@pytest.mark.parametrize("field", ["memoryId", "title", "description", "whySaved", "tags"])
def test_missing_required_field_is_rejected(field: str) -> None:
    payload = {k: v for k, v in VALID_REQUEST.items() if k != field}

    with pytest.raises(ValidationError):
        ProcessMemoryRequest.model_validate(payload)


def test_extra_field_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ProcessMemoryRequest.model_validate({**VALID_REQUEST, "unexpected": "value"})


@pytest.mark.parametrize(
    "field",
    ["userId", "embeddingVersion", "createdAt", "updatedAt", "jobId", "source", "about"],
)
def test_out_of_scope_fields_are_rejected(field: str) -> None:
    """Sprint 1 Note requests carry none of these. Link fields are Phase 17."""
    with pytest.raises(ValidationError):
        ProcessMemoryRequest.model_validate({**VALID_REQUEST, field: "x"})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("memoryId", 456),
        ("title", 1),
        ("description", None),
        ("whySaved", ["a"]),
        ("tags", "mongodb"),
        ("tags", [1, 2]),
    ],
)
def test_wrong_type_is_rejected(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        ProcessMemoryRequest.model_validate({**VALID_REQUEST, field: value})


def test_snake_case_input_is_rejected() -> None:
    """The public contract is camelCase; snake_case must not be silently accepted."""
    payload = {
        "memory_id": "memory_456",
        "title": "t",
        "description": "d",
        "why_saved": "w",
        "tags": [],
    }

    with pytest.raises(ValidationError):
        ProcessMemoryRequest.model_validate(payload)


def test_empty_tags_list_is_accepted() -> None:
    """`tags` is required but may be empty (doc 05 lists empty tags as an input case)."""
    request = ProcessMemoryRequest.model_validate({**VALID_REQUEST, "tags": []})

    assert request.tags == []


def test_blank_strings_are_accepted() -> None:
    """No source document defines blank-string rejection, so the schema stays permissive."""
    request = ProcessMemoryRequest.model_validate(
        {**VALID_REQUEST, "description": "", "whySaved": ""}
    )

    assert request.description == ""
    assert request.why_saved == ""


@pytest.mark.parametrize(
    "title",
    ["بحث المتجهات في مونجو", "MongoDB بحث", "Ünïcodé ✅ 検索"],
)
def test_unicode_and_arabic_content_is_accepted(title: str) -> None:
    request = ProcessMemoryRequest.model_validate({**VALID_REQUEST, "title": title})

    assert request.title == title


# --------------------------------------------------------------------------
# ProcessedChunk
# --------------------------------------------------------------------------
def test_valid_chunk_is_accepted() -> None:
    chunk = ProcessedChunk.model_validate(VALID_CHUNK)

    assert chunk.chunk_id == "chunk_001"
    assert chunk.chunk_index == 0
    assert chunk.embedding == [0.012, -0.034, 0.087]


def test_negative_chunk_index_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ProcessedChunk.model_validate({**VALID_CHUNK, "chunkIndex": -1})


@pytest.mark.parametrize("value", ["0", 1.0, True, None])
def test_non_integer_chunk_index_is_rejected(value: object) -> None:
    """Strict mode: no coercion from string, float, or bool."""
    with pytest.raises(ValidationError):
        ProcessedChunk.model_validate({**VALID_CHUNK, "chunkIndex": value})


def test_extra_chunk_field_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ProcessedChunk.model_validate({**VALID_CHUNK, "score": 0.9})


def test_integer_embedding_values_are_accepted() -> None:
    """Widening int to float is legitimate; an embedding value of 0 is valid."""
    chunk = ProcessedChunk.model_validate({**VALID_CHUNK, "embedding": [0, 1, -1]})

    assert chunk.embedding == [0.0, 1.0, -1.0]


@pytest.mark.parametrize("embedding", [["0.1"], [None], "0.1", [[0.1]]])
def test_non_numeric_embedding_is_rejected(embedding: object) -> None:
    with pytest.raises(ValidationError):
        ProcessedChunk.model_validate({**VALID_CHUNK, "embedding": embedding})


def test_embedding_dimension_is_not_enforced_by_the_transport_schema() -> None:
    """Dimension 1024 is validated at the embedding provider boundary (Phase 05).

    Enforcing it here would duplicate that check in the wrong layer and would
    make the transport schema depend on a model decision.
    """
    chunk = ProcessedChunk.model_validate({**VALID_CHUNK, "embedding": [0.1, 0.2]})

    assert len(chunk.embedding) == 2


# --------------------------------------------------------------------------
# ProcessMemoryResponse
# --------------------------------------------------------------------------
def test_valid_response_is_accepted() -> None:
    response = ProcessMemoryResponse.model_validate(
        {
            "intent": "save_memory",
            "content": "How MongoDB Vector Search works...",
            "chunks": [VALID_CHUNK],
        }
    )

    assert response.intent == "save_memory"
    assert len(response.chunks) == 1


def test_intent_defaults_to_save_memory() -> None:
    response = ProcessMemoryResponse.model_validate({"content": "c", "chunks": []})

    assert response.intent == "save_memory"


@pytest.mark.parametrize("intent", ["search_memory", "create_note", "general_chat", "", None, 1])
def test_unsupported_intent_is_rejected(intent: object) -> None:
    with pytest.raises(ValidationError):
        ProcessMemoryResponse.model_validate({"intent": intent, "content": "c", "chunks": []})


def test_extra_response_field_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ProcessMemoryResponse.model_validate(
            {"content": "c", "chunks": [], "embeddingVersion": "v1"}
        )
