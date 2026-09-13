"""Validation behaviour for the ``/ai/memories/process`` schemas.

Every limit asserted here is copied from contract v1.9, section 8.1 (Note)
and section 9 (response). None is invented; where the contract sets no limit
-- ``memoryId`` above all -- none is asserted. Link-specific rules live in
``test_link_schemas.py``.
"""

import pytest
from pydantic import ValidationError

from memovo_ai.schemas import (
    LinkProcessRequest,
    NoteProcessRequest,
    ProcessedChunk,
    ProcessMemoryRequestAdapter,
    ProcessMemoryResponse,
)

pytestmark = pytest.mark.unit

#: Contract section 8.1, verbatim.
NOTE: dict[str, object] = {
    "type": "note",
    "memoryId": "67890abcdef1234567890abc",
    "title": "MongoDB Performance Tips",
    "content": "Use indexes for frequently queried fields...",
    "tags": ["database", "performance"],
}

VALID_CHUNK: dict[str, object] = {
    "chunkId": "chunk_001",
    "chunkIndex": 0,
    "content": "MongoDB Vector Search...",
    "embedding": [0.012, -0.034, 0.087],
}


def validate(payload: dict[str, object]) -> NoteProcessRequest | LinkProcessRequest:
    return ProcessMemoryRequestAdapter.validate_python(payload)


def rejected(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        validate(payload)


# --------------------------------------------------------------------------
# The documented Note
# --------------------------------------------------------------------------
def test_the_documented_note_is_accepted() -> None:
    request = validate(NOTE)

    assert isinstance(request, NoteProcessRequest)
    assert request.memory_type == "note"
    assert request.memory_id == "67890abcdef1234567890abc"
    assert request.title == "MongoDB Performance Tips"
    assert request.content == "Use indexes for frequently queried fields..."
    assert request.tags == ["database", "performance"]


@pytest.mark.parametrize("field", ["type", "memoryId", "title", "content"])
def test_a_missing_required_field_is_rejected(field: str) -> None:
    rejected({key: value for key, value in NOTE.items() if key != field})


@pytest.mark.parametrize("field", ["memoryId", "title", "content"])
def test_a_null_required_field_is_rejected(field: str) -> None:
    """Required and not nullable (section 8.1)."""
    rejected({**NOTE, field: None})


# --------------------------------------------------------------------------
# The discriminator
# --------------------------------------------------------------------------
@pytest.mark.parametrize("memory_type", ["Note", "NOTE", "memo", "", None, 1, True])
def test_an_unknown_type_is_rejected(memory_type: object) -> None:
    """Only the two literal values; no case folding and no coercion."""
    rejected({**NOTE, "type": memory_type})


def test_the_type_selects_the_shape() -> None:
    assert isinstance(validate(NOTE), NoteProcessRequest)
    assert not isinstance(validate(NOTE), LinkProcessRequest)


# --------------------------------------------------------------------------
# title: 0-200 characters
# --------------------------------------------------------------------------
def test_an_empty_title_is_accepted() -> None:
    assert validate({**NOTE, "title": ""}).title == ""


def test_a_two_hundred_character_title_is_accepted() -> None:
    assert len(validate({**NOTE, "title": "t" * 200}).title) == 200


def test_a_two_hundred_and_one_character_title_is_rejected() -> None:
    rejected({**NOTE, "title": "t" * 201})


# --------------------------------------------------------------------------
# content: 1-10,000 characters
# --------------------------------------------------------------------------
def test_empty_content_is_rejected() -> None:
    rejected({**NOTE, "content": ""})


def test_one_character_content_is_accepted() -> None:
    assert validate({**NOTE, "content": "x"}).content == "x"


def test_ten_thousand_character_content_is_accepted() -> None:
    assert len(validate({**NOTE, "content": "c" * 10_000}).content) == 10_000


def test_ten_thousand_and_one_character_content_is_rejected() -> None:
    rejected({**NOTE, "content": "c" * 10_001})


# --------------------------------------------------------------------------
# tags: optional, nullable, at most 20 of at most 50 characters
# --------------------------------------------------------------------------
def test_absent_tags_are_none() -> None:
    assert validate({key: value for key, value in NOTE.items() if key != "tags"}).tags is None


def test_null_tags_are_accepted() -> None:
    assert validate({**NOTE, "tags": None}).tags is None


def test_empty_tags_are_accepted() -> None:
    assert validate({**NOTE, "tags": []}).tags == []


def test_twenty_tags_are_accepted() -> None:
    assert len(validate({**NOTE, "tags": [f"t{n}" for n in range(20)]}).tags or []) == 20


def test_twenty_one_tags_are_rejected() -> None:
    rejected({**NOTE, "tags": [f"t{n}" for n in range(21)]})


def test_a_fifty_character_tag_is_accepted() -> None:
    assert validate({**NOTE, "tags": ["t" * 50]}).tags == ["t" * 50]


def test_a_fifty_one_character_tag_is_rejected() -> None:
    rejected({**NOTE, "tags": ["t" * 51]})


@pytest.mark.parametrize("tags", ["database", [1, 2], [None], [["nested"]]])
def test_tags_must_be_a_list_of_strings(tags: object) -> None:
    rejected({**NOTE, "tags": tags})


# --------------------------------------------------------------------------
# memoryId: non-empty, and otherwise unconstrained
# --------------------------------------------------------------------------
def test_an_empty_memory_id_is_rejected() -> None:
    rejected({**NOTE, "memoryId": ""})


def test_memory_id_has_no_invented_upper_bound() -> None:
    """The contract sets no length; the log field truncates instead."""
    assert len(validate({**NOTE, "memoryId": "m" * 5000}).memory_id) == 5000


def test_a_non_string_memory_id_is_rejected() -> None:
    rejected({**NOTE, "memoryId": 456})


# --------------------------------------------------------------------------
# Nothing else is accepted
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "field",
    ["whySaved", "description", "about", "userId", "jobId", "embeddingVersion", "createdAt"],
)
def test_legacy_and_out_of_scope_fields_are_rejected(field: str) -> None:
    """``whySaved`` is permanently removed (section 8.5); the rest never
    belonged to the request."""
    rejected({**NOTE, field: "x"})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("url", "https://example.com"),
        ("extractedContent", "page text"),
        (
            "source",
            {
                "sourceTitle": None,
                "sourceDescription": None,
                "authorName": None,
                "publicationDate": None,
            },
        ),
    ],
)
def test_link_only_fields_are_rejected_on_a_note(field: str, value: object) -> None:
    rejected({**NOTE, field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [("memoryId", 456), ("title", 1), ("content", ["a"]), ("tags", "database")],
)
def test_a_wrong_type_is_rejected(field: str, value: object) -> None:
    rejected({**NOTE, field: value})


def test_snake_case_input_is_rejected() -> None:
    """The public contract is camelCase; snake_case must not be silently accepted."""
    rejected({"type": "note", "memory_id": "m", "title": "t", "content": "c"})


@pytest.mark.parametrize("title", ["بحث المتجهات في مونجو", "MongoDB بحث", "Ünïcodé ✅ 検索"])
def test_unicode_and_arabic_content_is_accepted(title: str) -> None:
    assert validate({**NOTE, "title": title}).title == title


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
    """Dimension 1024 is validated at the embedding provider boundary.

    Enforcing it here would duplicate that check in the wrong layer and would
    make the transport schema depend on a model decision.
    """
    chunk = ProcessedChunk.model_validate({**VALID_CHUNK, "embedding": [0.1, 0.2]})

    assert len(chunk.embedding) == 2


# --------------------------------------------------------------------------
# ProcessMemoryResponse: memoryId + the complete chunk set (section 9)
# --------------------------------------------------------------------------
def test_valid_response_is_accepted() -> None:
    response = ProcessMemoryResponse.model_validate(
        {"memoryId": "memory-id", "chunks": [VALID_CHUNK]}
    )

    assert response.memory_id == "memory-id"
    assert len(response.chunks) == 1


def test_an_empty_chunk_set_is_a_valid_response() -> None:
    assert ProcessMemoryResponse.model_validate({"memoryId": "m", "chunks": []}).chunks == []


def test_a_missing_memory_id_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ProcessMemoryResponse.model_validate({"chunks": []})


def test_an_empty_memory_id_is_rejected_on_the_response() -> None:
    with pytest.raises(ValidationError):
        ProcessMemoryResponse.model_validate({"memoryId": "", "chunks": []})


@pytest.mark.parametrize(
    ("field", "value"), [("intent", "save_memory"), ("content", "c"), ("embeddingVersion", "v1")]
)
def test_legacy_and_extra_response_fields_are_rejected(field: str, value: object) -> None:
    """``intent`` and top-level ``content`` were the pre-v1.9 response."""
    with pytest.raises(ValidationError):
        ProcessMemoryResponse.model_validate({"memoryId": "m", "chunks": [], field: value})
