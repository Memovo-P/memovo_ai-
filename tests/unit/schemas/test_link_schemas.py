"""Link request fields: ``about`` and the ``source`` object.

Phase 17. Extraction is Backend-owned: the AI Service receives already
extracted page content and never fetches anything.
"""

import pytest
from pydantic import ValidationError

from memovo_ai.schemas import LinkSource, ProcessMemoryRequest

pytestmark = pytest.mark.unit

NOTE = {
    "memoryId": "memory_456",
    "title": "MongoDB Vector Search",
    "description": "How Atlas Vector Search works...",
    "whySaved": "Useful for the memory project",
    "tags": ["mongodb", "vector-search", "ai"],
}

LINK = {
    "memoryId": "memory_link_1",
    "title": "Atlas Vector Search docs",
    "description": "Atlas Vector Search indexes embeddings for similarity queries.",
    "whySaved": "",
    "tags": ["mongodb", "docs"],
    "about": "Read this before the vector index migration",
    "source": {
        "siteName": "MongoDB Docs",
        "favicon": "https://cdn.example/favicon.ico",
        "ogImage": "https://cdn.example/og.png",
        "publishedAt": "2024-01-15",
    },
}


# --------------------------------------------------------------------------
# The Note contract is untouched
# --------------------------------------------------------------------------
def test_a_note_request_is_unchanged() -> None:
    request = ProcessMemoryRequest.model_validate(NOTE)

    assert request.about is None
    assert request.source is None
    assert request.is_link is False


@pytest.mark.parametrize("field", ["memoryId", "title", "description", "whySaved", "tags"])
def test_note_fields_are_still_required(field: str) -> None:
    """Adding Link support must not make any existing field optional."""
    payload = {key: value for key, value in NOTE.items() if key != field}

    with pytest.raises(ValidationError):
        ProcessMemoryRequest.model_validate(payload)


@pytest.mark.parametrize("field", ["userId", "embeddingVersion", "jobId", "createdAt"])
def test_still_out_of_scope_fields_stay_rejected(field: str) -> None:
    with pytest.raises(ValidationError):
        ProcessMemoryRequest.model_validate({**NOTE, field: "x"})


# --------------------------------------------------------------------------
# A Link request is accepted
# --------------------------------------------------------------------------
def test_a_full_link_request_is_accepted() -> None:
    request = ProcessMemoryRequest.model_validate(LINK)

    assert request.about == "Read this before the vector index migration"
    assert request.source is not None
    assert request.source.site_name == "MongoDB Docs"
    assert request.source.published_at == "2024-01-15"
    assert request.is_link is True


def test_the_source_object_is_the_link_discriminator() -> None:
    """The contract requires `source` on every Link request, so its presence
    identifies one. No `type` field is introduced."""
    assert ProcessMemoryRequest.model_validate(LINK).is_link is True
    assert ProcessMemoryRequest.model_validate(NOTE).is_link is False


# --------------------------------------------------------------------------
# `about` is optional (doc 01, decision 28)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("about", [None, "", "   ", "some context"])
def test_about_accepts_absent_null_empty_or_present(about: str | None) -> None:
    request = ProcessMemoryRequest.model_validate({**LINK, "about": about})

    assert request.about == about


def test_a_link_without_about_is_accepted() -> None:
    payload = {key: value for key, value in LINK.items() if key != "about"}

    assert ProcessMemoryRequest.model_validate(payload).about is None


def test_a_note_may_also_carry_about() -> None:
    """Nothing in the contract restricts `about` to Links."""
    assert ProcessMemoryRequest.model_validate({**NOTE, "about": "context"}).is_link is False


# --------------------------------------------------------------------------
# `source` internals are optional / nullable (doc 01, decision 29)
# --------------------------------------------------------------------------
def test_an_empty_source_object_is_accepted() -> None:
    """The object must be present; its contents may be entirely unknown."""
    request = ProcessMemoryRequest.model_validate({**LINK, "source": {}})

    assert request.is_link is True
    assert request.source == LinkSource()


@pytest.mark.parametrize("field", ["siteName", "favicon", "ogImage", "publishedAt"])
def test_each_source_field_may_be_null(field: str) -> None:
    source = {**LINK["source"], field: None}  # type: ignore[dict-item]

    assert ProcessMemoryRequest.model_validate({**LINK, "source": source}).is_link is True


@pytest.mark.parametrize("field", ["siteName", "favicon", "ogImage", "publishedAt"])
def test_each_source_field_may_be_absent(field: str) -> None:
    source = {k: v for k, v in LINK["source"].items() if k != field}  # type: ignore[union-attr]

    assert ProcessMemoryRequest.model_validate({**LINK, "source": source}).is_link is True


def test_an_all_null_source_is_accepted_without_error() -> None:
    """Graceful handling: no field available must not be an error."""
    source = dict.fromkeys(("siteName", "favicon", "ogImage", "publishedAt"))

    request = ProcessMemoryRequest.model_validate({**LINK, "source": source})

    assert request.source is not None
    assert request.source.site_name is None


def test_source_uses_the_contract_field_names() -> None:
    schema = LinkSource.model_json_schema()

    assert set(schema["properties"]) == {"siteName", "favicon", "ogImage", "publishedAt"}
    assert schema.get("required", []) == []


def test_source_snake_case_input_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ProcessMemoryRequest.model_validate({**LINK, "source": {"site_name": "x"}})


def test_an_undocumented_source_field_is_rejected() -> None:
    """Only the four documented fields exist; a fifth is a contract change."""
    with pytest.raises(ValidationError):
        ProcessMemoryRequest.model_validate({**LINK, "source": {"author": "someone"}})


@pytest.mark.parametrize("value", ["a string", 123, [], True])
def test_a_non_object_source_is_rejected(value: object) -> None:
    with pytest.raises(ValidationError):
        ProcessMemoryRequest.model_validate({**LINK, "source": value})


@pytest.mark.parametrize("field", ["siteName", "favicon", "ogImage", "publishedAt"])
def test_a_non_string_source_value_is_rejected(field: str) -> None:
    with pytest.raises(ValidationError):
        ProcessMemoryRequest.model_validate({**LINK, "source": {field: 123}})


def test_a_null_source_means_not_a_link() -> None:
    assert ProcessMemoryRequest.model_validate({**NOTE, "source": None}).is_link is False


# --------------------------------------------------------------------------
# No URL is ever fetched
# --------------------------------------------------------------------------
def test_source_urls_are_treated_as_opaque_strings() -> None:
    """The AI stores what it is given; it never dereferences a URL."""
    request = ProcessMemoryRequest.model_validate(
        {**LINK, "source": {"favicon": "not-a-url-at-all", "ogImage": "::::"}}
    )

    assert request.source is not None
    assert request.source.favicon == "not-a-url-at-all"
