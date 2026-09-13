"""Link request rules: the ``link`` shape and its ``source`` object.

Contract v1.9, sections 8.2-8.5. Extraction is Backend-owned: the AI Service
receives already-extracted page text and never fetches anything.
"""

import pytest
from pydantic import ValidationError

from memovo_ai.schemas import LinkProcessRequest, LinkSource, ProcessMemoryRequestAdapter

pytestmark = pytest.mark.unit

#: Contract section 8.3, verbatim.
SOURCE: dict[str, object] = {
    "sourceTitle": "Tech Blog",
    "sourceDescription": "Articles about AI and ML",
    "authorName": "Jane Doe",
    "publicationDate": "2026-09-01",
}

#: Contract section 8.2, verbatim.
LINK: dict[str, object] = {
    "type": "link",
    "memoryId": "12345abcdef6789012345abc",
    "url": "https://example.com/article",
    "title": "Understanding Vector Embeddings",
    "content": "Good reference for understanding embeddings",
    "tags": ["ai", "embeddings"],
    "source": SOURCE,
    "extractedContent": "Vector embeddings are numerical representations...",
}

SOURCE_FIELDS = ("sourceTitle", "sourceDescription", "authorName", "publicationDate")


def validate(payload: dict[str, object]) -> LinkProcessRequest:
    request = ProcessMemoryRequestAdapter.validate_python(payload)
    assert isinstance(request, LinkProcessRequest)
    return request


def rejected(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ProcessMemoryRequestAdapter.validate_python(payload)


def with_source(**overrides: object) -> dict[str, object]:
    return {**LINK, "source": {**SOURCE, **overrides}}


# --------------------------------------------------------------------------
# The documented Link
# --------------------------------------------------------------------------
def test_the_documented_link_is_accepted() -> None:
    request = validate(LINK)

    assert request.memory_type == "link"
    assert request.url == "https://example.com/article"
    assert request.content == "Good reference for understanding embeddings"
    assert request.extracted_content == "Vector embeddings are numerical representations..."
    assert request.source.source_title == "Tech Blog"
    assert request.source.author_name == "Jane Doe"
    assert request.source.publication_date == "2026-09-01"


@pytest.mark.parametrize("field", ["type", "memoryId", "url", "title", "source"])
def test_a_missing_required_field_is_rejected(field: str) -> None:
    rejected({key: value for key, value in LINK.items() if key != field})


@pytest.mark.parametrize("field", ["memoryId", "url", "title", "source"])
def test_a_null_required_field_is_rejected(field: str) -> None:
    rejected({**LINK, field: None})


# --------------------------------------------------------------------------
# url: required, http/https, at most 2048 characters, never dereferenced
# --------------------------------------------------------------------------
def test_a_2048_character_url_is_accepted() -> None:
    url = "https://example.com/" + "a" * (2048 - len("https://example.com/"))

    assert len(validate({**LINK, "url": url}).url) == 2048


def test_a_2049_character_url_is_rejected() -> None:
    rejected({**LINK, "url": "https://example.com/" + "a" * (2049 - len("https://example.com/"))})


def test_an_empty_url_is_rejected() -> None:
    rejected({**LINK, "url": ""})


@pytest.mark.parametrize(
    "url",
    ["ftp://example.com/file", "javascript:alert(1)", "example.com/article", "mailto:a@b.c", "//x"],
)
def test_a_non_web_scheme_is_rejected(url: str) -> None:
    """The Backend's own URL validation admits http/https only."""
    rejected({**LINK, "url": url})


@pytest.mark.parametrize("url", ["HTTP://EXAMPLE.COM/A", "Https://example.com/a"])
def test_the_scheme_check_is_case_insensitive(url: str) -> None:
    assert validate({**LINK, "url": url}).url == url


def test_an_unreachable_url_is_still_just_a_string() -> None:
    """Nothing is resolved or opened; validation is a prefix check."""
    assert validate({**LINK, "url": "https://127.0.0.1:9/nope"}).url == "https://127.0.0.1:9/nope"


# --------------------------------------------------------------------------
# title: 1-500 characters
# --------------------------------------------------------------------------
def test_an_empty_link_title_is_rejected() -> None:
    rejected({**LINK, "title": ""})


def test_a_five_hundred_character_title_is_accepted() -> None:
    assert len(validate({**LINK, "title": "t" * 500}).title) == 500


def test_a_five_hundred_and_one_character_title_is_rejected() -> None:
    rejected({**LINK, "title": "t" * 501})


# --------------------------------------------------------------------------
# content: optional, nullable, empty allowed, at most 1000 characters
# --------------------------------------------------------------------------
def test_absent_content_is_none() -> None:
    assert validate({key: value for key, value in LINK.items() if key != "content"}).content is None


def test_null_content_is_accepted() -> None:
    assert validate({**LINK, "content": None}).content is None


def test_empty_content_is_accepted() -> None:
    assert validate({**LINK, "content": ""}).content == ""


def test_one_thousand_character_content_is_accepted() -> None:
    assert len(validate({**LINK, "content": "c" * 1000}).content or "") == 1000


def test_one_thousand_and_one_character_content_is_rejected() -> None:
    rejected({**LINK, "content": "c" * 1001})


# --------------------------------------------------------------------------
# extractedContent: optional, nullable, empty allowed, no AI-side limit
# --------------------------------------------------------------------------
def test_absent_extracted_content_is_none() -> None:
    payload = {key: value for key, value in LINK.items() if key != "extractedContent"}

    assert validate(payload).extracted_content is None


@pytest.mark.parametrize("value", [None, ""])
def test_null_and_empty_extracted_content_are_accepted(value: str | None) -> None:
    assert validate({**LINK, "extractedContent": value}).extracted_content == value


def test_extracted_content_has_no_invented_character_limit() -> None:
    """Section 8.4: the Backend controls extraction and payload size."""
    text = "word " * 100_000

    assert len(validate({**LINK, "extractedContent": text}).extracted_content or "") == len(text)


def test_a_non_string_extracted_content_is_rejected() -> None:
    rejected({**LINK, "extractedContent": ["page"]})


# --------------------------------------------------------------------------
# tags: the same rule as a Note
# --------------------------------------------------------------------------
def test_null_tags_are_accepted_on_a_link() -> None:
    assert validate({**LINK, "tags": None}).tags is None


def test_twenty_one_tags_are_rejected_on_a_link() -> None:
    rejected({**LINK, "tags": [f"t{n}" for n in range(21)]})


# --------------------------------------------------------------------------
# source: required object with exactly four required, nullable strings
# --------------------------------------------------------------------------
def test_an_empty_source_object_is_rejected() -> None:
    """Every key is required; ``{}`` is not a valid source (section 8.3)."""
    rejected({**LINK, "source": {}})


@pytest.mark.parametrize("field", SOURCE_FIELDS)
def test_each_source_field_may_be_null(field: str) -> None:
    assert validate(with_source(**{field: None})).source is not None


def test_an_all_null_source_is_accepted() -> None:
    request = validate({**LINK, "source": dict.fromkeys(SOURCE_FIELDS)})

    assert request.source == LinkSource(
        sourceTitle=None, sourceDescription=None, authorName=None, publicationDate=None
    )


@pytest.mark.parametrize("field", SOURCE_FIELDS)
def test_each_source_field_is_required(field: str) -> None:
    source = {key: value for key, value in SOURCE.items() if key != field}

    rejected({**LINK, "source": source})


@pytest.mark.parametrize(
    "field",
    [
        # Backend-only metadata (section 8.3).
        "platform",
        "contentType",
        "thumbnailUrl",
        "canonicalUrl",
        # The pre-v1.9 source shape.
        "siteName",
        "favicon",
        "ogImage",
        "publishedAt",
    ],
)
def test_an_undocumented_source_field_is_rejected(field: str) -> None:
    rejected(with_source(**{field: "x"}))


@pytest.mark.parametrize("field", SOURCE_FIELDS)
def test_a_non_string_source_value_is_rejected(field: str) -> None:
    rejected(with_source(**{field: 123}))


@pytest.mark.parametrize("value", ["a string", 123, [], True])
def test_a_non_object_source_is_rejected(value: object) -> None:
    rejected({**LINK, "source": value})


def test_source_snake_case_input_is_rejected() -> None:
    rejected({**LINK, "source": {**SOURCE, "source_title": "x"}})


def test_publication_date_is_any_string() -> None:
    """ISO 8601 *or another string representation*: no format is enforced."""
    assert validate(with_source(publicationDate="last Tuesday")).source.publication_date == (
        "last Tuesday"
    )


def test_source_uses_the_contract_field_names() -> None:
    schema = LinkSource.model_json_schema()

    assert set(schema["properties"]) == set(SOURCE_FIELDS)
    assert set(schema["required"]) == set(SOURCE_FIELDS)


# --------------------------------------------------------------------------
# Nothing legacy is accepted
# --------------------------------------------------------------------------
@pytest.mark.parametrize("field", ["whySaved", "description", "about", "userId", "jobId"])
def test_legacy_and_out_of_scope_fields_are_rejected(field: str) -> None:
    rejected({**LINK, field: "x"})


def test_a_link_shaped_payload_typed_as_a_note_is_rejected() -> None:
    """The discriminator picks the shape; the Link fields are then extras."""
    rejected({**LINK, "type": "note"})
