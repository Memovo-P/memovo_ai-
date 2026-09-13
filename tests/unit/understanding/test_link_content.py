"""Canonical content for Links: the ``Source`` and ``Extracted Content`` sections.

Contract v1.9 section 8.4: a Link embeds title, user content, tags, source
metadata and extracted content. A Note composes to Title/Content/Tags only,
which is what keeps Note chunk IDs independent of Link support.
"""

import pytest

from memovo_ai.understanding import compose_canonical_content

pytestmark = pytest.mark.unit

NOTE = {
    "title": "MongoDB Vector Search",
    "content": "How Atlas Vector Search works...",
    "tags": ["mongodb", "vector-search", "ai"],
}

NOTE_TEXT = """Title:
MongoDB Vector Search

Content:
How Atlas Vector Search works...

Tags:
mongodb, vector-search, ai"""


# --------------------------------------------------------------------------
# Notes are unaffected by the Link sections
# --------------------------------------------------------------------------
def test_a_note_composes_to_title_content_tags() -> None:
    assert compose_canonical_content(**NOTE) == NOTE_TEXT


def test_explicitly_empty_link_fields_change_nothing() -> None:
    assert compose_canonical_content(**NOTE, source=(), extracted_content=None) == NOTE_TEXT
    assert compose_canonical_content(**NOTE, source=[], extracted_content="") == NOTE_TEXT


@pytest.mark.parametrize("source", [(), [], ["", "  "]])
def test_a_blank_source_omits_the_section(source: list[str]) -> None:
    composed = compose_canonical_content(**NOTE, source=source)

    assert "Source:" not in composed
    assert composed == NOTE_TEXT


@pytest.mark.parametrize("extracted", [None, "", "   ", "\n\t "])
def test_blank_extracted_content_omits_the_section(extracted: str | None) -> None:
    composed = compose_canonical_content(**NOTE, extracted_content=extracted)

    assert "Extracted Content:" not in composed
    assert composed == NOTE_TEXT


# --------------------------------------------------------------------------
# Link sections and their order
# --------------------------------------------------------------------------
def test_source_appears_after_content() -> None:
    composed = compose_canonical_content(**NOTE, source=["Tech Blog"])

    assert "Source:\nTech Blog" in composed
    assert composed.index("Content:") < composed.index("Source:")


def test_extracted_content_appears_after_source_and_before_tags() -> None:
    composed = compose_canonical_content(**NOTE, source=["Tech Blog"], extracted_content="Body.")

    source_at = composed.index("Source:")
    extracted_at = composed.index("Extracted Content:")

    assert source_at < extracted_at < composed.index("Tags:")


def test_source_parts_are_joined_on_one_line() -> None:
    composed = compose_canonical_content(**NOTE, source=["Tech Blog", "Jane Doe", "2026-09-01"])

    assert "Source:\nTech Blog, Jane Doe, 2026-09-01" in composed


def test_blank_source_parts_are_dropped() -> None:
    composed = compose_canonical_content(**NOTE, source=["Tech Blog", "", "  ", "2026"])

    assert "Source:\nTech Blog, 2026" in composed


def test_source_part_order_is_preserved() -> None:
    composed = compose_canonical_content(**NOTE, source=["zebra", "apple"])

    assert "Source:\nzebra, apple" in composed


def test_the_full_link_section_order_is_fixed() -> None:
    composed = compose_canonical_content(
        title="Understanding Vector Embeddings",
        content="Good reference for understanding embeddings",
        tags=["ai", "embeddings"],
        source=["Tech Blog", "Articles about AI and ML", "Jane Doe", "2026-09-01"],
        extracted_content="Vector embeddings are numerical representations...",
    )

    assert composed == (
        "Title:\nUnderstanding Vector Embeddings\n\n"
        "Content:\nGood reference for understanding embeddings\n\n"
        "Source:\nTech Blog, Articles about AI and ML, Jane Doe, 2026-09-01\n\n"
        "Extracted Content:\nVector embeddings are numerical representations...\n\n"
        "Tags:\nai, embeddings"
    )


def test_user_content_and_extracted_content_stay_distinct() -> None:
    """Section 8.5: user context is not the page text and vice versa."""
    composed = compose_canonical_content(
        title="T", content="my note", tags=[], extracted_content="page body"
    )

    assert "Content:\nmy note" in composed
    assert "Extracted Content:\npage body" in composed


def test_extracted_content_is_normalized_like_any_text() -> None:
    composed = compose_canonical_content(
        title="T", content=None, tags=None, extracted_content="  Line one.\r\n\r\nLine two.  \n"
    )

    assert composed == "Title:\nT\n\nExtracted Content:\nLine one.\n\nLine two."


# --------------------------------------------------------------------------
# Determinism and graceful degradation
# --------------------------------------------------------------------------
def test_link_composition_is_deterministic() -> None:
    kwargs = {**NOTE, "source": ["Tech Blog", "2026-09-01"], "extracted_content": "Body."}

    assert len({compose_canonical_content(**kwargs) for _ in range(25)}) == 1  # type: ignore[arg-type]


def test_a_link_with_only_a_title_still_composes() -> None:
    composed = compose_canonical_content(
        title="Just a link title", content=None, tags=None, source=(), extracted_content=None
    )

    assert composed == "Title:\nJust a link title"


def test_a_link_with_no_extractable_metadata_still_composes() -> None:
    """Every source field null: graceful, not an error, and no literal null."""
    composed = compose_canonical_content(
        title="Some page", content="", tags=[], source=[], extracted_content="Extracted body."
    )

    assert composed == "Title:\nSome page\n\nExtracted Content:\nExtracted body."
    assert "None" not in composed


def test_arabic_link_fields_are_preserved() -> None:
    composed = compose_canonical_content(
        title="مقال عن المتجهات",
        content="اقرأه قبل الترحيل",
        tags=["ذكاء-اصطناعي"],
        source=["مدونة مونجو"],
        extracted_content="محتوى الصفحة المستخرج",
    )

    assert "Content:\nاقرأه قبل الترحيل" in composed
    assert "Source:\nمدونة مونجو" in composed
    assert "Extracted Content:\nمحتوى الصفحة المستخرج" in composed


def test_changing_source_changes_the_content() -> None:
    first = compose_canonical_content(**NOTE, source=["Tech Blog"])
    second = compose_canonical_content(**NOTE, source=["Other Site"])

    assert first != second


def test_changing_extracted_content_changes_the_content() -> None:
    first = compose_canonical_content(**NOTE, extracted_content="one")
    second = compose_canonical_content(**NOTE, extracted_content="two")

    assert first != second
