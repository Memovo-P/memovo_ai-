"""Canonical content for Links: the `About` and `Source` sections.

Phase 17. The decisive property is that a Note composes to exactly the string
it did before Link support existed, so no existing chunk ID moves.
"""

import pytest

from memovo_ai.understanding import compose_canonical_content

pytestmark = pytest.mark.unit

NOTE = {
    "title": "MongoDB Vector Search",
    "description": "How Atlas Vector Search works...",
    "why_saved": "Useful for the memory project",
    "tags": ["mongodb", "vector-search", "ai"],
}

#: Byte-for-byte what Phase 02 produced, before `about`/`source` existed.
NOTE_BEFORE_PHASE_17 = """Title:
MongoDB Vector Search

Content:
How Atlas Vector Search works...

Why Saved:
Useful for the memory project

Tags:
mongodb, vector-search, ai"""


# --------------------------------------------------------------------------
# Notes are unaffected -- this is what keeps chunk IDs stable
# --------------------------------------------------------------------------
def test_a_note_composes_exactly_as_before_phase_17() -> None:
    assert compose_canonical_content(**NOTE) == NOTE_BEFORE_PHASE_17


def test_explicitly_empty_link_fields_change_nothing() -> None:
    assert compose_canonical_content(**NOTE, about=None, source=()) == NOTE_BEFORE_PHASE_17
    assert compose_canonical_content(**NOTE, about="", source=[]) == NOTE_BEFORE_PHASE_17


@pytest.mark.parametrize("about", [None, "", "   ", "\n\t "])
def test_a_blank_about_omits_the_section(about: str | None) -> None:
    composed = compose_canonical_content(**NOTE, about=about)

    assert "About:" not in composed
    assert composed == NOTE_BEFORE_PHASE_17


@pytest.mark.parametrize("source", [(), [], ["", "  "]])
def test_a_blank_source_omits_the_section(source: list[str]) -> None:
    composed = compose_canonical_content(**NOTE, source=source)

    assert "Source:" not in composed
    assert composed == NOTE_BEFORE_PHASE_17


# --------------------------------------------------------------------------
# Link sections
# --------------------------------------------------------------------------
def test_about_appears_after_content() -> None:
    composed = compose_canonical_content(**NOTE, about="Read before the migration")

    assert "About:\nRead before the migration" in composed
    assert composed.index("Content:") < composed.index("About:")


def test_about_appears_before_why_saved() -> None:
    composed = compose_canonical_content(**NOTE, about="context")

    assert composed.index("About:") < composed.index("Why Saved:")


def test_source_appears_between_why_saved_and_tags() -> None:
    composed = compose_canonical_content(**NOTE, source=["MongoDB Docs"])

    assert composed.index("Why Saved:") < composed.index("Source:") < composed.index("Tags:")


def test_source_parts_are_joined_on_one_line() -> None:
    composed = compose_canonical_content(**NOTE, source=["MongoDB Docs", "2024-01-15"])

    assert "Source:\nMongoDB Docs, 2024-01-15" in composed


def test_blank_source_parts_are_dropped() -> None:
    composed = compose_canonical_content(**NOTE, source=["MongoDB Docs", "", "  ", "2024"])

    assert "Source:\nMongoDB Docs, 2024" in composed


def test_source_part_order_is_preserved() -> None:
    composed = compose_canonical_content(**NOTE, source=["zebra", "apple"])

    assert "Source:\nzebra, apple" in composed


def test_the_full_link_section_order_is_fixed() -> None:
    composed = compose_canonical_content(
        title="Atlas docs",
        description="Extracted page body.",
        why_saved="For the migration",
        tags=["mongodb"],
        about="My note about it",
        source=["MongoDB Docs", "2024-01-15"],
    )

    assert composed == (
        "Title:\nAtlas docs\n\n"
        "Content:\nExtracted page body.\n\n"
        "About:\nMy note about it\n\n"
        "Why Saved:\nFor the migration\n\n"
        "Source:\nMongoDB Docs, 2024-01-15\n\n"
        "Tags:\nmongodb"
    )


# --------------------------------------------------------------------------
# Determinism and graceful degradation
# --------------------------------------------------------------------------
def test_link_composition_is_deterministic() -> None:
    kwargs = {**NOTE, "about": "context", "source": ["MongoDB Docs", "2024-01-15"]}

    assert len({compose_canonical_content(**kwargs) for _ in range(25)}) == 1  # type: ignore[arg-type]


def test_a_link_with_only_a_title_still_composes() -> None:
    composed = compose_canonical_content(
        title="Just a link title", description="", why_saved="", tags=[], about=None, source=()
    )

    assert composed == "Title:\nJust a link title"


def test_a_link_with_no_extractable_metadata_still_composes() -> None:
    """Every source field null: graceful, not an error."""
    composed = compose_canonical_content(
        title="Some page",
        description="Extracted body.",
        why_saved="",
        tags=[],
        about=None,
        source=[],
    )

    assert composed == "Title:\nSome page\n\nContent:\nExtracted body."


def test_arabic_link_fields_are_preserved() -> None:
    composed = compose_canonical_content(
        title="مقال عن المتجهات",
        description="محتوى الصفحة المستخرج",
        why_saved="",
        tags=["ذكاء-اصطناعي"],
        about="اقرأه قبل الترحيل",
        source=["مدونة مونجو"],
    )

    assert "About:\nاقرأه قبل الترحيل" in composed
    assert "Source:\nمدونة مونجو" in composed


def test_changing_about_changes_the_content() -> None:
    """A different `about` must produce different chunks, hence new chunk IDs."""
    first = compose_canonical_content(**NOTE, about="one")
    second = compose_canonical_content(**NOTE, about="two")

    assert first != second


def test_changing_source_changes_the_content() -> None:
    first = compose_canonical_content(**NOTE, source=["MongoDB Docs"])
    second = compose_canonical_content(**NOTE, source=["Other Site"])

    assert first != second
