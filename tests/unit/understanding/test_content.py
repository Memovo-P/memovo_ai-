"""Behaviour of canonical content composition."""

import unicodedata

import pytest

from memovo_ai.understanding import (
    CANONICAL_CONTENT_VERSION,
    compose_canonical_content,
)

pytestmark = pytest.mark.unit

MEMORY = {
    "title": "MongoDB Vector Search",
    "description": "How Atlas Vector Search works...",
    "why_saved": "Useful for the memory project",
    "tags": ["mongodb", "vector-search", "ai"],
}

#: Copied from 03_IMPLEMENTATION_PLAN.md, Phase 02.
DOCUMENTED_REPRESENTATION = """Title:
MongoDB Vector Search

Content:
How Atlas Vector Search works...

Why Saved:
Useful for the memory project

Tags:
mongodb, vector-search, ai"""


# --------------------------------------------------------------------------
# Documented representation
# --------------------------------------------------------------------------
def test_matches_the_documented_representation() -> None:
    assert compose_canonical_content(**MEMORY) == DOCUMENTED_REPRESENTATION


def test_sections_appear_in_the_locked_order() -> None:
    composed = compose_canonical_content(**MEMORY)
    positions = [
        composed.index("Title:"),
        composed.index("Content:"),
        composed.index("Why Saved:"),
        composed.index("Tags:"),
    ]

    assert positions == sorted(positions)


def test_every_embedding_input_field_is_present() -> None:
    """Embedding input = Title + Content + WhySaved + Tags (doc 01, decisions 1-2)."""
    composed = compose_canonical_content(**MEMORY)

    assert "MongoDB Vector Search" in composed
    assert "How Atlas Vector Search works..." in composed
    assert "Useful for the memory project" in composed
    assert "mongodb, vector-search, ai" in composed


def test_output_has_no_trailing_newline() -> None:
    composed = compose_canonical_content(**MEMORY)

    assert composed == composed.strip()


def test_sections_are_separated_by_one_blank_line() -> None:
    assert compose_canonical_content(**MEMORY).count("\n\n") == 3


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------
def test_repeated_composition_is_identical() -> None:
    """Retries with unchanged input must produce identical chunk IDs downstream."""
    results = {compose_canonical_content(**MEMORY) for _ in range(25)}

    assert len(results) == 1


def test_changed_input_changes_the_output() -> None:
    baseline = compose_canonical_content(**MEMORY)

    assert compose_canonical_content(**{**MEMORY, "title": "Something else"}) != baseline
    assert compose_canonical_content(**{**MEMORY, "description": "Other"}) != baseline
    assert compose_canonical_content(**{**MEMORY, "why_saved": "Other"}) != baseline
    assert compose_canonical_content(**{**MEMORY, "tags": ["other"]}) != baseline


def test_composition_is_idempotent_under_renormalization() -> None:
    """Feeding normalized text back in must not change it further."""
    once = compose_canonical_content(title="  A  \r\n B  ", description="", why_saved="", tags=[])
    twice = compose_canonical_content(title=once, description="", why_saved="", tags=[])

    assert twice == f"Title:\n{once}"


# --------------------------------------------------------------------------
# Whitespace and line-ending normalization
# --------------------------------------------------------------------------
@pytest.mark.parametrize("newline", ["\r\n", "\r", "\n"])
def test_line_endings_are_normalized(newline: str) -> None:
    """The same Memory must not chunk differently by submitting platform."""
    description = f"First paragraph.{newline}{newline}Second paragraph."
    composed = compose_canonical_content(**{**MEMORY, "description": description})

    assert "\r" not in composed
    assert "First paragraph.\n\nSecond paragraph." in composed


def test_leading_and_trailing_whitespace_is_stripped() -> None:
    padded = compose_canonical_content(**{**MEMORY, "title": "   MongoDB Vector Search   \n\n"})

    assert padded == DOCUMENTED_REPRESENTATION


def test_whitespace_only_lines_become_truly_blank() -> None:
    """The chunker splits on paragraph boundaries, so "\\n   \\n" must equal "\\n\\n"."""
    spaced = compose_canonical_content(**{**MEMORY, "description": "One.\n   \nTwo."})
    clean = compose_canonical_content(**{**MEMORY, "description": "One.\n\nTwo."})

    assert spaced == clean


def test_interior_structure_is_preserved() -> None:
    """Paragraph breaks and indentation carry structure the chunker needs."""
    description = "# Heading\n\nParagraph one.\n\n    indented block\n\nParagraph two."
    composed = compose_canonical_content(**{**MEMORY, "description": description})

    assert description in composed


def test_interior_blank_lines_are_not_collapsed() -> None:
    composed = compose_canonical_content(**{**MEMORY, "description": "A\n\n\n\nB"})

    assert "A\n\n\n\nB" in composed


# --------------------------------------------------------------------------
# Empty and optional field handling
# --------------------------------------------------------------------------
def test_empty_why_saved_omits_the_section() -> None:
    composed = compose_canonical_content(**{**MEMORY, "why_saved": ""})

    assert "Why Saved:" not in composed
    assert "Title:" in composed
    assert "Tags:" in composed


def test_empty_tags_omit_the_section() -> None:
    composed = compose_canonical_content(**{**MEMORY, "tags": []})

    assert "Tags:" not in composed
    assert composed.endswith("Useful for the memory project")


def test_empty_description_omits_the_section() -> None:
    composed = compose_canonical_content(**{**MEMORY, "description": ""})

    assert "Content:" not in composed


def test_empty_title_omits_the_section() -> None:
    composed = compose_canonical_content(**{**MEMORY, "title": ""})

    assert "Title:" not in composed
    assert composed.startswith("Content:")


def test_whitespace_only_value_counts_as_empty() -> None:
    composed = compose_canonical_content(**{**MEMORY, "why_saved": "   \n\t  "})

    assert "Why Saved:" not in composed


def test_all_fields_empty_yields_empty_string() -> None:
    """Whether this is a request error belongs to the service and error contract."""
    assert compose_canonical_content(title="", description="", why_saved="", tags=[]) == ""


def test_omitted_section_leaves_no_double_separator() -> None:
    composed = compose_canonical_content(**{**MEMORY, "why_saved": ""})

    assert "\n\n\n" not in composed


def test_only_one_populated_field_produces_one_section() -> None:
    composed = compose_canonical_content(title="Solo", description="", why_saved="", tags=[])

    assert composed == "Title:\nSolo"


# --------------------------------------------------------------------------
# Tags
# --------------------------------------------------------------------------
def test_tags_are_joined_with_comma_space() -> None:
    composed = compose_canonical_content(**{**MEMORY, "tags": ["a", "b", "c"]})

    assert composed.endswith("Tags:\na, b, c")


def test_tag_order_is_preserved_not_sorted() -> None:
    """Reordering tags is Backend-visible; sorting here would be an invented rule."""
    composed = compose_canonical_content(**{**MEMORY, "tags": ["zebra", "apple", "mango"]})

    assert composed.endswith("Tags:\nzebra, apple, mango")


def test_individual_tags_are_stripped() -> None:
    composed = compose_canonical_content(**{**MEMORY, "tags": ["  mongodb  ", "\tai\n"]})

    assert composed.endswith("Tags:\nmongodb, ai")


def test_blank_tags_are_dropped() -> None:
    composed = compose_canonical_content(**{**MEMORY, "tags": ["mongodb", "", "   ", "ai"]})

    assert composed.endswith("Tags:\nmongodb, ai")


def test_tags_that_are_all_blank_omit_the_section() -> None:
    composed = compose_canonical_content(**{**MEMORY, "tags": ["", "  "]})

    assert "Tags:" not in composed


def test_duplicate_tags_are_preserved() -> None:
    """Deduplication is a product decision no source document makes."""
    composed = compose_canonical_content(**{**MEMORY, "tags": ["ai", "ai"]})

    assert composed.endswith("Tags:\nai, ai")


def test_a_tuple_of_tags_is_accepted() -> None:
    assert compose_canonical_content(
        **{**MEMORY, "tags": ("mongodb", "ai")}
    ) == compose_canonical_content(**{**MEMORY, "tags": ["mongodb", "ai"]})


# --------------------------------------------------------------------------
# Unicode / multilingual
# --------------------------------------------------------------------------
def test_arabic_content_is_preserved() -> None:
    composed = compose_canonical_content(
        title="بحث المتجهات",
        description="كيف يعمل البحث الدلالي في مونجو دي بي",
        why_saved="مفيد لمشروع الذاكرة",
        tags=["مونجو", "ذكاء-اصطناعي"],
    )

    assert "بحث المتجهات" in composed
    assert "كيف يعمل البحث الدلالي في مونجو دي بي" in composed
    assert "مفيد لمشروع الذاكرة" in composed
    assert composed.endswith("Tags:\nمونجو, ذكاء-اصطناعي")


def test_mixed_arabic_and_english_is_preserved() -> None:
    composed = compose_canonical_content(
        title="MongoDB بحث المتجهات",
        description="Atlas Vector Search يعمل عبر الفهارس",
        why_saved="مفيد for the memory project",
        tags=["mongodb", "مونجو"],
    )

    assert "MongoDB بحث المتجهات" in composed
    assert "مفيد for the memory project" in composed


@pytest.mark.parametrize(
    "text",
    ["Ünïcodé ✅", "日本語の検索", "emoji 🧠 memory", "math ∑ ≥ 0.75", "Ω≈ç√∫µ"],
)
def test_unicode_is_passed_through_unchanged(text: str) -> None:
    composed = compose_canonical_content(title=text, description="d", why_saved="w", tags=[])

    assert composed.startswith(f"Title:\n{text}")


def test_unicode_is_not_normalized() -> None:
    """Composed vs precomposed forms are left as supplied.

    Applying NFC here would silently rewrite user text; no source document
    asks for it. Flagged in the phase report as an open question.
    """
    precomposed = "é"
    decomposed = "é"
    assert precomposed != decomposed
    assert unicodedata.normalize("NFC", decomposed) == precomposed

    assert compose_canonical_content(
        title=precomposed, description="", why_saved="", tags=[]
    ) != compose_canonical_content(title=decomposed, description="", why_saved="", tags=[])


# --------------------------------------------------------------------------
# Versioning
# --------------------------------------------------------------------------
def test_canonical_content_version_is_internal_only() -> None:
    """Embedding version is not a Sprint 1 contract field (doc 01, decision 30)."""
    from memovo_ai import schemas

    assert CANONICAL_CONTENT_VERSION == 1

    for model in (
        schemas.ProcessMemoryRequest,
        schemas.ProcessMemoryResponse,
        schemas.ProcessedChunk,
        schemas.SearchMemoryRequest,
        schemas.SearchResult,
    ):
        assert not any("version" in field.lower() for field in model.model_fields)
