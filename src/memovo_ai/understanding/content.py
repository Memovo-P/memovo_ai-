"""Canonical content composition.

Turns the fields of a Memory into one stable block of text. That text is the
single input to chunking (Phase 03) and, through the chunks, to embedding
(Phase 06). Because ``chunkId`` is derived from final chunk content, any
instability here would change chunk IDs on an unchanged Memory -- so this
module is deliberately deterministic and total.

The representation follows ``03_IMPLEMENTATION_PLAN.md``, Phase 02::

    Title:
    MongoDB Vector Search

    Content:
    How Atlas Vector Search works...

    Why Saved:
    Useful for the memory project

    Tags:
    mongodb, vector-search, ai

Embedding input is Title + Description/Content + WhySaved + Tags (doc 01,
locked decisions 1 and 2). Section order is fixed and never varies.
"""

from collections.abc import Sequence

__all__ = [
    "CANONICAL_CONTENT_VERSION",
    "CONTENT_LABEL",
    "SECTION_SEPARATOR",
    "TAGS_LABEL",
    "TAG_SEPARATOR",
    "TITLE_LABEL",
    "WHY_SAVED_LABEL",
    "compose_canonical_content",
]

#: Internal representation version. Bumping it changes canonical content and
#: therefore every chunkId, so it exists to make a future migration explicit
#: rather than silent. It is NOT part of the public API contract: embedding
#: version is deferred and is not a Sprint 1 contract field (doc 01,
#: locked decision 30).
CANONICAL_CONTENT_VERSION = 1

TITLE_LABEL = "Title:"
#: The request field is named ``description``; the canonical label is
#: ``Content:``, matching the documented representation.
CONTENT_LABEL = "Content:"
WHY_SAVED_LABEL = "Why Saved:"
TAGS_LABEL = "Tags:"

TAG_SEPARATOR = ", "
SECTION_SEPARATOR = "\n\n"


def _normalize_text(value: str) -> str:
    """Normalize a field value so equivalent input yields identical output.

    Three transformations, each needed for deterministic chunk IDs:

    1. Line endings collapse to ``\\n``. Without this the same Memory produces
       different chunk IDs depending on the platform that submitted it.
    2. Trailing whitespace is stripped per line, so a "blank" line containing
       spaces really is blank. The chunker splits on paragraph boundaries, so
       ``"\\n   \\n"`` and ``"\\n\\n"`` must not chunk differently.
    3. Leading and trailing whitespace is stripped from the value as a whole.

    Interior blank lines and indentation are preserved: they carry the
    paragraph structure the hybrid chunker depends on.

    The function is idempotent -- normalizing normalized text is a no-op.
    """
    unified_newlines = value.replace("\r\n", "\n").replace("\r", "\n")
    without_trailing_spaces = "\n".join(line.rstrip() for line in unified_newlines.split("\n"))
    return without_trailing_spaces.strip()


def _normalize_tags(tags: Sequence[str]) -> list[str]:
    """Strip each tag and drop any that are empty afterwards.

    Order is preserved exactly as supplied. Tags are not sorted and not
    deduplicated: reordering is a Backend-visible change, and inventing either
    behaviour here would be a product decision no source document makes.
    """
    stripped = (_normalize_text(tag) for tag in tags)
    return [tag for tag in stripped if tag]


def compose_canonical_content(
    *,
    title: str,
    description: str,
    why_saved: str,
    tags: Sequence[str],
) -> str:
    """Compose the canonical text for a Memory.

    Sections always appear in the order Title, Content, Why Saved, Tags. A
    section whose value is empty after normalization is omitted entirely,
    rather than emitted as a bare label with nothing under it -- an empty
    labelled section would add tokens that carry no meaning to the embedding.

    Returns an empty string when every field is empty. Deciding whether that
    is a request error belongs to the processing service and the error
    contract, not to this function.
    """
    sections: list[str] = []

    for label, value in (
        (TITLE_LABEL, _normalize_text(title)),
        (CONTENT_LABEL, _normalize_text(description)),
        (WHY_SAVED_LABEL, _normalize_text(why_saved)),
        (TAGS_LABEL, TAG_SEPARATOR.join(_normalize_tags(tags))),
    ):
        if value:
            sections.append(f"{label}\n{value}")

    return SECTION_SEPARATOR.join(sections)
