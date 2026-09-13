"""Canonical content composition.

Turns the fields of a Memory into one stable block of text. That text is the
single input to chunking and, through the chunks, to embedding. Because
``chunkId`` is derived from final chunk content, any instability here would
change chunk IDs on an unchanged Memory -- so this module is deliberately
deterministic and total.

The representation::

    Title:
    MongoDB Vector Search

    Content:
    How Atlas Vector Search works...

    Tags:
    mongodb, vector-search, ai

A Link adds two sections between ``Content`` and ``Tags``: ``Source`` (the
Backend-extracted metadata) and ``Extracted Content`` (the page text the
Backend extracted). Section order is fixed and never varies.

What is embedded follows contract v1.9 section 8.4: title, content, tags,
source metadata and extracted content. ``whySaved`` is gone (section 8.5) and
nothing stands in for it. The ``url`` is deliberately **not** part of the
text: the contract lists it as context rather than semantic input, and a URL
tokenizes into fragments that carry little meaning to a query. Adding it is a
one-line, chunk-ID-changing decision and is pinned by tests either way.

This module -- and this service -- never fetches anything.
"""

from collections.abc import Sequence

__all__ = [
    "CANONICAL_CONTENT_VERSION",
    "CONTENT_LABEL",
    "EXTRACTED_CONTENT_LABEL",
    "SECTION_SEPARATOR",
    "SOURCE_LABEL",
    "SOURCE_SEPARATOR",
    "TAGS_LABEL",
    "TAG_SEPARATOR",
    "TITLE_LABEL",
    "compose_canonical_content",
]

#: Internal representation version. Bumped to 2 by the contract v1.9
#: migration: ``Why Saved`` and ``About`` no longer exist, ``Extracted
#: Content`` does, and ``Source`` carries different fields. Existing chunk IDs
#: for Links, and for Notes that had a ``whySaved``, therefore change; the
#: Backend reprocesses and reconciles. It is NOT part of the public contract.
CANONICAL_CONTENT_VERSION = 2

TITLE_LABEL = "Title:"
CONTENT_LABEL = "Content:"
#: Backend-extracted source metadata (contract section 8.3).
SOURCE_LABEL = "Source:"
#: Backend-extracted page text (contract section 8.4).
EXTRACTED_CONTENT_LABEL = "Extracted Content:"
TAGS_LABEL = "Tags:"

TAG_SEPARATOR = ", "
SOURCE_SEPARATOR = ", "
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


def _normalize_parts(parts: Sequence[str]) -> list[str]:
    """Strip each part and drop any that are empty afterwards.

    Order is preserved exactly as supplied. Parts are not sorted and not
    deduplicated: reordering is a Backend-visible change, and inventing either
    behaviour here would be a product decision no source document makes.
    """
    stripped = (_normalize_text(part) for part in parts)
    return [part for part in stripped if part]


def compose_canonical_content(
    *,
    title: str,
    content: str | None,
    tags: Sequence[str] | None = None,
    source: Sequence[str] = (),
    extracted_content: str | None = None,
) -> str:
    """Compose the canonical text for a Memory.

    Sections always appear in the order Title, Content, Source, Extracted
    Content, Tags. A section whose value is empty after normalization is
    omitted entirely, rather than emitted as a bare label with nothing under
    it -- an empty labelled section would add tokens that carry no meaning to
    the embedding, and ``null`` is never rendered as text.

    ``source`` and ``extracted_content`` belong to Links and default to empty,
    so a Note composes to exactly Title, Content, Tags. ``source`` receives
    already-selected text parts rather than a schema object, so this module
    stays free of transport types; choosing which metadata fields are worth
    embedding is the caller's job.

    Returns an empty string when every field is empty. Deciding whether that
    is a request error belongs to the processing service and the error
    contract, not to this function.
    """
    sections: list[str] = []

    for label, value in (
        (TITLE_LABEL, _normalize_text(title)),
        (CONTENT_LABEL, _normalize_text(content or "")),
        (SOURCE_LABEL, SOURCE_SEPARATOR.join(_normalize_parts(source))),
        (EXTRACTED_CONTENT_LABEL, _normalize_text(extracted_content or "")),
        (TAGS_LABEL, TAG_SEPARATOR.join(_normalize_parts(tags or ()))),
    ):
        if value:
            sections.append(f"{label}\n{value}")

    return SECTION_SEPARATOR.join(sections)
