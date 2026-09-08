"""Content understanding and preparation.

Sprint 1 keeps this layer deterministic: canonical content composition only.
There is no LLM classifier, no prompt-based routing and no intent detection --
the endpoint already identifies the operation (doc 01, section 6).
"""

from memovo_ai.understanding.content import (
    CANONICAL_CONTENT_VERSION,
    CONTENT_LABEL,
    SECTION_SEPARATOR,
    TAG_SEPARATOR,
    TAGS_LABEL,
    TITLE_LABEL,
    WHY_SAVED_LABEL,
    compose_canonical_content,
)

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
