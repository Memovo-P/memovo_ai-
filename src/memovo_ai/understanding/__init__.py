"""Content understanding and preparation.

This layer is deterministic: canonical content composition only. There is no
LLM classifier, no prompt-based routing and no intent detection -- the
endpoint already identifies the operation (contract section 4).
"""

from memovo_ai.understanding.content import (
    CANONICAL_CONTENT_VERSION,
    CONTENT_LABEL,
    EXTRACTED_CONTENT_LABEL,
    SECTION_SEPARATOR,
    SOURCE_LABEL,
    SOURCE_SEPARATOR,
    TAG_SEPARATOR,
    TAGS_LABEL,
    TITLE_LABEL,
    compose_canonical_content,
)

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
