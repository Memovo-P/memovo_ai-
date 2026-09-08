"""Token counting for the chunker.

The ~500-token target and ~50-token overlap are locked (doc 01, decisions 5
and 6), but no source document names a tokenizer. The chunker therefore
depends on the :class:`TokenCounter` protocol rather than on any concrete
implementation, and :class:`HeuristicTokenCounter` is the Sprint 1 default.

Swapping in the real Qwen3 tokenizer at Phase 06 means passing a different
``TokenCounter`` to :class:`~memovo_ai.chunking.hybrid_chunker.HybridChunker`.
No chunker logic changes. Doing that now would pull a model-runtime dependency
into the project three phases before the embedding provider exists.

Because the target is explicitly soft, an approximate count is acceptable: a
count that is off by 10% shifts a boundary, it does not break the contract.
Chunk IDs stay deterministic either way -- they hash final chunk content, and
this counter is a pure function. Note that changing the counter later *will*
move boundaries and therefore change chunk IDs, which the Backend reconciles
exactly as it does for any content change.
"""

import math
from typing import Protocol, runtime_checkable

__all__ = ["CHARS_PER_TOKEN", "HeuristicTokenCounter", "TokenCounter"]

#: Average characters per token for space-separated scripts. The usual
#: rule-of-thumb figure for subword vocabularies.
CHARS_PER_TOKEN = 4


@runtime_checkable
class TokenCounter(Protocol):
    """Counts tokens in a piece of text.

    Implementations must be deterministic and pure: the same text must always
    produce the same count, because chunk boundaries -- and therefore chunk
    IDs -- depend on it.
    """

    def count(self, text: str) -> int:
        """Return the number of tokens in ``text``."""
        ...


def _is_dense_script(character: str) -> bool:
    """True for scripts that do not separate words with spaces.

    CJK ideographs, kana and Hangul tokenize at roughly one token per
    character, so counting them by character length would undercount badly.
    """
    codepoint = ord(character)
    return (
        0x3040 <= codepoint <= 0x30FF  # Hiragana + Katakana
        or 0x3400 <= codepoint <= 0x4DBF  # CJK Extension A
        or 0x4E00 <= codepoint <= 0x9FFF  # CJK Unified Ideographs
        or 0xAC00 <= codepoint <= 0xD7AF  # Hangul syllables
        or 0xF900 <= codepoint <= 0xFAFF  # CJK compatibility ideographs
    )


class HeuristicTokenCounter:
    """Provisional, dependency-free token estimate.

    Dense-script characters count as one token each; everything else is
    counted at :data:`CHARS_PER_TOKEN` characters per token. Whitespace is not
    counted, so the estimate does not drift with indentation.

    This is an approximation, not the Qwen3 vocabulary. It exists so Phase 03
    can be built, tested and reviewed offline.
    """

    __slots__ = ("_chars_per_token",)

    def __init__(self, *, chars_per_token: int = CHARS_PER_TOKEN) -> None:
        if chars_per_token < 1:
            message = "chars_per_token must be at least 1"
            raise ValueError(message)
        self._chars_per_token = chars_per_token

    def count(self, text: str) -> int:
        dense = 0
        sparse = 0
        for character in text:
            if character.isspace():
                continue
            if _is_dense_script(character):
                dense += 1
            else:
                sparse += 1

        return dense + math.ceil(sparse / self._chars_per_token)
