"""Hybrid chunker.

Splits canonical content (Phase 02) into chunks of roughly
:data:`~memovo_ai.chunking.models.DEFAULT_TARGET_TOKENS` tokens, preferring
natural boundaries over arithmetic ones.

The 500-token target is soft. Semantic units are packed *toward* it; a unit is
only broken apart when it does not fit on its own, and then only one level
deeper (doc 03, Phase 03)::

    section -> paragraph -> sentence -> token/character split

Nothing is cut at 500 tokens first. Every step is a pure function of the input
text, so reprocessing an unchanged Memory reproduces byte-identical chunks and
therefore identical chunk IDs (Phase 04).
"""

import re
from collections.abc import Callable, Iterable, Sequence
from itertools import pairwise

from memovo_ai.chunking.models import Chunk, ChunkingConfig
from memovo_ai.chunking.tokenization import HeuristicTokenCounter, TokenCounter

__all__ = ["HybridChunker"]

#: ATX markdown headings (``# Heading`` through ``###### Heading``).
_HEADING = re.compile(r"^#{1,6}\s+\S")

#: A blank line, optionally containing whitespace, separates paragraphs.
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")

#: Sentence terminators. The first branch covers space-separated scripts --
#: Latin, plus the Arabic question mark (U+061F) and Arabic full stop
#: (U+06D4) -- allows trailing quotes or brackets, and requires following
#: whitespace so that decimals and "e.g." split less eagerly. The second
#: branch covers CJK terminators (U+3002, U+FF01, U+FF1F), which are not
#: followed by a space.
#:
#: The literal characters are kept rather than \u escapes because they are the
#: subject matter; pyproject exempts this file from RUF001/RUF003, whose
#: suggested ASCII replacements would break the pattern.
_SENTENCE_END = re.compile(
    r"(?:[.!?…؟۔]+[)\]\"'”’」]*(?=\s|$)"
    r"|[。！？]+)"
)

_PARAGRAPH_JOINER = "\n\n"
_INLINE_JOINER = " "


class HybridChunker:
    """Chunks canonical content along natural boundaries.

    The token counter is injected so the real Qwen3 tokenizer can replace the
    Sprint 1 heuristic at Phase 06 without touching this class.
    """

    __slots__ = ("_config", "_tokens")

    def __init__(
        self,
        *,
        config: ChunkingConfig | None = None,
        token_counter: TokenCounter | None = None,
    ) -> None:
        self._config = config if config is not None else ChunkingConfig()
        self._tokens = token_counter if token_counter is not None else HeuristicTokenCounter()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def chunk(self, content: str) -> list[Chunk]:
        """Split ``content`` into chunks with contiguous indices from zero.

        Returns an empty list for blank input. No chunk is ever empty.
        """
        text = content.strip()
        if not text:
            return []

        bodies = self._pack(self._split_sections(text), _PARAGRAPH_JOINER, self._split_section)
        bodies = self._apply_overlap(bodies)

        return [
            Chunk(index=index, content=body, token_count=self._tokens.count(body))
            for index, body in enumerate(_non_empty(bodies))
        ]

    # ------------------------------------------------------------------
    # Packing
    # ------------------------------------------------------------------
    def _pack(
        self,
        units: Sequence[str],
        joiner: str,
        decompose: Callable[[str], list[str]],
    ) -> list[str]:
        """Greedily fill chunks with whole units, without exceeding the target.

        A unit that exceeds the target on its own is handed to ``decompose``,
        which re-splits it at the next boundary level down.
        """
        target = self._config.target_tokens
        bodies: list[str] = []
        current: list[str] = []

        def flush() -> None:
            if current:
                bodies.append(joiner.join(current))
                current.clear()

        for unit in units:
            if self._tokens.count(unit) > target:
                flush()
                bodies.extend(decompose(unit))
                continue

            if current and self._tokens.count(joiner.join([*current, unit])) > target:
                flush()
            current.append(unit)

        flush()
        return bodies

    def _split_section(self, section: str) -> list[str]:
        return self._pack(self._split_paragraphs(section), _PARAGRAPH_JOINER, self._split_paragraph)

    def _split_paragraph(self, paragraph: str) -> list[str]:
        return self._pack(self._split_sentences(paragraph), _INLINE_JOINER, self._split_sentence)

    def _split_sentence(self, sentence: str) -> list[str]:
        return self._pack(sentence.split(), _INLINE_JOINER, self._split_by_characters)

    def _split_by_characters(self, word: str) -> list[str]:
        """Last resort for a single token-run longer than the whole target."""
        target = self._config.target_tokens
        pieces: list[str] = []
        remaining = word

        while remaining:
            if self._tokens.count(remaining) <= target:
                pieces.append(remaining)
                break
            cut = self._longest_prefix_within_target(remaining)
            pieces.append(remaining[:cut])
            remaining = remaining[cut:]

        return pieces

    def _longest_prefix_within_target(self, text: str) -> int:
        """Binary-search the longest prefix of ``text`` that fits the target.

        Always returns at least 1 so :meth:`_split_by_characters` terminates.
        """
        target = self._config.target_tokens
        low, high = 1, len(text)

        while low < high:
            middle = (low + high + 1) // 2
            if self._tokens.count(text[:middle]) <= target:
                low = middle
            else:
                high = middle - 1

        return low

    # ------------------------------------------------------------------
    # Overlap
    # ------------------------------------------------------------------
    def _apply_overlap(self, bodies: Sequence[str]) -> list[str]:
        """Prefix each chunk with the tail of its predecessor.

        Overlap is best-effort, as doc 03 specifies ("where practical"). When
        no whole sentence or word fits the budget the chunk is left unchanged
        rather than cut mid-word.
        """
        overlap = self._config.overlap_tokens
        if overlap <= 0 or len(bodies) < 2:
            return list(bodies)

        overlapped = [bodies[0]]
        for previous, body in pairwise(bodies):
            tail = self._tail_within_budget(previous, overlap)
            overlapped.append(f"{tail}{_PARAGRAPH_JOINER}{body}" if tail else body)

        return overlapped

    def _tail_within_budget(self, text: str, budget: int) -> str:
        """Return the longest trailing run of ``text`` that fits ``budget``.

        Whole sentences are preferred; trailing words are the fallback.
        """
        sentences = self._take_tail(self._split_sentences(text), budget)
        if sentences:
            return _INLINE_JOINER.join(sentences)

        return _INLINE_JOINER.join(self._take_tail(text.split(), budget))

    def _take_tail(self, units: Sequence[str], budget: int) -> list[str]:
        selected: list[str] = []
        total = 0

        for unit in reversed(units):
            cost = self._tokens.count(unit)
            if total + cost > budget:
                break
            selected.append(unit)
            total += cost

        selected.reverse()
        return selected

    # ------------------------------------------------------------------
    # Boundary detection
    # ------------------------------------------------------------------
    @staticmethod
    def _split_sections(text: str) -> list[str]:
        """Split on markdown headings; each heading starts a new section."""
        sections: list[str] = []
        current: list[str] = []

        for line in text.split("\n"):
            if _HEADING.match(line) and any(existing.strip() for existing in current):
                sections.append("\n".join(current).strip())
                current = [line]
            else:
                current.append(line)

        sections.append("\n".join(current).strip())
        return _non_empty(sections) or [text]

    @staticmethod
    def _split_paragraphs(section: str) -> list[str]:
        return _non_empty(_PARAGRAPH_BREAK.split(section)) or [section]

    @staticmethod
    def _split_sentences(paragraph: str) -> list[str]:
        sentences: list[str] = []
        start = 0

        for match in _SENTENCE_END.finditer(paragraph):
            sentences.append(paragraph[start : match.end()])
            start = match.end()
        sentences.append(paragraph[start:])

        return _non_empty(sentences) or [paragraph]


def _non_empty(values: Iterable[str]) -> list[str]:
    return [stripped for stripped in map(str.strip, values) if stripped]
