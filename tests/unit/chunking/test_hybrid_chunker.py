"""Behaviour of the hybrid chunker.

Covers the test matrix in doc 03 (Phase 03) and doc 05, section 3.

Most tests use a small ``ChunkingConfig`` so that documents stay readable.
Boundary behaviour is scale-independent; a dedicated test pins the locked
production defaults of 500/50.
"""

import pytest

from memovo_ai.chunking import (
    DEFAULT_OVERLAP_TOKENS,
    DEFAULT_TARGET_TOKENS,
    Chunk,
    ChunkingConfig,
    HeuristicTokenCounter,
    HybridChunker,
)
from memovo_ai.understanding import compose_canonical_content

pytestmark = pytest.mark.unit

TOKENS = HeuristicTokenCounter()


def make_chunker(target: int = 40, overlap: int = 0) -> HybridChunker:
    return HybridChunker(config=ChunkingConfig(target_tokens=target, overlap_tokens=overlap))


_BASE36 = "0123456789abcdefghijklmnopqrstuvwxyz"


def _suffix(index: int) -> str:
    """Three base-36 digits, so every word is unique up to 46656."""
    return _BASE36[index // 1296 % 36] + _BASE36[index // 36 % 36] + _BASE36[index % 36]


#: Single-character stems, so distinct runs of words stay distinguishable.
STEMS = "abcdefghijklmnopqrstuvwxyz"


def words(count: int, stem: str = "a") -> str:
    """A run of `count` unique words that each cost exactly one token.

    One stem character plus three base-36 digits is four non-space
    characters, which is exactly `CHARS_PER_TOKEN`. Keeping words at one token
    apiece lets these tests state expected chunk counts directly, so the stem
    must stay a single character.
    """
    if len(stem) != 1:
        message = f"stem must be exactly one character, got {stem!r}"
        raise ValueError(message)

    return " ".join(f"{stem}{_suffix(index)}" for index in range(count))


# --------------------------------------------------------------------------
# Locked configuration
# --------------------------------------------------------------------------
def test_default_config_uses_the_locked_values() -> None:
    """Target ~500 tokens, overlap ~50 (doc 01, decisions 5 and 6)."""
    config = ChunkingConfig()

    assert config.target_tokens == DEFAULT_TARGET_TOKENS == 500
    assert config.overlap_tokens == DEFAULT_OVERLAP_TOKENS == 50


def test_chunker_defaults_to_the_locked_config() -> None:
    chunks = HybridChunker().chunk(words(4000))

    assert len(chunks) > 1
    for chunk in chunks:
        assert TOKENS.count(chunk.content) <= DEFAULT_TARGET_TOKENS + DEFAULT_OVERLAP_TOKENS


@pytest.mark.parametrize(
    ("target", "overlap"),
    [(0, 0), (-1, 0), (100, -1), (100, 100), (100, 200)],
)
def test_invalid_config_is_rejected(target: int, overlap: int) -> None:
    with pytest.raises(ValueError, match="tokens"):
        ChunkingConfig(target_tokens=target, overlap_tokens=overlap)


# --------------------------------------------------------------------------
# Short documents
# --------------------------------------------------------------------------
def test_short_document_is_a_single_chunk() -> None:
    chunks = make_chunker().chunk("A short memory about vector search.")

    assert len(chunks) == 1
    assert chunks[0].index == 0
    assert chunks[0].content == "A short memory about vector search."


def test_empty_content_yields_no_chunks() -> None:
    assert make_chunker().chunk("") == []


@pytest.mark.parametrize("blank", ["   ", "\n\n\n", "\t \n \t"])
def test_blank_content_yields_no_chunks(blank: str) -> None:
    assert make_chunker().chunk(blank) == []


def test_single_word_document_is_one_chunk() -> None:
    chunks = make_chunker().chunk("mongodb")

    assert [chunk.content for chunk in chunks] == ["mongodb"]


# --------------------------------------------------------------------------
# Structural invariants
# --------------------------------------------------------------------------
def test_no_chunk_is_ever_empty() -> None:
    text = "# H1\n\n\n\nOne.\n\n\n\n# H2\n\n\n\nTwo.\n\n\n\n"
    chunks = make_chunker(target=5, overlap=2).chunk(text)

    assert chunks
    for chunk in chunks:
        assert chunk.content.strip()


def test_indices_are_contiguous_and_zero_based() -> None:
    chunks = make_chunker().chunk(words(500))

    assert [chunk.index for chunk in chunks] == list(range(len(chunks)))


def test_token_count_matches_the_final_content() -> None:
    """Phase 04 hashes final content, so the reported count must describe it."""
    for chunk in make_chunker(overlap=10).chunk(words(300)):
        assert chunk.token_count == TOKENS.count(chunk.content)


def test_chunks_are_immutable() -> None:
    chunk = make_chunker().chunk("hello")[0]

    with pytest.raises(AttributeError):
        chunk.content = "mutated"  # type: ignore[misc]


def test_no_content_is_lost_without_overlap() -> None:
    text = words(300)
    rejoined = " ".join(chunk.content for chunk in make_chunker(overlap=0).chunk(text))

    assert rejoined.split() == text.split()


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------
def test_repeated_chunking_is_identical() -> None:
    """Retries with unchanged input must produce identical chunk IDs."""
    text = words(400)
    chunker = make_chunker(overlap=10)
    runs = [tuple(chunk.content for chunk in chunker.chunk(text)) for _ in range(10)]

    assert len(set(runs)) == 1


def test_separate_instances_agree() -> None:
    text = words(400)

    assert make_chunker(overlap=10).chunk(text) == make_chunker(overlap=10).chunk(text)


def test_changed_content_changes_chunks() -> None:
    baseline = make_chunker().chunk(words(100))
    modified = make_chunker().chunk(words(100) + " extra")

    assert baseline != modified


def test_unchanged_prefix_keeps_its_early_chunks() -> None:
    """Appending to a Memory must not disturb chunks before the change."""
    prefix = words(120)
    first = make_chunker(overlap=0).chunk(prefix)
    second = make_chunker(overlap=0).chunk(f"{prefix} {words(40, stem='b')}")

    assert [c.content for c in second][: len(first) - 1] == [c.content for c in first][:-1]


# --------------------------------------------------------------------------
# Boundary preference: sections
# --------------------------------------------------------------------------
def test_headings_start_new_sections() -> None:
    text = f"# Alpha\n\n{words(10)}\n\n# Beta\n\n{words(10, stem='b')}"
    chunks = make_chunker(target=15).chunk(text)

    assert len(chunks) == 2
    assert chunks[0].content.startswith("# Alpha")
    assert chunks[1].content.startswith("# Beta")


def test_multiple_headings_are_each_respected() -> None:
    text = "\n\n".join(f"# Section {n}\n\n{words(30)}" for n in range(4))
    chunks = make_chunker(target=45).chunk(text)

    assert len(chunks) == 4
    for index, chunk in enumerate(chunks):
        assert chunk.content.startswith(f"# Section {index}")


@pytest.mark.parametrize("hashes", ["#", "##", "###", "####", "#####", "######"])
def test_all_atx_heading_levels_are_detected(hashes: str) -> None:
    text = f"Intro body.\n\n{hashes} Heading\n\nBody."
    chunks = make_chunker(target=4).chunk(text)

    assert any(chunk.content.startswith(f"{hashes} Heading") for chunk in chunks)


def test_hash_without_a_space_is_not_a_heading() -> None:
    """`#hashtag` is content, not structure."""
    chunks = make_chunker(target=200).chunk("Intro.\n\n#hashtag stays inline.")

    assert len(chunks) == 1


def test_small_sections_are_packed_together() -> None:
    """Sections are packed toward the target, not emitted one per chunk."""
    text = "\n\n".join(f"# S{n}\n\nshort" for n in range(4))
    chunks = make_chunker(target=200).chunk(text)

    assert len(chunks) == 1


# --------------------------------------------------------------------------
# Boundary preference: paragraphs
# --------------------------------------------------------------------------
def test_multiple_paragraphs_split_on_paragraph_boundaries() -> None:
    text = "\n\n".join(words(30, stem=STEMS[n]) for n in range(4))
    chunks = make_chunker(target=35).chunk(text)

    assert len(chunks) == 4
    for index, chunk in enumerate(chunks):
        assert chunk.content.startswith(f"{STEMS[index]}000")


def test_paragraphs_are_packed_until_the_target() -> None:
    text = "\n\n".join(words(10, stem=STEMS[n]) for n in range(4))
    chunks = make_chunker(target=25).chunk(text)

    assert len(chunks) == 2


def test_paragraph_separator_is_preserved_when_packing() -> None:
    text = "First para.\n\nSecond para."
    chunks = make_chunker(target=200).chunk(text)

    assert chunks[0].content == "First para.\n\nSecond para."


# --------------------------------------------------------------------------
# Boundary preference: sentences
# --------------------------------------------------------------------------
def test_long_paragraph_falls_back_to_sentences() -> None:
    paragraph = " ".join(f"Sentence number {n} about vectors." for n in range(12))
    chunks = make_chunker(target=20).chunk(paragraph)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.content.endswith(".")


def test_sentences_are_not_cut_mid_sentence() -> None:
    paragraph = " ".join(f"Alpha beta gamma delta {n}." for n in range(10))
    chunks = make_chunker(target=15).chunk(paragraph)

    for chunk in chunks:
        assert chunk.content.count(".") >= 1


@pytest.mark.parametrize("terminator", [".", "!", "?", "…"])
def test_latin_sentence_terminators_split(terminator: str) -> None:
    paragraph = " ".join(f"Sentence {n}{terminator}" for n in range(10))
    chunks = make_chunker(target=6).chunk(paragraph)

    assert len(chunks) > 1


def test_decimals_do_not_split_sentences() -> None:
    """A period with no following whitespace is not a terminator."""
    text = " ".join(f"The threshold is 0.75 for retrieval number {n}." for n in range(10))
    chunks = make_chunker(target=20).chunk(text)

    assert len(chunks) > 1
    for chunk in chunks:
        assert "0." not in chunk.content or "0.75" in chunk.content


def test_punctuation_with_closing_quote_splits() -> None:
    paragraph = " ".join(f'He said "thing {n}." ' for n in range(10)).strip()
    chunks = make_chunker(target=10).chunk(paragraph)

    assert len(chunks) > 1


# --------------------------------------------------------------------------
# Boundary preference: token fallback
# --------------------------------------------------------------------------
def test_long_sentence_falls_back_to_words() -> None:
    sentence = words(200) + "."
    chunks = make_chunker(target=30, overlap=0).chunk(sentence)

    assert len(chunks) > 1
    for chunk in chunks:
        assert TOKENS.count(chunk.content) <= 30


def test_single_enormous_word_is_character_split() -> None:
    """The last resort: one token-run longer than the whole target."""
    chunks = make_chunker(target=10, overlap=0).chunk("z" * 500)

    assert len(chunks) > 1
    for chunk in chunks:
        assert TOKENS.count(chunk.content) <= 10
    assert "".join(chunk.content for chunk in chunks) == "z" * 500


def test_character_split_terminates_on_pathological_input() -> None:
    chunks = make_chunker(target=1, overlap=0).chunk("y" * 200)

    assert chunks
    assert "".join(chunk.content for chunk in chunks) == "y" * 200


def test_fallback_order_prefers_paragraphs_over_words() -> None:
    """A document that fits paragraph-wise must not be word-split."""
    text = "\n\n".join(words(20, stem=STEMS[n]) for n in range(3))
    chunks = make_chunker(target=25, overlap=0).chunk(text)

    assert len(chunks) == 3
    for index, chunk in enumerate(chunks):
        assert chunk.content == words(20, stem=STEMS[index])


# --------------------------------------------------------------------------
# Target adherence
# --------------------------------------------------------------------------
def test_chunks_stay_within_target_plus_overlap() -> None:
    text = "\n\n".join(words(60, stem=STEMS[n]) for n in range(10))
    chunks = make_chunker(target=50, overlap=10).chunk(text)

    for chunk in chunks:
        assert chunk.token_count <= 50 + 10


def test_target_is_soft_and_boundaries_win() -> None:
    """Chunks land near, not exactly at, the target."""
    text = " ".join(f"Sentence {n} has several words in it." for n in range(40))
    chunks = make_chunker(target=40, overlap=0).chunk(text)

    assert len({chunk.token_count for chunk in chunks}) > 1
    for chunk in chunks:
        assert chunk.token_count <= 40


# --------------------------------------------------------------------------
# Overlap
# --------------------------------------------------------------------------
def test_overlap_repeats_the_tail_of_the_previous_chunk() -> None:
    text = "\n\n".join(words(30, stem=STEMS[n]) for n in range(3))
    without = make_chunker(target=35, overlap=0).chunk(text)
    with_overlap = make_chunker(target=35, overlap=8).chunk(text)

    assert len(without) == len(with_overlap) == 3
    for previous, current in zip(without[:-1], with_overlap[1:], strict=True):
        tail_word = previous.content.split()[-1]
        assert tail_word in current.content


def test_first_chunk_is_never_prefixed() -> None:
    text = "\n\n".join(words(30, stem=STEMS[n]) for n in range(3))
    without = make_chunker(target=35, overlap=0).chunk(text)
    with_overlap = make_chunker(target=35, overlap=8).chunk(text)

    assert without[0].content == with_overlap[0].content


def test_zero_overlap_leaves_chunks_untouched() -> None:
    text = "\n\n".join(words(30, stem=STEMS[n]) for n in range(3))

    assert make_chunker(target=35, overlap=0).chunk(text) == make_chunker(
        target=35, overlap=0
    ).chunk(text)


def test_overlap_does_not_apply_to_a_single_chunk() -> None:
    chunks = make_chunker(target=200, overlap=50).chunk("Only one chunk here.")

    assert len(chunks) == 1
    assert chunks[0].content == "Only one chunk here."


def test_overlap_stays_within_its_budget() -> None:
    text = "\n\n".join(words(40, stem=STEMS[n]) for n in range(4))
    without = make_chunker(target=45, overlap=12).chunk(text)

    for chunk in without[1:]:
        assert chunk.token_count <= 45 + 12


def test_overlap_prefers_whole_sentences() -> None:
    text = "Alpha one. Beta two. Gamma three.\n\nDelta four. Epsilon five."
    chunks = make_chunker(target=12, overlap=6).chunk(text)

    assert len(chunks) > 1
    assert "Gamma three." in chunks[1].content


def test_overlap_is_skipped_when_nothing_fits_the_budget() -> None:
    """No mid-word cutting: the chunk is left unchanged instead."""
    text = "\n\n".join("z" * 60 for _ in range(3))
    chunks = make_chunker(target=20, overlap=1).chunk(text)

    for chunk in chunks:
        assert chunk.content.strip()


def test_overlap_is_deterministic() -> None:
    text = "\n\n".join(words(40, stem=STEMS[n]) for n in range(5))
    chunker = make_chunker(target=45, overlap=10)
    runs = [tuple(c.content for c in chunker.chunk(text)) for _ in range(10)]

    assert len(set(runs)) == 1


# --------------------------------------------------------------------------
# Language coverage
# --------------------------------------------------------------------------
def test_english_document_chunks() -> None:
    text = " ".join(f"The vector index stores embeddings for memory {n}." for n in range(20))
    chunks = make_chunker(target=25).chunk(text)

    assert len(chunks) > 1


def test_arabic_document_chunks_on_arabic_sentence_terminators() -> None:
    sentences = [f"هذه الجملة رقم {n} عن البحث الدلالي." for n in range(12)]
    chunks = make_chunker(target=25).chunk(" ".join(sentences))

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.content.strip()


def test_arabic_question_mark_terminates_a_sentence() -> None:
    text = " ".join(f"ما الذي حفظته عن المتجهات {n}؟" for n in range(10))
    chunks = make_chunker(target=20).chunk(text)

    assert len(chunks) > 1


def test_mixed_arabic_and_english_chunks() -> None:
    text = " ".join(f"MongoDB يخزن المتجهات in a vector index number {n}." for n in range(15))
    chunks = make_chunker(target=30).chunk(text)

    assert len(chunks) > 1
    joined = " ".join(chunk.content for chunk in chunks)
    assert "MongoDB" in joined
    assert "المتجهات" in joined


def test_cjk_terminators_split_without_spaces() -> None:
    text = "".join(f"これは文章です{n}。" for n in range(30))
    chunks = make_chunker(target=20).chunk(text)

    assert len(chunks) > 1


@pytest.mark.parametrize(
    "text",
    ["Ünïcodé ✅ content here.", "emoji 🧠 memory note.", "math ∑ ≥ 0.75 threshold."],
)
def test_unicode_survives_chunking(text: str) -> None:
    chunks = make_chunker(target=200).chunk(text)

    assert [chunk.content for chunk in chunks] == [text]


# --------------------------------------------------------------------------
# Integration with canonical content (Phase 02)
# --------------------------------------------------------------------------
def test_chunks_canonical_content_including_tags() -> None:
    canonical = compose_canonical_content(
        title="MongoDB Vector Search",
        content="How Atlas Vector Search works.",
        tags=["mongodb", "vector-search", "ai"],
    )
    chunks = make_chunker(target=500).chunk(canonical)

    assert len(chunks) == 1
    assert "Tags:\nmongodb, vector-search, ai" in chunks[0].content


def test_canonical_content_with_empty_optional_values_still_chunks() -> None:
    canonical = compose_canonical_content(title="Just a title", content="", tags=[])
    chunks = make_chunker().chunk(canonical)

    assert [chunk.content for chunk in chunks] == ["Title:\nJust a title"]


def test_fully_empty_canonical_content_yields_no_chunks() -> None:
    canonical = compose_canonical_content(title="", content="", tags=[])

    assert make_chunker().chunk(canonical) == []


def test_long_canonical_content_splits_on_its_labels() -> None:
    canonical = compose_canonical_content(
        title="Vector search",
        content=words(200),
        tags=["a", "b"],
    )
    chunks = make_chunker(target=60, overlap=0).chunk(canonical)

    assert len(chunks) > 1
    assert chunks[0].content.startswith("Title:")
    assert any("Tags:" in chunk.content for chunk in chunks)


# --------------------------------------------------------------------------
# Injected token counter
# --------------------------------------------------------------------------
def test_a_custom_token_counter_is_used() -> None:
    """Phase 06 swaps in the real Qwen3 tokenizer through this seam."""

    class OneTokenPerWord:
        def count(self, text: str) -> int:
            return len(text.split())

    chunker = HybridChunker(
        config=ChunkingConfig(target_tokens=10, overlap_tokens=0),
        token_counter=OneTokenPerWord(),
    )
    chunks = chunker.chunk(words(45))

    assert len(chunks) == 5
    for chunk in chunks:
        assert len(chunk.content.split()) <= 10


def test_chunk_is_a_plain_domain_model() -> None:
    """No chunkId (Phase 04) and no embedding (Phase 06) at this layer."""
    fields = Chunk.__dataclass_fields__

    assert set(fields) == {"index", "content", "token_count"}
