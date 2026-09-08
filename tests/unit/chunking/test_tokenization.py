"""Behaviour of the provisional token counter."""

import pytest

from memovo_ai.chunking import CHARS_PER_TOKEN, HeuristicTokenCounter, TokenCounter

pytestmark = pytest.mark.unit


def test_satisfies_the_token_counter_protocol() -> None:
    assert isinstance(HeuristicTokenCounter(), TokenCounter)


def test_empty_text_counts_zero() -> None:
    assert HeuristicTokenCounter().count("") == 0


def test_whitespace_only_counts_zero() -> None:
    assert HeuristicTokenCounter().count("   \n\t  ") == 0


def test_is_deterministic() -> None:
    counter = HeuristicTokenCounter()
    text = "MongoDB Atlas Vector Search uses cosine similarity."

    assert len({counter.count(text) for _ in range(25)}) == 1


def test_whitespace_does_not_affect_the_count() -> None:
    """Indentation must not shift chunk boundaries."""
    counter = HeuristicTokenCounter()

    assert counter.count("alpha beta") == counter.count("  alpha\n\n\t beta  ")


def test_latin_text_counts_by_characters_per_token() -> None:
    counter = HeuristicTokenCounter()

    assert counter.count("a" * (CHARS_PER_TOKEN * 10)) == 10


def test_dense_scripts_count_one_token_per_character() -> None:
    """CJK tokenizes near one token per character; charwise would undercount."""
    counter = HeuristicTokenCounter()

    assert counter.count("日本語の検索") == 6


def test_mixed_dense_and_latin_text_adds_both() -> None:
    counter = HeuristicTokenCounter()

    assert counter.count("日本語 " + "a" * CHARS_PER_TOKEN) == 4


def test_count_grows_monotonically_with_length() -> None:
    counter = HeuristicTokenCounter()
    text = "The quick brown fox jumps over the lazy dog. " * 20
    counts = [counter.count(text[:length]) for length in range(0, len(text), 25)]

    assert counts == sorted(counts)


def test_chars_per_token_is_configurable() -> None:
    assert HeuristicTokenCounter(chars_per_token=1).count("abcd") == 4


def test_chars_per_token_must_be_positive() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        HeuristicTokenCounter(chars_per_token=0)


def test_arabic_is_counted_as_a_space_separated_script() -> None:
    """Arabic is not dense-script; it counts by characters like Latin."""
    counter = HeuristicTokenCounter()

    assert counter.count("مونجو") > 0
    assert counter.count("مونجو") < len("مونجو")
