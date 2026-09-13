"""Output validation: reasoning stripping, JSON extraction, schema checks."""

import pytest
from pydantic import BaseModel, ConfigDict

from memovo_ai.generation import (
    GenerationResult,
    InvalidGenerationOutputError,
    extract_json_object,
    strip_reasoning,
    validate_output,
)

pytestmark = pytest.mark.unit


class Answer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str
    used: list[str]


def result(text: str, finish_reason: str = "stop") -> GenerationResult:
    return GenerationResult(text=text, finish_reason=finish_reason, model="m")


# --------------------------------------------------------------------------
# Reasoning traces
# --------------------------------------------------------------------------
def test_a_closed_think_block_is_removed() -> None:
    assert strip_reasoning('<think>private chain of thought</think>\n{"a": 1}') == '{"a": 1}'


def test_an_unterminated_think_block_is_removed() -> None:
    assert strip_reasoning('{"a": 1}\n<think>ran out of budget') == '{"a": 1}'


def test_think_tags_are_case_insensitive_and_multiline() -> None:
    assert strip_reasoning("<THINK>\nline one\nline two\n</THINK>x") == "x"


def test_text_without_reasoning_is_only_trimmed() -> None:
    assert strip_reasoning("  plain  ") == "plain"


# --------------------------------------------------------------------------
# JSON extraction
# --------------------------------------------------------------------------
def test_a_bare_object_is_parsed() -> None:
    assert extract_json_object('{"answer": "x", "used": []}') == {"answer": "x", "used": []}


def test_a_fenced_object_is_parsed() -> None:
    assert extract_json_object('```json\n{"answer": "x", "used": []}\n```') == {
        "answer": "x",
        "used": [],
    }


def test_surrounding_prose_is_tolerated() -> None:
    text = 'Here you go:\n{"answer": "x", "used": ["S1"]}\nHope that helps.'

    assert extract_json_object(text) == {"answer": "x", "used": ["S1"]}


@pytest.mark.parametrize("text", ["", "no braces here", "}{", "{not json}", "[1, 2]", '"str"'])
def test_non_objects_are_rejected(text: str) -> None:
    with pytest.raises(InvalidGenerationOutputError):
        extract_json_object(text)


def test_rejection_messages_never_quote_the_output() -> None:
    private = "ZZQ-secret-memory-text"

    with pytest.raises(InvalidGenerationOutputError) as raised:
        extract_json_object(f"{{oops {private}")

    assert private not in str(raised.value)


# --------------------------------------------------------------------------
# Full validation
# --------------------------------------------------------------------------
def test_valid_output_becomes_the_typed_model() -> None:
    parsed = validate_output(result('{"answer": "42", "used": ["S1"]}'), Answer)

    assert parsed == Answer(answer="42", used=["S1"])


def test_reasoning_is_stripped_before_parsing() -> None:
    text = '<think>hmm</think>{"answer": "42", "used": []}'

    assert validate_output(result(text), Answer).answer == "42"


@pytest.mark.parametrize("finish_reason", ["length", "max_tokens"])
def test_a_truncated_completion_is_invalid_even_if_it_parses(finish_reason: str) -> None:
    with pytest.raises(InvalidGenerationOutputError, match="truncated"):
        validate_output(result('{"answer": "42", "used": []}', finish_reason), Answer)


def test_empty_output_is_invalid() -> None:
    with pytest.raises(InvalidGenerationOutputError, match="empty"):
        validate_output(result("<think>only thoughts</think>"), Answer)


def test_a_missing_field_is_invalid_and_named_without_values() -> None:
    with pytest.raises(InvalidGenerationOutputError) as raised:
        validate_output(result('{"answer": "ZZQ-private"}'), Answer)

    assert "used" in str(raised.value)
    assert "ZZQ-private" not in str(raised.value)


def test_an_extra_field_is_invalid_when_the_schema_forbids_it() -> None:
    with pytest.raises(InvalidGenerationOutputError):
        validate_output(result('{"answer": "x", "used": [], "confidence": 0.9}'), Answer)


def test_a_wrong_type_is_invalid() -> None:
    with pytest.raises(InvalidGenerationOutputError):
        validate_output(result('{"answer": "x", "used": "S1"}'), Answer)
