"""Turning raw model text into validated, typed output.

Every generated answer passes through here before it can become a response.
The rules (contract section 21.9):

- a truncated completion (``finish_reason == "length"``) is invalid output,
  not a partial success;
- reasoning traces are stripped and never surface; a thinking model may wrap
  them in ``<think>`` tags inside the content;
- the JSON object is located leniently (code fences and surrounding prose are
  tolerated) but validated strictly against the caller's schema;
- anything that fails is :class:`InvalidGenerationOutputError`, which the
  transport layer maps to ``502 AI_INVALID_RESPONSE``.

Error messages describe structure only. The model's text never appears in
them, because they reach logs.
"""

import json
import re

from pydantic import BaseModel, ValidationError

from memovo_ai.generation.models import GenerationResult, InvalidGenerationOutputError

__all__ = ["extract_json_object", "strip_reasoning", "validate_output"]

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
#: An unterminated block: the model ran out of budget mid-thought.
_OPEN_THINK = re.compile(r"<think>.*\Z", re.IGNORECASE | re.DOTALL)
_CODE_FENCE = re.compile(r"^```[a-zA-Z0-9_-]*\s*|\s*```$", re.MULTILINE)

#: Completion states that mean the text is not the whole answer.
_TRUNCATED_FINISH_REASONS = frozenset({"length", "max_tokens"})


def strip_reasoning(text: str) -> str:
    """Remove ``<think>`` traces, closed or not, and trim."""
    without_closed = _THINK_BLOCK.sub("", text)
    without_open = _OPEN_THINK.sub("", without_closed)

    return without_open.strip()


def extract_json_object(text: str) -> dict[str, object]:
    """Find and parse the single JSON object in ``text``.

    Tolerates a Markdown code fence and prose before or after the object;
    rejects anything that is not exactly one object.

    Raises:
        InvalidGenerationOutputError: if no object can be parsed.
    """
    candidate = _CODE_FENCE.sub("", text).strip()

    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end == -1 or end < start:
        message = "generated output contains no JSON object"
        raise InvalidGenerationOutputError(message)

    try:
        parsed = json.loads(candidate[start : end + 1])
    except json.JSONDecodeError as error:
        message = "generated output is not valid JSON"
        raise InvalidGenerationOutputError(message) from error

    if not isinstance(parsed, dict):
        message = f"generated output is a JSON {type(parsed).__name__}, expected an object"
        raise InvalidGenerationOutputError(message)

    return parsed


def validate_output[OutputT: BaseModel](result: GenerationResult, schema: type[OutputT]) -> OutputT:
    """Validate a completion against ``schema``.

    Raises:
        InvalidGenerationOutputError: if the completion was truncated, empty,
            not a JSON object, or does not satisfy ``schema``.
    """
    if result.finish_reason in _TRUNCATED_FINISH_REASONS:
        message = "generated output was truncated by the output token limit"
        raise InvalidGenerationOutputError(message)

    text = strip_reasoning(result.text)
    if not text:
        message = "generated output is empty"
        raise InvalidGenerationOutputError(message)

    payload = extract_json_object(text)

    try:
        return schema.model_validate(payload)
    except ValidationError as error:
        # Field locations only: the values are model output and may quote
        # memory content.
        fields = sorted(
            {".".join(str(part) for part in e.get("loc", ())) or "<root>" for e in error.errors()}
        )
        message = f"generated output does not match the expected schema at: {', '.join(fields)}"
        raise InvalidGenerationOutputError(message) from None
