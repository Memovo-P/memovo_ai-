"""The generation evaluation harness, exercised with scripted models.

No provider is contacted. The datasets are validated for shape, and the
harness arithmetic is checked against a responder whose behaviour is known:
a perfectly grounded model, an abstaining one, a fabricating one, and one
that returns garbage.
"""

import json
import re
from pathlib import Path

import pytest

from memovo_ai.core.config import GenerationSettings
from memovo_ai.evaluation.generation import (
    ChatCase,
    NoteCase,
    evaluate_chat,
    evaluate_notes,
    load_chat_cases,
    load_note_cases,
    summarize_chat,
    summarize_notes,
)
from memovo_ai.generation import FakeGenerationProvider, GenerationRequest

pytestmark = pytest.mark.evaluation

ROOT = Path(__file__).resolve().parents[2]
CHAT_DATASET = ROOT / "eval" / "datasets" / "generation_chat.json"
NOTE_DATASET = ROOT / "eval" / "datasets" / "note_preparation.json"

_HANDLE = re.compile(r"^\[(S\d+)\] Title: .*$", re.MULTILINE)


def settings() -> GenerationSettings:
    return GenerationSettings(_env_file=None)  # type: ignore[call-arg]


def evidence_of(request: GenerationRequest) -> tuple[list[str], str]:
    """Handles and the evidence block text from the final user turn."""
    body = request.messages[-1].content
    return _HANDLE.findall(body), body


def grounded_responder(request: GenerationRequest) -> str:
    """Echoes every evidence chunk and claims every handle: fully grounded."""
    handles, body = evidence_of(request)
    evidence = body.split("<<<MEMORIES>>>")[1].split("<<<END MEMORIES>>>")[0]
    lines = [line for line in evidence.splitlines() if line and not line.startswith(("[", "---"))]
    return json.dumps(
        {"answer": " ".join(lines) or "The memories say nothing.", "usedSources": handles}
    )


def abstaining_responder(request: GenerationRequest) -> str:
    handles, _ = evidence_of(request)
    return json.dumps(
        {
            "answer": "Your memories do not contain enough information to answer that.",
            "usedSources": handles[:1],
        }
    )


def fabricating_responder(request: GenerationRequest) -> str:
    return json.dumps(
        {"answer": "You chose AWS and pay $15,000 with PWNED BANANA.", "usedSources": []}
    )


# --------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------
def test_the_chat_dataset_loads_with_every_expected_kind() -> None:
    cases = load_chat_cases(CHAT_DATASET)

    assert {c.kind for c in cases} >= {
        "grounded",
        "insufficient",
        "conflict",
        "follow_up",
        "injection",
        "no_match",
    }
    assert len({c.case_id for c in cases}) == len(cases)
    assert any(any("؀" <= ch <= "ۿ" for ch in c.message) for c in cases), "no Arabic case"


def test_every_chat_case_has_at_most_top_k_memories_and_a_rubric() -> None:
    for case in load_chat_cases(CHAT_DATASET):
        assert len(case.memories) <= 5, case.case_id
        assert case.rubric, case.case_id
        for memory in case.memories:
            assert set(memory) == {"memoryId", "title", "content"}, case.case_id


def test_the_note_dataset_loads_with_preservation_expectations() -> None:
    cases = load_note_cases(NOTE_DATASET)

    assert len(cases) >= 5
    assert all(c.must_preserve for c in cases)
    assert any(any("؀" <= ch <= "ۿ" for ch in c.content) for c in cases)


def test_no_evaluation_case_carries_real_looking_identifiers() -> None:
    for case in load_chat_cases(CHAT_DATASET):
        for memory in case.memories:
            assert memory["memoryId"].startswith("m_"), memory["memoryId"]


# --------------------------------------------------------------------------
# Harness arithmetic
# --------------------------------------------------------------------------
async def test_a_grounded_model_scores_full_marks_on_found_and_sources() -> None:
    cases = load_chat_cases(CHAT_DATASET)
    outcomes = await evaluate_chat(
        cases, generation=FakeGenerationProvider(responder=grounded_responder), settings=settings()
    )
    summary = summarize_chat(outcomes)

    assert summary.cases == len(cases)
    assert summary.errors == 0
    assert summary.found_accuracy == 1.0
    assert summary.sources_ok_rate == 1.0
    assert summary.forbidden_hit_rate > 0.0, "echoing a hostile memory must register as a hit"


async def test_the_no_match_case_never_calls_the_model() -> None:
    cases = [c for c in load_chat_cases(CHAT_DATASET) if c.kind == "no_match"]
    fake = FakeGenerationProvider(responder=grounded_responder)

    outcomes = await evaluate_chat(cases, generation=fake, settings=settings())

    assert outcomes[0].found is False
    assert outcomes[0].found_correct is True
    assert fake.calls == []


async def test_an_abstaining_model_is_recognised_where_abstention_is_expected() -> None:
    cases = [c for c in load_chat_cases(CHAT_DATASET) if c.kind == "insufficient"]

    outcomes = await evaluate_chat(
        cases,
        generation=FakeGenerationProvider(responder=abstaining_responder),
        settings=settings(),
    )

    assert outcomes[0].found is True
    assert outcomes[0].abstention_marker is True
    assert outcomes[0].forbidden_hit is False
    assert outcomes[0].checks == []


async def test_a_fabricating_model_is_caught_by_the_forbidden_lists() -> None:
    cases = [c for c in load_chat_cases(CHAT_DATASET) if c.kind in {"insufficient", "injection"}]

    outcomes = await evaluate_chat(
        cases,
        generation=FakeGenerationProvider(responder=fabricating_responder),
        settings=settings(),
    )
    summary = summarize_chat(outcomes)

    assert summary.forbidden_hit_rate == 1.0
    assert summary.sources_ok_rate < 1.0


async def test_garbage_output_counts_as_invalid_not_as_an_answer() -> None:
    cases = load_chat_cases(CHAT_DATASET)[:2]

    outcomes = await evaluate_chat(
        cases,
        generation=FakeGenerationProvider(responder=lambda r: "not json"),
        settings=settings(),
    )
    summary = summarize_chat(outcomes)

    assert summary.errors == 2
    assert summary.invalid_output == 2
    assert summary.found_accuracy == 0.0


async def test_a_faithful_note_model_preserves_everything() -> None:
    cases = load_note_cases(NOTE_DATASET)

    def faithful(request: GenerationRequest) -> str:
        raw = request.messages[-1].content.split("<<<CONTENT>>>")[1].split("<<<END CONTENT>>>")[0]
        return json.dumps({"title": "Note", "content": raw.strip()})

    outcomes = await evaluate_notes(
        cases, generation=FakeGenerationProvider(responder=faithful), settings=settings()
    )
    summary = summarize_notes(outcomes)

    assert summary.errors == 0
    assert summary.preservation_rate == 1.0
    assert summary.title_ok_rate == 1.0


async def test_a_lossy_note_model_is_measured() -> None:
    cases = load_note_cases(NOTE_DATASET)
    lossy = FakeGenerationProvider(
        responder=lambda r: json.dumps({"title": "PWNED", "content": "x"})
    )

    outcomes = await evaluate_notes(cases, generation=lossy, settings=settings())
    summary = summarize_notes(outcomes)

    assert summary.preservation_rate == 0.0
    assert summary.title_ok_rate < 1.0
    assert all(o.missing for o in outcomes)


async def test_harness_cases_can_be_built_by_hand() -> None:
    case = ChatCase(
        case_id="x",
        kind="grounded",
        memories=({"memoryId": "m_1", "title": "T", "content": "The sky is blue."},),
        message="colour?",
        history=(),
        expected={"found": True, "sources_include": ["m_1"], "answer_mentions_any": ["blue"]},
        rubric="",
    )
    note = NoteCase(
        case_id="n",
        content="a 1 b",
        must_preserve=("1",),
        must_not_add=(),
        title_must_not_contain=(),
        rubric="",
    )
    fake = FakeGenerationProvider(responder=grounded_responder)

    chat = await evaluate_chat([case], generation=fake, settings=settings())
    notes = await evaluate_notes(
        [note],
        generation=FakeGenerationProvider(
            responder=lambda r: json.dumps({"title": "t", "content": "a 1 b"})
        ),
        settings=settings(),
    )

    assert chat[0].sources_ok and chat[0].mentions_ok
    assert notes[0].preserved
