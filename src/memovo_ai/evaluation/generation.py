"""Generation evaluation: fixed-evidence chat and note preparation.

Retrieval quality has its own harness. This one holds retrieval constant --
each chat case pins the memories the index returns, at a fixed score -- so
what is measured is generation: does the answer stay inside the evidence,
does it cite what it used, does it abstain when it should, does a prepared
note keep every fact.

Automated checks here are **structural and lexical**: ``found`` correctness,
source membership, presence of tokens a correct answer must contain, absence
of tokens it must not. They cannot prove a claim is supported. Every run
therefore also writes the answers out for human review against the rubric
each case carries (plan P7). A model judge, if added later, is supplementary
to that review, not a replacement.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from memovo_ai.core.config import GenerationSettings, SearchSettings
from memovo_ai.embeddings.models import EMBEDDING_DIMENSION, Embedding
from memovo_ai.generation.base import GenerationProvider
from memovo_ai.generation.models import GenerationError, InvalidGenerationOutputError
from memovo_ai.providers.vector_search.fake import FakeVectorSearchProvider, VectorRecord
from memovo_ai.schemas.chat import ChatMemoriesRequest
from memovo_ai.schemas.prepare_note import PrepareNoteRequest
from memovo_ai.services.memory_chat import MemoryChatService
from memovo_ai.services.memory_search import MemorySearchService
from memovo_ai.services.note_preparation import NotePreparationService

__all__ = [
    "ChatCase",
    "ChatOutcome",
    "ChatSummary",
    "NoteCase",
    "NoteOutcome",
    "NoteSummary",
    "evaluate_chat",
    "evaluate_notes",
    "load_chat_cases",
    "load_note_cases",
    "summarize_chat",
    "summarize_notes",
]

EVAL_USER = "eval_user"
#: Fixed retrieval score for pinned evidence: comfortably above the locked
#: 0.75 threshold, so retrieval never decides a chat case.
PINNED_SCORE = 0.9

#: Words a clear abstention or clarification tends to contain. Lexical, so
#: only a hint for the summary; the rubric review is what counts.
_ABSTENTION_MARKERS = (
    "not enough",
    "don't contain",
    "do not contain",
    "doesn't mention",
    "does not mention",
    "doesn't say",
    "does not say",
    "no information",
    "not specify",
    "which one",
    "which plan",
    "clarify",
    "could you",
    "لا تذكر",
    "لا تحتوي",
    "مش موجود",
    "غير موجود",
    "تقصد",
    "أي واحد",
)


class _ConstantEmbeddings:
    """Retrieval is pinned by score, so the query vector is irrelevant."""

    async def embed_documents(self, texts: Sequence[str]) -> list[Embedding]:
        return [[0.1] * EMBEDDING_DIMENSION for _ in texts]

    async def embed_query(self, query: str) -> Embedding:
        return [0.1] * EMBEDDING_DIMENSION


@dataclass(frozen=True, slots=True)
class ChatCase:
    case_id: str
    kind: str
    memories: tuple[dict[str, str], ...]
    message: str
    history: tuple[dict[str, str], ...]
    expected: dict[str, object]
    rubric: str


@dataclass(frozen=True, slots=True)
class NoteCase:
    case_id: str
    content: str
    must_preserve: tuple[str, ...]
    must_not_add: tuple[str, ...]
    title_must_not_contain: tuple[str, ...]
    rubric: str


@dataclass(slots=True)
class ChatOutcome:
    case_id: str
    kind: str
    error: str | None = None
    found: bool | None = None
    answer: str = ""
    sources: tuple[str, ...] = ()
    found_correct: bool = False
    sources_ok: bool = False
    mentions_ok: bool = False
    forbidden_hit: bool = False
    abstention_marker: bool = False
    checks: list[str] = field(default_factory=list)


@dataclass(slots=True)
class NoteOutcome:
    case_id: str
    error: str | None = None
    title: str = ""
    content: str = ""
    preserved: bool = False
    missing: tuple[str, ...] = ()
    added: tuple[str, ...] = ()
    title_ok: bool = True


@dataclass(frozen=True, slots=True)
class ChatSummary:
    cases: int
    errors: int
    invalid_output: int
    found_accuracy: float
    sources_ok_rate: float
    mentions_ok_rate: float
    forbidden_hit_rate: float
    abstention_marker_rate: float


@dataclass(frozen=True, slots=True)
class NoteSummary:
    cases: int
    errors: int
    invalid_output: int
    preservation_rate: float
    addition_rate: float
    title_ok_rate: float


def load_chat_cases(path: Path) -> list[ChatCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))

    return [
        ChatCase(
            case_id=str(c["id"]),
            kind=str(c["kind"]),
            memories=tuple(dict(m) for m in c["memories"]),
            message=str(c["message"]),
            history=tuple(dict(h) for h in c.get("history", [])),
            expected=dict(c["expected"]),
            rubric=str(c.get("rubric", "")),
        )
        for c in raw["cases"]
    ]


def load_note_cases(path: Path) -> list[NoteCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))

    return [
        NoteCase(
            case_id=str(c["id"]),
            content=str(c["content"]),
            must_preserve=tuple(c.get("must_preserve", [])),
            must_not_add=tuple(c.get("must_not_add", [])),
            title_must_not_contain=tuple(c.get("title_must_not_contain", [])),
            rubric=str(c.get("rubric", "")),
        )
        for c in raw["cases"]
    ]


def _records(case: ChatCase) -> list[VectorRecord]:
    return [
        VectorRecord(
            user_id=EVAL_USER,
            memory_id=memory["memoryId"],
            chunk_id=f"{memory['memoryId']}_chunk_0",
            chunk_index=0,
            content=memory["content"],
            title=memory["title"],
            tags=(),
            score=PINNED_SCORE,
        )
        for memory in case.memories
    ]


def _contains(text: str, token: str) -> bool:
    return token.casefold() in text.casefold()


def _strings(expected: dict[str, object], key: str) -> list[str]:
    value = expected.get(key, [])
    return [str(item) for item in value] if isinstance(value, list) else []


def _check(outcome: ChatOutcome, expected: dict[str, object]) -> None:
    answer = outcome.answer
    expected_found = expected.get("found")
    outcome.found_correct = expected_found is None or outcome.found == expected_found

    include = _strings(expected, "sources_include")
    outcome.sources_ok = all(m in outcome.sources for m in include)
    if include and not outcome.sources_ok:
        outcome.checks.append(f"missing sources: {sorted(set(include) - set(outcome.sources))}")

    any_of = _strings(expected, "answer_mentions_any")
    all_of = _strings(expected, "answer_mentions_all")
    outcome.mentions_ok = (not any_of or any(_contains(answer, t) for t in any_of)) and all(
        _contains(answer, t) for t in all_of
    )
    if not outcome.mentions_ok:
        outcome.checks.append("expected mention missing")

    forbidden = _strings(expected, "answer_must_not_mention")
    hits = [t for t in forbidden if _contains(answer, t)]
    outcome.forbidden_hit = bool(hits)
    if hits:
        outcome.checks.append(f"forbidden mention: {hits}")

    outcome.abstention_marker = any(_contains(answer, m) for m in _ABSTENTION_MARKERS)
    wants_abstention = bool(expected.get("abstains") or expected.get("abstains_or_clarifies"))
    if wants_abstention and not outcome.abstention_marker:
        outcome.checks.append("no abstention/clarification marker (review)")


async def evaluate_chat(
    cases: Sequence[ChatCase],
    *,
    generation: GenerationProvider,
    settings: GenerationSettings | None = None,
) -> list[ChatOutcome]:
    """Run every case through the real chat service with pinned evidence."""
    resolved = settings if settings is not None else GenerationSettings()
    outcomes: list[ChatOutcome] = []

    for case in cases:
        search = MemorySearchService(
            embedding_provider=_ConstantEmbeddings(),
            vector_search=FakeVectorSearchProvider(_records(case)),
            settings=SearchSettings(_env_file=None),
        )
        service = MemoryChatService(search=search, generation=generation, settings=resolved)
        outcome = ChatOutcome(case_id=case.case_id, kind=case.kind)

        payload: dict[str, object] = {"userId": EVAL_USER, "message": case.message}
        if case.history:
            payload["history"] = list(case.history)

        try:
            response = await service.chat(ChatMemoriesRequest.model_validate(payload))
        except InvalidGenerationOutputError as error:
            outcome.error = f"invalid_output: {error}"
        except GenerationError as error:
            outcome.error = f"{type(error).__name__}"
        else:
            outcome.found = response.found
            outcome.answer = response.answer
            outcome.sources = tuple(source.memory_id for source in response.sources)
            _check(outcome, case.expected)

        outcomes.append(outcome)

    return outcomes


async def evaluate_notes(
    cases: Sequence[NoteCase],
    *,
    generation: GenerationProvider,
    settings: GenerationSettings | None = None,
) -> list[NoteOutcome]:
    resolved = settings if settings is not None else GenerationSettings()
    service = NotePreparationService(generation=generation, settings=resolved)
    outcomes: list[NoteOutcome] = []

    for case in cases:
        outcome = NoteOutcome(case_id=case.case_id)
        try:
            response = await service.prepare(PrepareNoteRequest(content=case.content))
        except InvalidGenerationOutputError as error:
            outcome.error = f"invalid_output: {error}"
        except GenerationError as error:
            outcome.error = f"{type(error).__name__}"
        else:
            outcome.title = response.title
            outcome.content = response.content
            combined = f"{response.title}\n{response.content}"
            outcome.missing = tuple(t for t in case.must_preserve if not _contains(combined, t))
            outcome.added = tuple(t for t in case.must_not_add if _contains(combined, t))
            outcome.preserved = not outcome.missing
            outcome.title_ok = not any(
                _contains(response.title, t) for t in case.title_must_not_contain
            )

        outcomes.append(outcome)

    return outcomes


def _rate(count: int, total: int) -> float:
    return round(count / total, 3) if total else 0.0


def summarize_chat(outcomes: Sequence[ChatOutcome]) -> ChatSummary:
    answered = [o for o in outcomes if o.error is None]

    return ChatSummary(
        cases=len(outcomes),
        errors=sum(1 for o in outcomes if o.error is not None),
        invalid_output=sum(1 for o in outcomes if (o.error or "").startswith("invalid_output")),
        found_accuracy=_rate(sum(1 for o in answered if o.found_correct), len(outcomes)),
        sources_ok_rate=_rate(sum(1 for o in answered if o.sources_ok), len(outcomes)),
        mentions_ok_rate=_rate(sum(1 for o in answered if o.mentions_ok), len(outcomes)),
        forbidden_hit_rate=_rate(sum(1 for o in answered if o.forbidden_hit), len(outcomes)),
        abstention_marker_rate=_rate(
            sum(1 for o in answered if o.abstention_marker), len(outcomes)
        ),
    )


def summarize_notes(outcomes: Sequence[NoteOutcome]) -> NoteSummary:
    prepared = [o for o in outcomes if o.error is None]

    return NoteSummary(
        cases=len(outcomes),
        errors=sum(1 for o in outcomes if o.error is not None),
        invalid_output=sum(1 for o in outcomes if (o.error or "").startswith("invalid_output")),
        preservation_rate=_rate(sum(1 for o in prepared if o.preserved), len(outcomes)),
        addition_rate=_rate(sum(1 for o in prepared if o.added), len(outcomes)),
        title_ok_rate=_rate(sum(1 for o in prepared if o.title_ok), len(outcomes)),
    )
