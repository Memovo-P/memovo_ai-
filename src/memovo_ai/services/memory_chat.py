"""Memory Chat orchestration (contract v1.9, section 17).

The pipeline::

    ChatMemoriesRequest
      -> retrieval query (the message, contextualized by history when short)
      -> the same user-scoped retrieval /ai/memories/search uses
      -> no evidence: found=false, fixed answer, no generation call
      -> evidence block with request-local handles S1..Sn
      -> context budget check (nothing is trimmed to fit)
      -> generation
      -> output validation: JSON shape, handles that exist, no markers
      -> found=true, answer, sources = the memories actually used

History is context for reading the question, never evidence: every turn
retrieves afresh with the current user's filter, and it is passed to the
model whole -- the Backend bounded it, and this service never trims it
(section 17.2.1). ``conversationId`` is accepted and ignored (17.2.2).

What this service must never do: write anything, fetch anything, call a
tool, fall back to general knowledge, or claim ``found: false`` when
relevant memories exist.
"""

import logging
import re
from collections.abc import Sequence

from memovo_ai.core.config import GenerationSettings, LoggingSettings
from memovo_ai.core.errors import AiServiceError
from memovo_ai.core.logging import hash_user_id, log_event, timed
from memovo_ai.generation.base import GenerationProvider
from memovo_ai.generation.models import (
    GenerationDisabledError,
    GenerationMessage,
    GenerationRequest,
    InvalidGenerationOutputError,
)
from memovo_ai.generation.output import validate_output
from memovo_ai.generation.prompts import (
    CHAT_SYSTEM_PROMPT,
    ChatModelOutput,
    EvidenceItem,
    chat_user_turn,
)
from memovo_ai.retrieval.models import RankedMemory
from memovo_ai.schemas.chat import (
    ChatMemoriesRequest,
    ChatMemoriesResponse,
    ChatSource,
    ChatSourceChunk,
    HistoryMessage,
)
from memovo_ai.schemas.errors import ErrorCode
from memovo_ai.services.memory_search import MemorySearchService

__all__ = ["NO_MATCH_ANSWER", "MemoryChatService", "retrieval_query"]

_logger = logging.getLogger(__name__)

#: The fixed answer when retrieval finds nothing (section 17.3: the answer
#: "may indicate no memories found"). The Backend keys on ``found``.
NO_MATCH_ANSWER = "I couldn't find any relevant memories for that."

#: A message this short is treated as a possible follow-up and retrieved
#: together with the previous user turn. Bounded, deterministic, and cheap;
#: its recall/precision trade-off is measured in P7, not assumed here.
SHORT_MESSAGE_WORDS = 8

#: Internal handle markers the model was told never to write. Removed if it
#: does anyway; the ``sources`` array is the only linkage (section 17.4.2).
_HANDLE_MARKER = re.compile(r"\s?[\[(]\s*S\d+(?:\s*,\s*S\d+)*\s*[\])]")


def retrieval_query(message: str, history: Sequence[HistoryMessage] | None) -> str:
    """What to embed for retrieval.

    The current message alone, unless it is short enough to be a follow-up
    ("and the second one?") and there is a previous user turn to anchor it,
    in which case that turn is appended. Nothing is rewritten and no model
    is called, so no fact can be introduced here.
    """
    if not history or len(message.split()) > SHORT_MESSAGE_WORDS:
        return message

    previous = next((turn.content for turn in reversed(history) if turn.role == "user"), None)
    if previous is None or not previous.strip():
        return message

    return f"{message}\n{previous}"


def _strip_handles(answer: str) -> str:
    return _HANDLE_MARKER.sub("", answer).strip()


class MemoryChatService:
    """Answers ``POST /ai/chat/memories``."""

    __slots__ = ("_generation", "_search", "_settings", "_user_salt")

    def __init__(
        self,
        *,
        search: MemorySearchService,
        generation: GenerationProvider | None,
        settings: GenerationSettings | None = None,
        logging_settings: LoggingSettings | None = None,
    ) -> None:
        self._search = search
        self._generation = generation
        self._settings = settings if settings is not None else GenerationSettings()

        resolved_logging = logging_settings if logging_settings is not None else LoggingSettings()
        self._user_salt = resolved_logging.user_salt.get_secret_value()

    async def chat(self, request: ChatMemoriesRequest) -> ChatMemoriesResponse:
        if self._generation is None:
            message = "answer generation is not enabled"
            raise GenerationDisabledError(message)

        history = list(request.history or ())
        query = retrieval_query(request.message, history)

        ranked = await self._search.retrieve(user_id=request.user_id, query=query)

        if not ranked:
            # No evidence: no answer generation. The rewrite above made no
            # model call either, so this outcome costs zero generation.
            self._log(request, history_count=len(history), evidence_count=0, source_count=0)
            return ChatMemoriesResponse(found=False, answer=NO_MATCH_ANSWER, sources=[])

        evidence = [
            EvidenceItem(
                handle=f"S{index}",
                title=memory.title,
                chunks=tuple(chunk.content for chunk in memory.chunks),
            )
            for index, memory in enumerate(ranked, start=1)
        ]
        by_handle = {item.handle: memory for item, memory in zip(evidence, ranked, strict=True)}

        messages = (
            GenerationMessage(role="system", content=CHAT_SYSTEM_PROMPT),
            *(GenerationMessage(role=turn.role, content=turn.content) for turn in history),
            GenerationMessage(
                role="user", content=chat_user_turn(question=request.message, evidence=evidence)
            ),
        )
        self._check_budget(messages, self._settings.chat_max_output_tokens)

        with timed() as elapsed:
            result = await self._generation.generate(
                GenerationRequest(
                    messages=messages, max_output_tokens=self._settings.chat_max_output_tokens
                )
            )

        output = validate_output(result, ChatModelOutput)

        used: list[str] = []
        for handle in output.used_sources:
            if handle not in by_handle:
                # The model named evidence it was never given. Not a source
                # to invent -- malformed output (section 21.9).
                message = "generated output references a source that was not provided"
                raise InvalidGenerationOutputError(message)
            if handle not in used:
                used.append(handle)

        answer = _strip_handles(output.answer)
        if not answer:
            message = "generated answer is empty after removing citation markers"
            raise InvalidGenerationOutputError(message)

        sources = [_to_source(by_handle[handle]) for handle in used]

        self._log(
            request,
            history_count=len(history),
            evidence_count=len(evidence),
            source_count=len(sources),
            generation_ms=elapsed.ms,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            finish_reason=result.finish_reason,
        )

        return ChatMemoriesResponse(found=True, answer=answer, sources=sources)

    def _check_budget(self, messages: Sequence[GenerationMessage], reserve: int) -> None:
        """Refuse what cannot fit, rather than trimming history or evidence.

        A conservative character estimate stands in for the generator's
        tokenizer, which is not available in-process. Over budget is a
        deterministic, non-retryable condition on this request.
        """
        capacity_tokens = self._settings.context_window_tokens - reserve
        capacity_chars = int(capacity_tokens * self._settings.chars_per_token)
        used_chars = sum(len(m.content) for m in messages)

        if used_chars > capacity_chars:
            raise AiServiceError(
                ErrorCode.INVALID_INPUT,
                message="Request exceeds the generation context budget",
            )

    def _log(
        self,
        request: ChatMemoriesRequest,
        *,
        history_count: int,
        evidence_count: int,
        source_count: int,
        generation_ms: float | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        finish_reason: str | None = None,
    ) -> None:
        """Counts, durations and codes. Never the message, history, evidence
        or answer. Generation fields are ``None`` on the no-match path, where
        no model was called."""
        log_event(
            _logger,
            "chat.answered",
            user=hash_user_id(request.user_id, salt=self._user_salt),
            query_chars=len(request.message),
            history_count=history_count,
            evidence_count=evidence_count,
            source_count=source_count,
            matched=evidence_count > 0,
            generation_ms=generation_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason=finish_reason,
        )


def _to_source(memory: RankedMemory) -> ChatSource:
    """Reconstruct a source from retrieval, never from generated metadata."""
    return ChatSource(
        memoryId=memory.memory_id,
        score=memory.score,
        title=memory.title,
        chunks=[
            ChatSourceChunk(chunkId=chunk.chunk_id, content=chunk.content)
            for chunk in memory.chunks
        ],
    )
