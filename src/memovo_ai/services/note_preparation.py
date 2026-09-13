"""Explicit note preparation (contract v1.9, section 6).

Raw content in, ``title`` + cleaned ``content`` out. No retrieval, no
conversation, no identity, no persistence: the Backend creates the Note only
after validating what comes back, then queues ordinary processing.

Output bounds are the Note limits the Backend will apply when it creates the
Note (section 8.1: title 0-200, content 1-10,000). A preparation that
breaks them would be rejected downstream anyway, so it is reported here as
malformed output rather than handed on.
"""

import logging

from memovo_ai.core.config import GenerationSettings
from memovo_ai.core.errors import AiServiceError
from memovo_ai.core.logging import log_event, timed
from memovo_ai.generation.base import GenerationProvider
from memovo_ai.generation.models import (
    GenerationDisabledError,
    GenerationMessage,
    GenerationRequest,
    InvalidGenerationOutputError,
)
from memovo_ai.generation.output import validate_output
from memovo_ai.generation.prompts import NOTE_SYSTEM_PROMPT, NoteModelOutput, note_user_turn
from memovo_ai.schemas.errors import ErrorCode
from memovo_ai.schemas.prepare_note import PrepareNoteRequest, PrepareNoteResponse
from memovo_ai.schemas.process import NOTE_CONTENT_MAX_LENGTH, NOTE_TITLE_MAX_LENGTH

__all__ = ["NotePreparationService"]

_logger = logging.getLogger(__name__)


class NotePreparationService:
    """Answers ``POST /ai/memories/prepare-note``."""

    __slots__ = ("_generation", "_settings")

    def __init__(
        self,
        *,
        generation: GenerationProvider | None,
        settings: GenerationSettings | None = None,
    ) -> None:
        self._generation = generation
        self._settings = settings if settings is not None else GenerationSettings()

    async def prepare(self, request: PrepareNoteRequest) -> PrepareNoteResponse:
        if self._generation is None:
            message = "note preparation is not enabled"
            raise GenerationDisabledError(message)

        if not request.content.strip():
            raise AiServiceError(ErrorCode.INVALID_INPUT, message="Note content is empty")

        messages = (
            GenerationMessage(role="system", content=NOTE_SYSTEM_PROMPT),
            GenerationMessage(role="user", content=note_user_turn(request.content)),
        )
        self._check_budget(messages)

        with timed() as elapsed:
            result = await self._generation.generate(
                GenerationRequest(
                    messages=messages, max_output_tokens=self._settings.note_max_output_tokens
                )
            )

        output = validate_output(result, NoteModelOutput)
        title = output.title.strip()
        content = output.content.strip()

        if not title or len(title) > NOTE_TITLE_MAX_LENGTH:
            message = "prepared title is empty or over the Note title limit"
            raise InvalidGenerationOutputError(message)
        if not content or len(content) > NOTE_CONTENT_MAX_LENGTH:
            message = "prepared content is empty or over the Note content limit"
            raise InvalidGenerationOutputError(message)

        log_event(
            _logger,
            "note.prepared",
            content_chars=len(request.content),
            generation_ms=elapsed.ms,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            finish_reason=result.finish_reason,
        )

        return PrepareNoteResponse(title=title, content=content)

    def _check_budget(self, messages: tuple[GenerationMessage, ...]) -> None:
        capacity_tokens = (
            self._settings.context_window_tokens - self._settings.note_max_output_tokens
        )
        capacity_chars = int(capacity_tokens * self._settings.chars_per_token)

        if sum(len(m.content) for m in messages) > capacity_chars:
            raise AiServiceError(
                ErrorCode.INVALID_INPUT,
                message="Request exceeds the generation context budget",
            )
