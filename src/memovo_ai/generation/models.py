"""Generation value types and the failures a provider can raise.

Privacy
-------
Messages carried by these exceptions describe *what kind* of thing went wrong
-- a status class, a missing field -- and never carry prompt text, model
output, provider error bodies or credentials. They reach logs, and the
privacy rules forbid all of those there.
"""

from dataclasses import dataclass
from typing import Literal

__all__ = [
    "GenerationDisabledError",
    "GenerationError",
    "GenerationMessage",
    "GenerationRateLimitedError",
    "GenerationRequest",
    "GenerationResult",
    "GenerationTimeoutError",
    "GenerationUnavailableError",
    "InvalidGenerationOutputError",
]

type MessageRole = Literal["system", "user", "assistant"]


@dataclass(frozen=True, slots=True)
class GenerationMessage:
    """One chat-completion message."""

    role: MessageRole
    content: str


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    """What the provider is asked to complete.

    ``temperature`` defaults to 0 so runs are as repeatable as the provider
    allows; ``max_output_tokens`` is a hard cap the adapter forwards verbatim.
    """

    messages: tuple[GenerationMessage, ...]
    max_output_tokens: int
    temperature: float = 0.0

    def __post_init__(self) -> None:
        if not self.messages:
            message = "a generation request needs at least one message"
            raise ValueError(message)
        if self.max_output_tokens < 1:
            message = f"max_output_tokens must be at least 1, got {self.max_output_tokens}"
            raise ValueError(message)


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """The provider's answer, already stripped of any reasoning trace.

    ``finish_reason`` is the provider's own value (``stop``, ``length`` ...),
    kept so output validation can treat a truncated completion as invalid.
    Token counts are ``None`` when the provider did not report usage.
    """

    text: str
    finish_reason: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class GenerationError(Exception):
    """Base class for generation failures.

    The transport layer maps subclasses onto the public error contract.
    Nothing here decides an HTTP status.
    """


class GenerationUnavailableError(GenerationError):
    """The provider could not answer: outage, network failure, upstream 5xx,
    authentication or credit problems, or an unknown model.

    Public mapping: ``503 GENERATION_UNAVAILABLE``, retryable.
    """


class GenerationDisabledError(GenerationUnavailableError):
    """Generation is switched off or not configured for this deployment."""


class GenerationRateLimitedError(GenerationError):
    """The provider answered 429.

    Public mapping: ``429 RATE_LIMITED``, retryable, with the provider's
    ``Retry-After`` forwarded once bounded.
    """

    def __init__(self, message: str, *, retry_after_seconds: int | None = None) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class GenerationTimeoutError(GenerationError, TimeoutError):
    """The provider did not answer within the deadline.

    Public mapping: ``504 TIMEOUT``, retryable.
    """


class InvalidGenerationOutputError(GenerationError):
    """The provider answered, but not with usable output.

    Malformed response body, missing or empty content, a truncated
    completion, or text that fails the expected output schema. Public
    mapping: ``502 AI_INVALID_RESPONSE``, **never** retryable (contract
    section 21.9).
    """
