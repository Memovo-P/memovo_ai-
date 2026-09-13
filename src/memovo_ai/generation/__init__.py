"""Answer generation: provider boundary, deterministic fake, output validation.

Chat and note preparation share a provider but keep separate prompts and
output schemas. Nothing in this package knows about HTTP; the OpenRouter
adapter lives in :mod:`memovo_ai.providers.generation`.
"""

from memovo_ai.generation.base import BoundedGenerationProvider, GenerationProvider
from memovo_ai.generation.fake import FakeGenerationProvider, ScriptedResponse
from memovo_ai.generation.models import (
    GenerationDisabledError,
    GenerationError,
    GenerationMessage,
    GenerationRateLimitedError,
    GenerationRequest,
    GenerationResult,
    GenerationTimeoutError,
    GenerationUnavailableError,
    InvalidGenerationOutputError,
)
from memovo_ai.generation.output import extract_json_object, strip_reasoning, validate_output

__all__ = [
    "BoundedGenerationProvider",
    "FakeGenerationProvider",
    "GenerationDisabledError",
    "GenerationError",
    "GenerationMessage",
    "GenerationProvider",
    "GenerationRateLimitedError",
    "GenerationRequest",
    "GenerationResult",
    "GenerationTimeoutError",
    "GenerationUnavailableError",
    "InvalidGenerationOutputError",
    "ScriptedResponse",
    "extract_json_object",
    "strip_reasoning",
    "validate_output",
]
