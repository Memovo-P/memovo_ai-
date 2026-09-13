"""Opt-in synthetic smoke test against the real OpenRouter endpoint.

Skipped unless ``MEMOVO_RUN_OPENROUTER_SMOKE=1`` **and** a key is configured
through ``MEMOVO_OPENROUTER_API_KEY`` (or the local ``.env``). Ordinary CI
never runs it. Content is synthetic; no memory or user data is sent.

What it proves when it passes: the approved model is served, accepts the
request shape the adapter sends, and returns a completion that survives
output validation. It says nothing about answer quality -- that is P7 --
and a catalog listing would prove nothing at all, so none is consulted.

The key is never printed. Run with ``-s`` to see the capability summary.
"""

import os

import pytest
from pydantic import BaseModel, ConfigDict

from memovo_ai.core.config import GenerationSettings, OpenRouterSettings
from memovo_ai.generation import GenerationMessage, GenerationRequest, validate_output
from memovo_ai.providers.generation import OpenRouterGenerationProvider

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("MEMOVO_RUN_OPENROUTER_SMOKE") != "1",
        reason="opt-in: set MEMOVO_RUN_OPENROUTER_SMOKE=1 to call OpenRouter",
    ),
]


class Echo(BaseModel):
    model_config = ConfigDict(extra="ignore")

    echo: str


async def test_the_approved_model_completes_a_synthetic_request() -> None:
    credentials = OpenRouterSettings()
    if not credentials.is_configured:
        pytest.skip("MEMOVO_OPENROUTER_API_KEY is not configured")

    settings = GenerationSettings()
    provider = OpenRouterGenerationProvider.connect(
        settings=credentials,
        model=settings.model,
        timeout_seconds=settings.timeout_seconds,
        retry_after_max_seconds=settings.retry_after_max_seconds,
    )
    try:
        result = await provider.generate(
            GenerationRequest(
                messages=(
                    GenerationMessage(
                        role="system",
                        content=(
                            "Reply with exactly one JSON object of the form "
                            '{"echo": "<word>"} and nothing else. Do not think aloud.'
                        ),
                    ),
                    GenerationMessage(role="user", content="The word is: pineapple"),
                ),
                max_output_tokens=min(settings.chat_max_output_tokens, 512),
            )
        )
    finally:
        await provider.aclose()

    parsed = validate_output(result, Echo)

    print(
        "\nOpenRouter smoke:",
        f"model={result.model}",
        f"finish_reason={result.finish_reason}",
        f"prompt_tokens={result.prompt_tokens}",
        f"completion_tokens={result.completion_tokens}",
        f"think_tags_in_content={'<think>' in result.text.lower()}",
    )

    # OpenRouter may omit the routing-only :free suffix in response metadata.
    assert result.model.removesuffix(":free") == settings.model.removesuffix(":free"), (
        "a different model answered"
    )
    assert parsed.echo.strip().lower() == "pineapple"
