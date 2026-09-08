"""Embedding configuration."""

import pytest

from memovo_ai.core.config import DEFAULT_EMBEDDING_MODEL, EmbeddingSettings
from memovo_ai.embeddings import EMBEDDING_DIMENSION

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _clear_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Settings must be read from a known state, not the developer's shell."""
    for name in (
        "MEMOVO_EMBEDDING_MODEL",
        "MEMOVO_EMBEDDING_DIMENSION",
        "MEMOVO_EMBEDDING_DEVICE",
        "MEMOVO_EMBEDDING_BATCH_SIZE",
        "MEMOVO_EMBEDDING_QUERY_PROMPT_NAME",
    ):
        monkeypatch.delenv(name, raising=False)


def test_defaults_match_the_sprint_1_decisions() -> None:
    config = EmbeddingSettings(_env_file=None)  # type: ignore[call-arg]

    assert config.model == DEFAULT_EMBEDDING_MODEL == "Qwen/Qwen3-Embedding-0.6B"
    assert config.dimension == EMBEDDING_DIMENSION == 1024
    assert config.device == "auto"
    assert config.batch_size == 32


def test_query_prompt_is_disabled_by_default() -> None:
    assert EmbeddingSettings(_env_file=None).query_prompt_name == ""  # type: ignore[call-arg]


@pytest.mark.parametrize(
    ("variable", "value", "attribute", "expected"),
    [
        ("MEMOVO_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-4B", "model", "Qwen/Qwen3-Embedding-4B"),
        ("MEMOVO_EMBEDDING_DIMENSION", "2560", "dimension", 2560),
        ("MEMOVO_EMBEDDING_DEVICE", "cuda:0", "device", "cuda:0"),
        ("MEMOVO_EMBEDDING_BATCH_SIZE", "64", "batch_size", 64),
        ("MEMOVO_EMBEDDING_QUERY_PROMPT_NAME", "query", "query_prompt_name", "query"),
    ],
)
def test_environment_overrides_are_applied(
    monkeypatch: pytest.MonkeyPatch,
    variable: str,
    value: str,
    attribute: str,
    expected: object,
) -> None:
    monkeypatch.setenv(variable, value)

    assert getattr(EmbeddingSettings(_env_file=None), attribute) == expected  # type: ignore[call-arg]


def test_auto_device_resolves_to_none() -> None:
    """`None` lets the runtime pick CUDA or CPU."""
    assert EmbeddingSettings(_env_file=None, device="auto").resolved_device is None  # type: ignore[call-arg]


def test_explicit_device_is_passed_through() -> None:
    assert EmbeddingSettings(_env_file=None, device="cpu").resolved_device == "cpu"  # type: ignore[call-arg]


@pytest.mark.parametrize(("field", "value"), [("dimension", 0), ("batch_size", 0)])
def test_non_positive_values_are_rejected(field: str, value: int) -> None:
    with pytest.raises(ValueError, match="greater than or equal to 1"):
        EmbeddingSettings(_env_file=None, **{field: value})  # type: ignore[call-arg]


def test_settings_are_frozen() -> None:
    """Configuration is read once at startup, not mutated per request."""
    config = EmbeddingSettings(_env_file=None)  # type: ignore[call-arg]

    with pytest.raises(ValueError, match="frozen"):
        config.batch_size = 64  # type: ignore[misc]


def test_config_dimension_agrees_with_the_embeddings_layer() -> None:
    """`core` cannot import `embeddings`, so the locked value is duplicated.

    This guards the duplication: the two must never drift apart.
    """
    from memovo_ai.core.config import DEFAULT_EMBEDDING_DIMENSION

    assert DEFAULT_EMBEDDING_DIMENSION == EMBEDDING_DIMENSION


@pytest.mark.parametrize(
    "module",
    ["memovo_ai.core.config", "memovo_ai.embeddings", "memovo_ai.embeddings.qwen3"],
)
def test_module_imports_cleanly_from_a_cold_interpreter(module: str) -> None:
    """Guards against a circular import between `core.config` and `embeddings`.

    Import order decides whether such a cycle raises, so importing each entry
    point first, in its own interpreter, is the only reliable check.
    """
    import subprocess
    import sys

    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", f"import {module}"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
