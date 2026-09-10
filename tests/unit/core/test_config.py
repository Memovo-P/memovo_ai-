"""Embedding configuration."""

import pytest

from memovo_ai.core.config import (
    DEFAULT_EMBEDDING_MODEL,
    EmbeddingSettings,
    SearchSettings,
)
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
        "MEMOVO_SEARCH_TOP_K",
        "MEMOVO_SEARCH_SIMILARITY_THRESHOLD",
    ):
        monkeypatch.delenv(name, raising=False)


def test_defaults_match_the_sprint_1_decisions() -> None:
    config = EmbeddingSettings(_env_file=None)  # type: ignore[call-arg]

    assert config.model == DEFAULT_EMBEDDING_MODEL == "Qwen/Qwen3-Embedding-0.6B"
    assert config.dimension == EMBEDDING_DIMENSION == 1024
    assert config.device == "auto"
    assert config.batch_size == 32


def test_the_query_prompt_is_enabled_by_default() -> None:
    """Enabled after the Phase 06 benchmark confirmed the model defines it
    and that it improves relevant-vs-unrelated separation."""
    assert EmbeddingSettings(_env_file=None).query_prompt_name == "query"  # type: ignore[call-arg]


def test_the_query_prompt_can_still_be_disabled() -> None:
    config = EmbeddingSettings(_env_file=None, query_prompt_name="")  # type: ignore[call-arg]

    assert config.query_prompt_name == ""


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


# --------------------------------------------------------------------------
# Search settings
# --------------------------------------------------------------------------
def test_search_defaults_are_the_locked_mvp_values() -> None:
    """Top-K 5 and threshold 0.75 (doc 01, decisions 9 and 10)."""
    config = SearchSettings(_env_file=None)  # type: ignore[call-arg]

    assert config.top_k == 5
    assert config.similarity_threshold == 0.75


def test_search_config_agrees_with_the_retrieval_layer() -> None:
    """`core` cannot import `retrieval`, so the locked values are duplicated."""
    from memovo_ai.core.config import DEFAULT_SEARCH_SIMILARITY_THRESHOLD
    from memovo_ai.retrieval import DEFAULT_SIMILARITY_THRESHOLD

    assert DEFAULT_SEARCH_SIMILARITY_THRESHOLD == DEFAULT_SIMILARITY_THRESHOLD


@pytest.mark.parametrize(
    ("variable", "value", "attribute", "expected"),
    [
        ("MEMOVO_SEARCH_TOP_K", "10", "top_k", 10),
        ("MEMOVO_SEARCH_SIMILARITY_THRESHOLD", "0.6", "similarity_threshold", 0.6),
    ],
)
def test_search_environment_overrides_are_applied(
    monkeypatch: pytest.MonkeyPatch,
    variable: str,
    value: str,
    attribute: str,
    expected: object,
) -> None:
    """Doc 05 section 13 sweeps thresholds; these settings make that possible."""
    monkeypatch.setenv(variable, value)

    assert getattr(SearchSettings(_env_file=None), attribute) == expected  # type: ignore[call-arg]


@pytest.mark.parametrize("top_k", [0, -1])
def test_a_non_positive_top_k_is_rejected(top_k: int) -> None:
    with pytest.raises(ValueError, match="greater than or equal to 1"):
        SearchSettings(_env_file=None, top_k=top_k)  # type: ignore[call-arg]


@pytest.mark.parametrize("threshold", ["nan", "inf", "-inf"])
def test_a_non_finite_threshold_is_rejected(threshold: str) -> None:
    with pytest.raises(ValueError, match="should be a finite number"):
        SearchSettings(_env_file=None, similarity_threshold=float(threshold))  # type: ignore[call-arg]


def test_search_settings_are_frozen() -> None:
    config = SearchSettings(_env_file=None)  # type: ignore[call-arg]

    with pytest.raises(ValueError, match="frozen"):
        config.top_k = 10  # type: ignore[misc]
