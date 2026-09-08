"""Phase 00 foundation checks.

These assert that the package is importable and that the repository is wired
up as the specification requires. They are intentionally minimal - no AI
behaviour exists yet.
"""

import tomllib
from pathlib import Path

import memovo_ai

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_package_is_importable_and_versioned() -> None:
    assert memovo_ai.__version__ == "0.1.0"


def test_pyproject_version_matches_package_version() -> None:
    with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
        pyproject = tomllib.load(handle)

    assert pyproject["project"]["version"] == memovo_ai.__version__


def test_python_version_is_pinned_to_312() -> None:
    assert (REPO_ROOT / ".python-version").read_text(encoding="utf-8").strip() == "3.12"


def test_env_example_documents_locked_retrieval_defaults() -> None:
    """Top-K 5 and threshold 0.75 are locked Sprint 1 product decisions."""
    env_example = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")

    assert "MEMOVO_SEARCH_TOP_K=5" in env_example
    assert "MEMOVO_SEARCH_SIMILARITY_THRESHOLD=0.75" in env_example
    assert "MEMOVO_EMBEDDING_DIMENSION=1024" in env_example


def test_real_env_file_is_not_committed() -> None:
    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")

    assert ".env" in gitignore
    assert "!.env.example" in gitignore
