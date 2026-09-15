"""Deployment artifacts (doc 03, Phase 23; doc 06, sections 7-8).

A Dockerfile is not covered by the rest of the suite, and the mistakes it
invites are the expensive kind: a secret baked into a layer, a root process,
a liveness probe wired to readiness. These tests read the artifacts and pin
the decisions that would otherwise only exist in a comment.

They check *content*, not that an image builds -- building belongs to CI,
where a daemon exists.
"""

import re
import tomllib
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]

DOCKERFILE = ROOT / "Dockerfile"
DOCKERIGNORE = ROOT / ".dockerignore"
COMPOSE = ROOT / "docker-compose.yml"
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
DEPLOYMENT_DOC = ROOT / "docs" / "deployment.md"


def dockerfile() -> str:
    return DOCKERFILE.read_text(encoding="utf-8")


def dockerfile_directives() -> list[str]:
    """Instruction lines only, with comments and blanks dropped."""
    return [
        line.strip()
        for line in dockerfile().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def compose() -> dict[str, object]:
    return dict(yaml.safe_load(COMPOSE.read_text(encoding="utf-8")))


def ai_service() -> dict[str, object]:
    return dict(compose()["services"]["ai"])  # type: ignore[index,call-overload]


def workflow() -> dict[str, object]:
    return dict(yaml.safe_load(WORKFLOW.read_text(encoding="utf-8")))


def ignored() -> list[str]:
    return [
        line.strip()
        for line in DOCKERIGNORE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


# ---------------------------------------------------------------------------
# The artifacts exist
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("artifact", [DOCKERFILE, DOCKERIGNORE, COMPOSE, WORKFLOW, DEPLOYMENT_DOC])
def test_the_artifact_exists(artifact: Path) -> None:
    assert artifact.is_file()


# ---------------------------------------------------------------------------
# Secrets
# ---------------------------------------------------------------------------


def test_the_env_file_is_excluded_from_the_build_context() -> None:
    """`.env` is the one file expected to hold a real Atlas URI.

    A build context is sent to the daemon whole, regardless of what COPY
    references, so excluding it is the only thing that keeps it out.
    """
    assert ".env" in ignored()


def test_the_example_env_is_still_available() -> None:
    assert "!.env.example" in ignored()


def test_no_credential_is_baked_into_the_image() -> None:
    """Directives only.

    Scanning the whole file would trip over the word "tokenizer" in a
    comment, and a scan that punishes accurate documentation gets deleted
    rather than fixed. What matters is what the build actually executes.
    """
    executed = " ".join(dockerfile_directives()).lower()

    for term in ("mongodb+srv://", "password", "secret", "api_key", "apikey"):
        assert term not in executed

    assert not re.search(r"\btokens?\b", executed)


def test_no_atlas_uri_default_is_set_in_the_image() -> None:
    assert "MEMOVO_ATLAS_URI" not in dockerfile()


def test_the_compose_file_carries_no_credential() -> None:
    """It is committed, so anything in it is public to the repository."""
    text = COMPOSE.read_text(encoding="utf-8").lower()

    assert "mongodb+srv://" not in text
    assert "memovo_atlas_uri:" not in text


def test_the_git_directory_is_excluded() -> None:
    assert ".git" in ignored()


# ---------------------------------------------------------------------------
# Least privilege
# ---------------------------------------------------------------------------


def test_the_container_does_not_run_as_root() -> None:
    users = [
        line.split(None, 1)[1].strip()
        for line in dockerfile_directives()
        if line.startswith("USER ")
    ]

    assert users
    assert users[-1] != "root"


def test_the_service_user_is_created_as_a_system_user() -> None:
    assert "useradd" in dockerfile()


def test_the_root_filesystem_is_read_only_in_compose() -> None:
    assert ai_service()["read_only"] is True


def test_privilege_escalation_is_disabled_in_compose() -> None:
    assert "no-new-privileges:true" in ai_service()["security_opt"]  # type: ignore[operator]


# ---------------------------------------------------------------------------
# Doc 06 section 7: the service is private
# ---------------------------------------------------------------------------


def test_compose_does_not_publish_the_port_on_every_interface() -> None:
    """The short form ``8000:8000`` binds 0.0.0.0. This service is private."""
    published = ai_service()["ports"]

    assert isinstance(published, list)
    for entry in published:
        assert isinstance(entry, dict), "use the long form so host_ip is explicit"
        assert entry["host_ip"] == "127.0.0.1"


# ---------------------------------------------------------------------------
# Doc 06 section 8: what the container holds
# ---------------------------------------------------------------------------


def test_the_embedding_runtime_is_installed() -> None:
    assert "--extra embeddings" in dockerfile()


def test_the_read_only_vector_adapter_is_installed() -> None:
    assert "--extra atlas" in dockerfile()


def test_development_dependencies_are_excluded() -> None:
    assert "--no-dev" in dockerfile()


def test_the_lockfile_is_respected() -> None:
    """``--frozen`` fails on drift instead of resolving something new."""
    assert "--frozen" in dockerfile()


def test_the_image_is_built_in_two_stages() -> None:
    """So no compiler, no uv and no build cache reach the runtime image."""
    stages = [line for line in dockerfile_directives() if line.startswith("FROM ")]

    assert len(stages) >= 2


def test_the_project_is_installed_non_editably() -> None:
    """An editable install leaves a .pth pointing at the build directory.

    That path does not exist in the runtime stage, so the import would fail
    at start -- or, worse, be papered over by putting the source on the path
    a second way.
    """
    assert "--no-editable" in dockerfile()


def test_the_virtualenv_path_matches_across_stages() -> None:
    """A virtualenv is not relocatable.

    Every console script in ``bin/`` carries an absolute shebang, so building
    at one path and copying to another leaves ``uvicorn`` pointing at a
    python that is not in the final image.
    """
    workdirs = {
        line.split(None, 1)[1].strip()
        for line in dockerfile_directives()
        if line.startswith("WORKDIR ")
    }

    assert workdirs == {"/app"}


def test_the_runtime_stage_does_not_install_uv() -> None:
    runtime = dockerfile().split("AS runtime", 1)[1]

    assert "astral-sh/uv" not in runtime


def test_the_python_version_matches_the_project_pin() -> None:
    pinned = (ROOT / ".python-version").read_text(encoding="utf-8").strip()

    assert f"python:{pinned}-slim" in dockerfile()


def test_no_queue_orchestration_is_bundled() -> None:
    """Doc 06 section 8: that responsibility stays with the Backend."""
    text = (dockerfile() + COMPOSE.read_text(encoding="utf-8")).lower()

    for term in ("redis", "bullmq", "celery", "rabbitmq", "kafka"):
        assert term not in text


# ---------------------------------------------------------------------------
# Doc 06 section 9: readiness
# ---------------------------------------------------------------------------


def test_the_image_healthcheck_uses_liveness_not_readiness() -> None:
    """A liveness probe on /ready turns a slow model load into a crash loop."""
    healthcheck = next(line for line in dockerfile_directives() if line.startswith("HEALTHCHECK"))
    command = dockerfile().split("HEALTHCHECK", 1)[1].split("CMD", 1)[1]

    assert "--start-period" in healthcheck
    assert "/health" in command
    assert "/ready" not in command


def test_the_compose_healthcheck_uses_liveness_too() -> None:
    test_command = " ".join(str(part) for part in ai_service()["healthcheck"]["test"])  # type: ignore[index,call-overload]

    assert "/health" in test_command
    assert "/ready" not in test_command


def test_the_healthcheck_allows_a_first_start_download_and_cold_load() -> None:
    """A container killed part-way through a download or load never
    finishes one, and a first start does both."""
    start_period = ai_service()["healthcheck"]["start_period"]  # type: ignore[index,call-overload]

    assert start_period == "300s"


# ---------------------------------------------------------------------------
# Runtime behaviour
# ---------------------------------------------------------------------------


def test_a_single_worker_is_configured() -> None:
    """The model is loaded per process and is over a gigabyte resident."""
    assert "--workers 1" in dockerfile()


def test_the_port_follows_the_platform() -> None:
    """Railway injects PORT and healthchecks that port; a hardcoded 8000 is
    unreachable there. The fallback keeps local runs on 8000."""
    assert "--port ${PORT:-8000}" in dockerfile()
    assert '"--port", "8000"' not in dockerfile()


def test_uvicorn_stays_pid_1_behind_the_shell() -> None:
    """Without exec, SIGTERM stops the shell and orphans uvicorn."""
    assert "exec uvicorn" in dockerfile()


def test_reload_is_not_enabled() -> None:
    assert "--reload" not in dockerfile()


def test_uvicorn_serves_the_application() -> None:
    assert "memovo_ai.main:app" in dockerfile()


def test_the_model_cache_is_writable_by_the_service_user() -> None:
    """The default cache is /root/.cache, which an unprivileged user cannot
    write -- the failure would surface on first load, not at build."""
    assert "HF_HOME=/home/memovo/" in dockerfile()


def test_the_runtime_downloads_weights_on_first_start() -> None:
    """Weights are not baked: the image must permit the runtime download."""
    assert "HF_HUB_OFFLINE=0" in dockerfile()
    assert not any("HF_HUB_OFFLINE=1" in line for line in dockerfile_directives())


def test_weights_are_not_baked_into_the_image() -> None:
    """A baked layer pushed the image past what the deployment target
    accepts. No build step may import the runtime to fetch weights."""
    assert "BAKE_MODEL" not in dockerfile()
    assert "sentence_transformers" not in dockerfile()


def test_the_configured_model_matches_the_locked_decision() -> None:
    manifest = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert manifest["project"]["name"] == "memovo-ai"
    assert "Qwen/Qwen3-Embedding-0.6B" in dockerfile()


# ---------------------------------------------------------------------------
# CI
# ---------------------------------------------------------------------------


def test_ci_runs_the_whole_quality_gate() -> None:
    steps = [
        step.get("run", "")
        for job in workflow()["jobs"].values()  # type: ignore[attr-defined]
        for step in job["steps"]
    ]
    commands = " ".join(steps)

    for gate in (
        "uv run pytest",
        "uv run ruff check .",
        "uv run ruff format --check .",
        "uv run mypy src",
    ):
        assert gate in commands


def test_ci_does_not_download_model_weights_in_the_default_job() -> None:
    """Doc 03 Phase 06: CI must stay light and must not fetch weights."""
    gate = workflow()["jobs"]["gate"]  # type: ignore[index,call-overload]
    commands = " ".join(step.get("run", "") for step in gate["steps"])

    assert "--extra embeddings" not in commands


def test_ci_also_runs_the_gate_with_the_extras_installed() -> None:
    """Installing the extras changed behaviour twice before.

    Both times the light job stayed green: a suite that quietly loaded real
    weights, and a mypy failure on an overloaded signature.
    """
    hermetic = workflow()["jobs"]["hermetic"]  # type: ignore[index,call-overload]
    commands = " ".join(step.get("run", "") for step in hermetic["steps"])

    assert "--extra embeddings" in commands
    assert "--extra atlas" in commands
    assert "uv run pytest" in commands


def test_ci_boots_the_image_offline_so_no_weights_are_fetched() -> None:
    """The image downloads weights on first start; CI must forbid that."""
    image = workflow()["jobs"]["image"]  # type: ignore[index,call-overload]
    commands = " ".join(step.get("run", "") for step in image["steps"])

    assert "HF_HUB_OFFLINE=1" in commands


def test_ci_boots_the_image_and_probes_it() -> None:
    """A container that builds but cannot start is what this catches."""
    image = workflow()["jobs"]["image"]  # type: ignore[index,call-overload]
    commands = " ".join(step.get("run", "") for step in image["steps"])

    assert "/health" in commands
    assert "/ready" in commands
