"""The trusted service-to-service boundary (doc 03, Phase 19; doc 06, 1-2).

    Browser -> Backend -> authentication -> trusted internal call -> AI Service

The Backend authenticates the end user and supplies a trusted ``userId`` in
the request body. This service must therefore implement **no** end-user
authentication of its own: no JWT verification, no sessions, no cookies, no
RBAC, no business authorization.

These are negative requirements, so the tests are written the way negative
requirements have to be: not "does auth work" but "is there provably none".
A behavioural half -- a forged credential changes nothing -- and a structural
half -- no such code or dependency exists to be reached.

Doc 06 section 2 lists an internal API key, mTLS and private-network
enforcement as *possible later transport hardening*, explicitly "not Sprint 1
user-auth logic inside AI". Nothing here adds one; the tests pin the absence
so it stays a deliberate decision rather than an oversight.
"""

import ast
import tomllib
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from memovo_ai.api.dependencies import (
    get_memory_processor_service,
    get_memory_search_service,
)
from memovo_ai.core.config import SearchSettings
from memovo_ai.embeddings import EMBEDDING_DIMENSION, Embedding
from memovo_ai.main import create_app
from memovo_ai.providers.vector_search import FakeVectorSearchProvider, VectorRecord
from memovo_ai.services import MemoryProcessorService, MemorySearchService

pytestmark = pytest.mark.integration

PROCESS = "/ai/memories/process"
SEARCH = "/ai/memories/search"

OWNER = "user_owner"
INTRUDER = "user_intruder"

#: Structurally a JWT -- three base64url segments -- claiming to be someone
#: else. Nothing in this service should parse it, which is the point.
FORGED_JWT = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
    ".eyJzdWIiOiJ1c2VyX2ludHJ1ZGVyIiwicm9sZSI6ImFkbWluIn0"
    ".c2lnbmF0dXJlLXRoYXQtaXMtbm90LWNoZWNrZWQ"
)

SRC = Path(__file__).resolve().parents[2] / "src" / "memovo_ai"
PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"

#: Packages that would mean this service does its own auth.
AUTH_PACKAGES = frozenset(
    {
        "jwt",
        "pyjwt",
        "jose",
        "python-jose",
        "authlib",
        "oauthlib",
        "requests-oauthlib",
        "passlib",
        "bcrypt",
        "argon2",
        "argon2-cffi",
        "itsdangerous",
        "starsessions",
        "fastapi-users",
        "casbin",
    }
)

#: Request attributes that would mean identity is being taken from transport
#: rather than from the trusted body field.
TRANSPORT_IDENTITY = ("cookies", "session", "auth", "user")


class StubEmbeddings:
    async def embed_documents(self, texts: Sequence[str]) -> list[Embedding]:
        return [[0.1] * EMBEDDING_DIMENSION for _ in texts]

    async def embed_query(self, query: str) -> Embedding:
        return [0.1] * EMBEDDING_DIMENSION


def record(user_id: str, memory_id: str) -> VectorRecord:
    return VectorRecord(
        user_id=user_id,
        memory_id=memory_id,
        chunk_id=f"{memory_id}_chunk_0",
        chunk_index=0,
        content="a stored chunk",
        title="a memory",
        tags=(),
        score=0.99,
    )


def build_app() -> FastAPI:
    embeddings = StubEmbeddings()
    records = [record(OWNER, "memory_owner"), record(INTRUDER, "memory_intruder")]

    application = create_app(load_services=False)
    application.dependency_overrides[get_memory_search_service] = lambda: MemorySearchService(
        embedding_provider=embeddings,  # type: ignore[arg-type]
        vector_search=FakeVectorSearchProvider(records),
        settings=SearchSettings(_env_file=None),  # type: ignore[call-arg]
    )
    application.dependency_overrides[get_memory_processor_service] = (
        lambda: MemoryProcessorService(embedding_provider=embeddings)  # type: ignore[arg-type]
    )
    return application


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(build_app()) as test_client:
        yield test_client


def search(client: TestClient, user_id: str, **kwargs: object) -> dict[str, object]:
    response = client.post(SEARCH, json={"userId": user_id, "query": "anything"}, **kwargs)  # type: ignore[arg-type]

    assert response.status_code == 200

    return dict(response.json())


def source_files() -> list[Path]:
    return sorted(SRC.rglob("*.py"))


def imported_names() -> set[str]:
    """Every module imported anywhere under ``src``, top-level name only."""
    found: set[str] = set()

    for path in source_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                found.add(node.module.split(".")[0])

    return found


def code_tokens(path: Path) -> set[str]:
    """Identifiers and literal strings that are actually *executed*.

    Docstrings and comments are excluded deliberately. This module's own
    subject matter means the word "JWT" appears all over the codebase in
    prose -- "this service does not decode JWTs" -- and a scan that counted
    those would fail on documentation that says the right thing. What matters
    is whether a name is referenced or a header string is looked up.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    documented = {
        docstring
        for node in ast.walk(tree)
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
        and (docstring := ast.get_docstring(node, clean=False)) is not None
    }

    tokens: set[str] = set()
    for node in ast.walk(tree):
        match node:
            case ast.Name():
                tokens.add(node.id)
            case ast.Attribute():
                tokens.add(node.attr)
            case ast.FunctionDef() | ast.AsyncFunctionDef() | ast.ClassDef():
                tokens.add(node.name)
            case ast.arg():
                tokens.add(node.arg)
            case ast.keyword() if node.arg:
                tokens.add(node.arg)
            case ast.Import():
                tokens.update(alias.name for alias in node.names)
            case ast.ImportFrom() if node.module:
                tokens.add(node.module)
            case ast.Constant() if isinstance(node.value, str) and node.value not in documented:
                tokens.add(node.value)

    return {token.lower() for token in tokens}


def files_mentioning(terms: Sequence[str]) -> list[str]:
    """Source files whose executable code references any of ``terms``."""
    return sorted(
        f"{path.relative_to(SRC).as_posix()}:{term}"
        for path in source_files()
        for term in terms
        if any(term in token for token in code_tokens(path))
    )


# ---------------------------------------------------------------------------
# Behaviour: a forged credential changes nothing
# ---------------------------------------------------------------------------


def test_an_authorization_header_is_ignored(client: TestClient) -> None:
    """The header is not consulted, so the response cannot depend on it."""
    without = search(client, OWNER)
    with_header = search(client, OWNER, headers={"Authorization": f"Bearer {FORGED_JWT}"})

    assert with_header == without


def test_a_bearer_token_cannot_select_another_user(client: TestClient) -> None:
    """Identity comes from the body. A token claiming ``sub`` must not win."""
    results = search(client, OWNER, headers={"Authorization": f"Bearer {FORGED_JWT}"})

    returned = [item["memoryId"] for item in results["results"]]  # type: ignore[union-attr,index]

    assert returned == ["memory_owner"]


def test_a_user_id_header_is_ignored(client: TestClient) -> None:
    """Doc 06 section 9 locks userId to the body, not a header."""
    results = search(client, OWNER, headers={"X-User-Id": INTRUDER})

    returned = [item["memoryId"] for item in results["results"]]  # type: ignore[union-attr,index]

    assert returned == ["memory_owner"]


def test_a_cookie_is_ignored(client: TestClient) -> None:
    without = search(client, OWNER)
    with_cookie = search(client, OWNER, headers={"Cookie": f"session={FORGED_JWT}"})

    assert with_cookie == without


def test_an_invalid_token_is_not_rejected(client: TestClient) -> None:
    """Nothing validates it, so garbage in the header must not cause a 401.

    A 401 here would mean the service had started verifying credentials.
    """
    response = client.post(
        SEARCH,
        json={"userId": OWNER, "query": "anything"},
        headers={"Authorization": "Bearer not-a-token-at-all"},
    )

    assert response.status_code == 200


def test_a_missing_user_id_is_a_validation_error_not_an_auth_error(
    client: TestClient,
) -> None:
    """Absent identity is malformed input, never "unauthenticated"."""
    response = client.post(SEARCH, json={"query": "anything"})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_INPUT"


def test_process_ignores_credentials_too(client: TestClient) -> None:
    body = {
        "type": "note",
        "memoryId": "memory_1",
        "title": "a title",
        "content": "a description",
        "tags": [],
    }

    plain = client.post(PROCESS, json=body).json()
    with_header = client.post(
        PROCESS, json=body, headers={"Authorization": f"Bearer {FORGED_JWT}"}
    ).json()

    assert with_header == plain


# ---------------------------------------------------------------------------
# Structure: there is no auth code to reach
# ---------------------------------------------------------------------------


def test_no_authentication_library_is_a_dependency() -> None:
    """The strongest form of the guarantee: the code cannot be written."""
    manifest = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))

    declared = list(manifest["project"]["dependencies"])
    for extra in manifest["project"].get("optional-dependencies", {}).values():
        declared.extend(extra)

    names = {
        requirement.split("[")[0].split(">")[0].split("=")[0].split("<")[0].strip().lower()
        for requirement in declared
    }

    assert names & AUTH_PACKAGES == set()


def test_no_authentication_library_is_imported() -> None:
    assert imported_names() & AUTH_PACKAGES == set()


#: The one module allowed to spell the header: it *sends* a bearer token to
#: the generation provider on an outbound request. Nothing under ``src``
#: reads an inbound one, and the list is exact so a second mention anywhere
#: still fails.
OUTBOUND_CREDENTIAL_CLIENTS = ("providers/generation/openrouter.py",)


def test_no_source_file_reads_the_authorization_header() -> None:
    assert files_mentioning(["authorization"]) == [
        f"{path}:authorization" for path in OUTBOUND_CREDENTIAL_CLIENTS
    ]


def test_no_source_file_reads_a_cookie() -> None:
    assert files_mentioning(["cookie", "set-cookie"]) == []


def test_nothing_takes_identity_from_the_transport_layer() -> None:
    """``request.user``, ``request.session``, ``request.cookies``, ``request.auth``.

    Each is a Starlette accessor that would move identity out of the trusted
    body field and into something the caller controls.
    """
    offenders = [
        f"{path.relative_to(SRC).as_posix()}:{node.lineno}"
        for path in source_files()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path)))
        if isinstance(node, ast.Attribute)
        and node.attr in TRANSPORT_IDENTITY
        and isinstance(node.value, ast.Name)
        and node.value.id == "request"
    ]

    assert offenders == []


def test_no_authentication_endpoint_exists() -> None:
    application = create_app(load_services=False)
    paths = {route.path for route in application.routes if hasattr(route, "path")}  # type: ignore[attr-defined]

    forbidden = {"/login", "/logout", "/token", "/auth", "/session", "/refresh", "/oauth"}

    assert paths & forbidden == set()


def test_the_openapi_schema_declares_no_security() -> None:
    """A declared scheme would tell the Backend to authenticate to this service."""
    with TestClient(create_app(load_services=False)) as client:
        schema = client.get("/openapi.json").json()

    assert "securitySchemes" not in schema.get("components", {})
    assert "security" not in schema


def test_no_endpoint_requires_security(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()["paths"]

    for methods in schema.values():
        for operation in methods.values():
            assert "security" not in operation


def test_no_authorization_or_role_logic_exists() -> None:
    """RBAC and business authorization belong to the Backend (doc 06, 1)."""
    assert files_mentioning(["rbac", "permission", "require_role", "is_admin", "scope_check"]) == []


def test_no_jwt_decoding_happens_anywhere() -> None:
    assert files_mentioning(["jwt", "decode_token", "verify_token"]) == []
    # "bearer" is spelled once: on the outbound provider request, never on
    # anything inbound. The list is exact, so any other mention fails.
    assert files_mentioning(["bearer"]) == [
        f"{path}:bearer" for path in OUTBOUND_CREDENTIAL_CLIENTS
    ]


def test_no_session_handling_exists() -> None:
    assert files_mentioning(["session", "login", "logout", "credential"]) == []


def test_the_scan_would_notice_a_violation(tmp_path: Path) -> None:
    """Proves the structural tests above are not vacuously passing."""
    planted = tmp_path / "planted.py"
    planted.write_text('token = request.headers.get("Authorization")\n', encoding="utf-8")

    assert "authorization" in code_tokens(planted)


def test_the_scan_ignores_prose(tmp_path: Path) -> None:
    """And that it does not fire on documentation saying the right thing."""
    documented = tmp_path / "documented.py"
    documented.write_text('"""This service never reads Authorization."""\n', encoding="utf-8")

    assert "authorization" not in code_tokens(documented)
