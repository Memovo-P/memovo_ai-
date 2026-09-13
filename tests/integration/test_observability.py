"""Correlation and privacy-safe logging through the HTTP layer (Phase 18).

Two things are checked here that a unit test cannot: that a real request
produces the operational record doc 06 section 5 asks for, and that no part
of that request's content reaches any log record, in any field, for any
outcome -- success, no-match, validation failure or crash.
"""

import json
import logging
from collections.abc import Iterator, Sequence

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from memovo_ai.api.dependencies import (
    get_memory_processor_service,
    get_memory_search_service,
)
from memovo_ai.core.config import LoggingSettings, SearchSettings
from memovo_ai.core.logging import CORRELATION_ID_HEADER, JsonFormatter, hash_user_id
from memovo_ai.embeddings import EMBEDDING_DIMENSION, Embedding
from memovo_ai.main import create_app
from memovo_ai.providers.vector_search import FakeVectorSearchProvider, VectorRecord
from memovo_ai.services import MemoryProcessorService, MemorySearchService

pytestmark = pytest.mark.integration

#: Pinned rather than inherited from the environment: a configured
#: ``MEMOVO_LOG_USER_SALT`` -- in a developer's ``.env`` or exported in a
#: shell -- would change the hash these tests compare against, and the suite
#: must behave the same on every machine. An explicit value outranks both.
LOG_SETTINGS = LoggingSettings(_env_file=None, user_salt="")  # type: ignore[call-arg]

PROCESS = "/ai/memories/process"
SEARCH = "/ai/memories/search"

USER = "user_123"

#: Deliberately distinctive, so a leak anywhere in a log line is unmissable.
PRIVATE_TITLE = "ZZQTITLE-appointment-with-the-cardiologist"
PRIVATE_DESCRIPTION = "ZZQDESC-blood-pressure-readings-were-high-again"
PRIVATE_TAG = "ZZQTAG-health"
PRIVATE_QUERY = "ZZQQUERY-what-did-the-cardiologist-say"
PRIVATE_CHUNK = "ZZQCHUNK-stored-chunk-text-from-the-index"

PRIVATE_VALUES = (
    PRIVATE_TITLE,
    PRIVATE_DESCRIPTION,
    PRIVATE_TAG,
    PRIVATE_QUERY,
    PRIVATE_CHUNK,
)

PROCESS_BODY: dict[str, object] = {
    "type": "note",
    "memoryId": "memory_456",
    "title": PRIVATE_TITLE,
    "content": PRIVATE_DESCRIPTION,
    "tags": [PRIVATE_TAG],
}

SEARCH_BODY: dict[str, object] = {"userId": USER, "query": PRIVATE_QUERY}


class StubEmbeddings:
    async def embed_documents(self, texts: Sequence[str]) -> list[Embedding]:
        return [[0.1] * EMBEDDING_DIMENSION for _ in texts]

    async def embed_query(self, query: str) -> Embedding:
        return [0.1] * EMBEDDING_DIMENSION


class ExplodingEmbeddings:
    """Fails with a message that must never be logged or returned."""

    async def embed_documents(self, texts: Sequence[str]) -> list[Embedding]:
        raise RuntimeError(f"connect failed for {PRIVATE_DESCRIPTION}")

    async def embed_query(self, query: str) -> Embedding:
        raise RuntimeError(f"connect failed for {PRIVATE_QUERY}")


def record(score: float = 0.94, content: str = PRIVATE_CHUNK) -> VectorRecord:
    return VectorRecord(
        user_id=USER,
        memory_id="memory_456",
        chunk_id="memory_456_chunk_0",
        chunk_index=0,
        content=content,
        title=PRIVATE_TITLE,
        tags=(PRIVATE_TAG,),
        score=score,
    )


def build_app(
    *,
    embeddings: object | None = None,
    records: Sequence[VectorRecord] = (),
) -> FastAPI:
    provider = embeddings if embeddings is not None else StubEmbeddings()
    application = create_app(load_services=False)
    application.dependency_overrides[get_memory_processor_service] = (
        lambda: MemoryProcessorService(embedding_provider=provider)  # type: ignore[arg-type]
    )
    application.dependency_overrides[get_memory_search_service] = lambda: MemorySearchService(
        embedding_provider=provider,  # type: ignore[arg-type]
        vector_search=FakeVectorSearchProvider(records),
        settings=SearchSettings(_env_file=None),  # type: ignore[call-arg]
        logging_settings=LOG_SETTINGS,
    )
    return application


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(build_app(records=[record()]), raise_server_exceptions=False) as test_client:
        yield test_client


@pytest.fixture
def logs(caplog: pytest.LogCaptureFixture) -> Iterator[pytest.LogCaptureFixture]:
    with caplog.at_level(logging.DEBUG):
        yield caplog


def events(caplog: pytest.LogCaptureFixture, name: str) -> list[dict[str, object]]:
    return [
        dict(fields)
        for item in caplog.records
        if (fields := getattr(item, "memovo_fields", None)) is not None
        and getattr(item, "memovo_event", None) == name
    ]


def rendered(caplog: pytest.LogCaptureFixture) -> str:
    """Every record as it would actually be written, tracebacks included."""
    formatter = JsonFormatter()

    return "\n".join(formatter.format(item) for item in caplog.records)


# ---------------------------------------------------------------------------
# Privacy: nothing the user wrote may reach a log
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("secret", PRIVATE_VALUES)
def test_no_request_content_reaches_the_logs(
    client: TestClient, logs: pytest.LogCaptureFixture, secret: str
) -> None:
    client.post(PROCESS, json=PROCESS_BODY)
    client.post(SEARCH, json=SEARCH_BODY)

    assert secret not in rendered(logs)


def test_a_no_match_search_logs_no_query_text(
    logs: pytest.LogCaptureFixture,
) -> None:
    """The no-match path is the one most likely to be debugged by eye."""
    with TestClient(build_app(), raise_server_exceptions=False) as client:
        assert client.post(SEARCH, json=SEARCH_BODY).status_code == 200

    assert PRIVATE_QUERY not in rendered(logs)


def test_a_failing_request_logs_neither_content_nor_the_exception_message(
    logs: pytest.LogCaptureFixture,
) -> None:
    """A traceback is recorded for a defect, so it must still stay clean.

    The exception message here embeds user content on purpose: this is the
    realistic shape of a driver error, and it is the reason every traceback
    is rendered without exception messages at all.

    Frames survive, including their source lines -- those are static code, so
    ``f"connect failed for {PRIVATE_QUERY}"`` appears as written and the value
    does not.
    """
    with TestClient(
        build_app(embeddings=ExplodingEmbeddings()), raise_server_exceptions=False
    ) as client:
        assert client.post(SEARCH, json=SEARCH_BODY).status_code == 500

    output = rendered(logs)

    assert PRIVATE_QUERY not in output
    assert "message omitted" in output
    assert "memory_search.py" in output


def test_a_validation_failure_logs_no_field_values(
    client: TestClient, logs: pytest.LogCaptureFixture
) -> None:
    body = dict(PROCESS_BODY)
    body.pop("memoryId")

    assert client.post(PROCESS, json=body).status_code == 422
    assert PRIVATE_DESCRIPTION not in rendered(logs)


def test_the_user_id_is_hashed_in_the_logs(
    client: TestClient, logs: pytest.LogCaptureFixture
) -> None:
    client.post(SEARCH, json=SEARCH_BODY)

    output = rendered(logs)

    assert USER not in output
    assert hash_user_id(USER) in output


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------


def test_the_backends_request_id_is_echoed_back(client: TestClient) -> None:
    response = client.post(SEARCH, json=SEARCH_BODY, headers={CORRELATION_ID_HEADER: "job-abc-1"})

    assert response.headers[CORRELATION_ID_HEADER] == "job-abc-1"


def test_a_request_without_an_id_still_gets_one(client: TestClient) -> None:
    response = client.post(SEARCH, json=SEARCH_BODY)

    assert response.headers[CORRELATION_ID_HEADER]


def test_two_requests_without_ids_do_not_share_one(client: TestClient) -> None:
    first = client.post(SEARCH, json=SEARCH_BODY)
    second = client.post(SEARCH, json=SEARCH_BODY)

    assert first.headers[CORRELATION_ID_HEADER] != second.headers[CORRELATION_ID_HEADER]


def test_the_service_log_carries_the_backends_id(
    client: TestClient, logs: pytest.LogCaptureFixture
) -> None:
    """This is the join that lets the Backend trace a job into this service."""
    client.post(SEARCH, json=SEARCH_BODY, headers={CORRELATION_ID_HEADER: "job-abc-1"})

    searched = events(logs, "memory.searched")

    assert [item["correlation_id"] for item in searched] == ["job-abc-1"]


def test_the_access_log_and_the_service_log_share_one_id(
    client: TestClient, logs: pytest.LogCaptureFixture
) -> None:
    client.post(SEARCH, json=SEARCH_BODY)

    assert (
        events(logs, "http.request")[0]["correlation_id"]
        == (events(logs, "memory.searched")[0]["correlation_id"])
    )


def test_a_forged_id_cannot_inject_a_log_record(
    client: TestClient, logs: pytest.LogCaptureFixture
) -> None:
    client.post(
        SEARCH,
        json=SEARCH_BODY,
        headers={CORRELATION_ID_HEADER: 'x", "level": "INFO", "event": "forged'},
    )

    lines = [line for line in rendered(logs).splitlines() if line.strip()]

    assert all(json.loads(line)["event"] != "forged" for line in lines)


def test_an_error_response_is_still_correlated(client: TestClient) -> None:
    response = client.post(SEARCH, json={"userId": USER}, headers={CORRELATION_ID_HEADER: "job-9"})

    assert response.status_code == 422
    assert response.headers[CORRELATION_ID_HEADER] == "job-9"


# ---------------------------------------------------------------------------
# The operational metadata doc 06 section 5 asks for
# ---------------------------------------------------------------------------


def test_every_request_produces_one_access_record(
    client: TestClient, logs: pytest.LogCaptureFixture
) -> None:
    client.post(SEARCH, json=SEARCH_BODY)

    assert len(events(logs, "http.request")) == 1


def test_the_access_record_carries_route_status_and_duration(
    client: TestClient, logs: pytest.LogCaptureFixture
) -> None:
    client.post(SEARCH, json=SEARCH_BODY)

    entry = events(logs, "http.request")[0]

    assert entry["method"] == "POST"
    assert entry["route"] == SEARCH
    assert entry["status"] == 200
    assert isinstance(entry["duration_ms"], float)


def test_an_unmatched_path_is_logged_without_echoing_it_raw(
    client: TestClient, logs: pytest.LogCaptureFixture
) -> None:
    """A 404 path is caller-controlled text, so it goes through sanitizing."""
    client.get("/ai/" + "x" * 500)

    route = events(logs, "http.request")[0]["route"]

    assert isinstance(route, str)
    assert len(route) <= 96


def test_the_search_record_carries_the_retrieval_metrics(
    client: TestClient, logs: pytest.LogCaptureFixture
) -> None:
    client.post(SEARCH, json=SEARCH_BODY)

    entry = events(logs, "memory.searched")[0]

    assert entry["top_k"] == 5
    assert entry["threshold"] == 0.75
    assert entry["hit_count"] == 1
    assert entry["result_count"] == 1
    assert entry["matched"] is True
    assert isinstance(entry["embedding_ms"], float)
    assert isinstance(entry["vector_search_ms"], float)


def test_a_no_match_search_is_recorded_as_such(logs: pytest.LogCaptureFixture) -> None:
    with TestClient(build_app(records=[record(score=0.10)])) as client:
        client.post(SEARCH, json=SEARCH_BODY)

    entry = events(logs, "memory.searched")[0]

    assert entry["hit_count"] == 1
    assert entry["result_count"] == 0
    assert entry["matched"] is False


def test_the_process_record_carries_the_ingestion_metrics(
    client: TestClient, logs: pytest.LogCaptureFixture
) -> None:
    client.post(PROCESS, json=PROCESS_BODY)

    entry = events(logs, "memory.processed")[0]

    assert entry["memory_id"] == "memory_456"
    assert entry["chunk_count"] == 1
    assert isinstance(entry["content_chars"], int)
    assert isinstance(entry["chunking_ms"], float)
    assert isinstance(entry["embedding_ms"], float)


def test_an_over_long_memory_id_cannot_break_ingestion(
    client: TestClient, logs: pytest.LogCaptureFixture
) -> None:
    """The contract sets no length limit, so the log field must truncate."""
    body = dict(PROCESS_BODY) | {"memoryId": "m" * 500}

    assert client.post(PROCESS, json=body).status_code == 200
    assert len(str(events(logs, "memory.processed")[0]["memory_id"])) <= 96


def test_an_empty_memory_is_still_recorded(
    client: TestClient, logs: pytest.LogCaptureFixture
) -> None:
    """The zero-chunk path must not be an observability blind spot."""
    empty = {"type": "note", "memoryId": "memory_789", "title": "", "content": " ", "tags": []}

    assert client.post(PROCESS, json=empty).status_code == 200
    assert events(logs, "memory.processed")[0]["chunk_count"] == 0


def test_a_failure_is_recorded_with_its_public_code(logs: pytest.LogCaptureFixture) -> None:
    with TestClient(
        build_app(embeddings=ExplodingEmbeddings()), raise_server_exceptions=False
    ) as client:
        client.post(SEARCH, json=SEARCH_BODY)

    entry = events(logs, "request.failed")[0]

    assert entry["error_code"] == "INTERNAL_ERROR"
    assert entry["error_type"] == "RuntimeError"
    assert entry["status"] == 500


def test_a_failed_request_still_produces_an_access_record(
    logs: pytest.LogCaptureFixture,
) -> None:
    with TestClient(
        build_app(embeddings=ExplodingEmbeddings()), raise_server_exceptions=False
    ) as client:
        client.post(SEARCH, json=SEARCH_BODY)

    assert events(logs, "http.request")[0]["status"] == 500


def test_startup_and_shutdown_are_recorded(logs: pytest.LogCaptureFixture) -> None:
    with TestClient(build_app()):
        pass

    assert events(logs, "service.started")[0]["services_available"] is False
    assert events(logs, "service.stopped")
