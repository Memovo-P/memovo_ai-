"""Correlation on failure responses (contract section 12.1).

The contract promises the Backend that ``X-Request-Id`` works on error
responses, not only on successes, and that the failure's own log record can
be joined to the access record for the same request. Both were false when
these tests were written.

Why they are written against real failures rather than a mocked handler: the
defect lived in the *interaction* between the middleware and the handlers.
Starlette's ``ServerErrorMiddleware`` hosts the ``Exception`` handler and
wraps this service's middleware, so a response built there never passes back
through it, and the middleware's ContextVar has already been reset. Nothing
short of driving a genuine failure through the real stack would have shown
that -- the middleware and the handlers are each correct in isolation.

The modes cover both handler locations:

* validation and ``AiServiceError`` are handled *inside* the middleware, by
  ``ExceptionMiddleware``
* ``ModelUnavailableError``, ``VectorSearchUnavailableError``, ``TimeoutError``
  and anything unrecognised are handled *outside* it, by
  ``ServerErrorMiddleware``
"""

import logging
from collections.abc import Iterator, Sequence

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from memovo_ai.api.dependencies import get_memory_search_service
from memovo_ai.core.config import SearchSettings
from memovo_ai.core.errors import AiServiceError
from memovo_ai.core.logging import CORRELATION_ID_HEADER, MAX_VALUE_LENGTH, JsonFormatter
from memovo_ai.embeddings import EMBEDDING_DIMENSION, Embedding
from memovo_ai.main import create_app
from memovo_ai.providers.vector_search import FakeVectorSearchProvider
from memovo_ai.providers.vector_search.base import VectorSearchUnavailableError
from memovo_ai.retrieval.models import VectorSearchHit
from memovo_ai.schemas.errors import ErrorCode, ErrorResponse
from memovo_ai.services import MemorySearchService

pytestmark = pytest.mark.integration

SEARCH = "/ai/memories/search"
USER = "user_123"
SUPPLIED_ID = "job-77"

#: Planted in the exception messages the providers raise. A driver error
#: realistically carries the text it failed on, and none of it may reach a
#: response or a log line.
PRIVATE = "ZZQ-what-did-the-cardiologist-say"

#: Strings that would mean the envelope leaked internals.
LEAK_MARKERS = ("Traceback", "site-packages", ".py", "mongodb", "C:\\Users", PRIVATE)


class StubEmbeddings:
    async def embed_documents(self, texts: Sequence[str]) -> list[Embedding]:
        return [[0.1] * EMBEDDING_DIMENSION for _ in texts]

    async def embed_query(self, query: str) -> Embedding:
        return [0.1] * EMBEDDING_DIMENSION


class FailingVectorSearch:
    """A provider that fails the way a real adapter fails."""

    def __init__(self, error: Exception) -> None:
        self._error = error

    async def search(
        self, *, user_id: str, query_embedding: Sequence[float], top_k: int
    ) -> list[VectorSearchHit]:
        raise self._error


def app_with(provider: object) -> FastAPI:
    application = create_app(load_services=False)
    application.dependency_overrides[get_memory_search_service] = lambda: MemorySearchService(
        embedding_provider=StubEmbeddings(),  # type: ignore[arg-type]
        vector_search=provider,  # type: ignore[arg-type]
        settings=SearchSettings(_env_file=None),  # type: ignore[call-arg]
    )
    return application


def unavailable_app() -> FastAPI:
    """Startup built nothing, so the dependency raises ModelUnavailableError.

    The realistic shape of a failed model load, and the one path that reaches
    ``MODEL_UNAVAILABLE`` without stubbing the dependency itself.
    """
    return create_app(load_services=False)


#: name -> (app factory, request body, expected status, expected code)
CASES: dict[str, tuple[object, dict[str, object], int, ErrorCode]] = {
    "validation": (
        lambda: app_with(FakeVectorSearchProvider()),
        {"userId": USER},
        422,
        ErrorCode.INVALID_INPUT,
    ),
    "model_unavailable": (
        unavailable_app,
        {"userId": USER, "query": "anything"},
        503,
        ErrorCode.MODEL_UNAVAILABLE,
    ),
    "vector_search_failed": (
        lambda: app_with(FailingVectorSearch(VectorSearchUnavailableError("vector search failed"))),
        {"userId": USER, "query": "anything"},
        503,
        ErrorCode.VECTOR_SEARCH_FAILED,
    ),
    "timeout": (
        lambda: app_with(FailingVectorSearch(TimeoutError(f"timed out reading {PRIVATE}"))),
        {"userId": USER, "query": "anything"},
        504,
        ErrorCode.TIMEOUT,
    ),
    "unexpected": (
        lambda: app_with(FailingVectorSearch(RuntimeError(f"driver exploded on {PRIVATE}"))),
        {"userId": USER, "query": "anything"},
        500,
        ErrorCode.INTERNAL_ERROR,
    ),
    "chunking_failed": (
        lambda: app_with(FailingVectorSearch(AiServiceError(ErrorCode.CHUNKING_FAILED))),
        {"userId": USER, "query": "anything"},
        500,
        ErrorCode.CHUNKING_FAILED,
    ),
}

MODES = list(CASES)


def client_for(name: str) -> Iterator[TestClient]:
    factory = CASES[name][0]
    with TestClient(factory(), raise_server_exceptions=False) as client:  # type: ignore[operator]
        yield client


@pytest.fixture
def logs(caplog: pytest.LogCaptureFixture) -> Iterator[pytest.LogCaptureFixture]:
    with caplog.at_level(logging.DEBUG):
        yield caplog


def correlation_ids(caplog: pytest.LogCaptureFixture) -> dict[str, str]:
    """Correlation id per request-scoped event name."""
    found: dict[str, str] = {}
    for record in caplog.records:
        event = getattr(record, "memovo_event", None)
        fields = getattr(record, "memovo_fields", None)
        if event in {"http.request", "request.failed"} and isinstance(fields, dict):
            found[str(event)] = str(fields.get("correlation_id"))

    return found


def rendered(caplog: pytest.LogCaptureFixture) -> str:
    formatter = JsonFormatter()

    return "\n".join(formatter.format(record) for record in caplog.records)


# ---------------------------------------------------------------------------
# Status and envelope
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_the_expected_status_and_code_are_returned(mode: str) -> None:
    _, body, status, code = CASES[mode]

    for client in client_for(mode):
        response = client.post(SEARCH, json=body)

        assert response.status_code == status
        assert response.json()["error"]["code"] == code.value


@pytest.mark.parametrize("mode", MODES)
def test_the_envelope_shape_is_unchanged(mode: str) -> None:
    """Three fields, nothing more. The fix must not alter a response body."""
    _, body, _, _ = CASES[mode]

    for client in client_for(mode):
        payload = client.post(SEARCH, json=body).json()

        assert set(payload) == {"error"}
        assert set(payload["error"]) == {"code", "message", "retryable"}
        # Re-parsing through the contract model is what proves the shape.
        ErrorResponse.model_validate(payload)


@pytest.mark.parametrize("mode", MODES)
def test_the_body_carries_no_internal_detail(mode: str) -> None:
    _, body, _, _ = CASES[mode]

    for client in client_for(mode):
        text = client.post(SEARCH, json=body).text

        for marker in LEAK_MARKERS:
            assert marker not in text


# ---------------------------------------------------------------------------
# Correlation on the response
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_a_supplied_request_id_is_echoed_on_the_failure(mode: str) -> None:
    """The defect these tests were written for.

    Before the fix every mode handled by ``ServerErrorMiddleware`` -- 500,
    503 and 504 -- returned no header at all.
    """
    _, body, _, _ = CASES[mode]

    for client in client_for(mode):
        response = client.post(SEARCH, json=body, headers={CORRELATION_ID_HEADER: SUPPLIED_ID})

        assert response.headers[CORRELATION_ID_HEADER] == SUPPLIED_ID


@pytest.mark.parametrize("mode", MODES)
def test_an_id_is_generated_when_none_is_supplied(mode: str) -> None:
    _, body, _, _ = CASES[mode]

    for client in client_for(mode):
        response = client.post(SEARCH, json=body)

        assert response.headers.get(CORRELATION_ID_HEADER)


@pytest.mark.parametrize("mode", MODES)
def test_a_generated_id_matches_the_logs_for_that_request(
    mode: str, logs: pytest.LogCaptureFixture
) -> None:
    """The Backend did not label this request, so the AI's own id is the only
    thing linking the response it received to the records it can be shown.

    Asserted against the *generated* value rather than a supplied one, which
    is the case a test using a fixed header never covers.
    """
    _, body, _, _ = CASES[mode]

    for client in client_for(mode):
        response = client.post(SEARCH, json=body)

    generated = response.headers[CORRELATION_ID_HEADER]
    found = correlation_ids(logs)

    assert generated
    assert found["http.request"] == generated, "access log lost the generated id"

    if mode not in UNLOGGED_MODES:
        assert found["request.failed"] == generated, "failure log lost the generated id"


@pytest.mark.parametrize("mode", MODES)
def test_the_id_appears_exactly_once(mode: str) -> None:
    """Both the middleware and the handler may stamp it; one must win."""
    _, body, _, _ = CASES[mode]

    for client in client_for(mode):
        response = client.post(SEARCH, json=body, headers={CORRELATION_ID_HEADER: SUPPLIED_ID})

        assert response.headers.get_list(CORRELATION_ID_HEADER) == [SUPPLIED_ID]


@pytest.mark.parametrize("mode", MODES)
def test_two_failing_requests_do_not_share_a_generated_id(mode: str) -> None:
    _, body, _, _ = CASES[mode]

    for client in client_for(mode):
        first = client.post(SEARCH, json=body).headers[CORRELATION_ID_HEADER]
        second = client.post(SEARCH, json=body).headers[CORRELATION_ID_HEADER]

        assert first != second


def test_a_control_character_in_the_request_id_is_stripped_before_echoing() -> None:
    """A tab is the strongest thing an HTTP client can actually smuggle in.

    A raw newline cannot reach here: ``httpx`` rejects it client-side and a
    conforming server rejects it at the wire, so the header value that
    arrives is always single-line. A tab *is* legal in a header value and is
    a control character, so it is what actually exercises the strip.
    """
    for client in client_for("unexpected"):
        response = client.post(
            SEARCH,
            json={"userId": USER, "query": "anything"},
            headers={CORRELATION_ID_HEADER: "job-1\tlevel=INFO event=forged"},
        )

        echoed = response.headers[CORRELATION_ID_HEADER]

        assert "\t" not in echoed
        assert echoed == "job-1level=INFO event=forged"


def test_an_over_long_request_id_is_truncated_before_echoing() -> None:
    """Truncation, exercised rather than asserted in the abstract."""
    for client in client_for("unexpected"):
        response = client.post(
            SEARCH,
            json={"userId": USER, "query": "anything"},
            headers={CORRELATION_ID_HEADER: "j" * 500},
        )

        echoed = response.headers[CORRELATION_ID_HEADER]

        assert len(echoed) == MAX_VALUE_LENGTH
        assert echoed == "j" * MAX_VALUE_LENGTH


def test_a_sanitized_request_id_matches_between_response_and_logs(
    logs: pytest.LogCaptureFixture,
) -> None:
    """The echoed value and the logged value must be the same sanitized id."""
    for client in client_for("unexpected"):
        response = client.post(
            SEARCH,
            json={"userId": USER, "query": "anything"},
            headers={CORRELATION_ID_HEADER: "job-1\tforged"},
        )

    echoed = response.headers[CORRELATION_ID_HEADER]
    found = correlation_ids(logs)

    assert found["http.request"] == echoed
    assert found["request.failed"] == echoed


def test_a_blank_request_id_is_replaced_rather_than_echoed_empty() -> None:
    """An empty header must not produce a response that correlates to nothing."""
    for client in client_for("unexpected"):
        response = client.post(
            SEARCH,
            json={"userId": USER, "query": "anything"},
            headers={CORRELATION_ID_HEADER: "   "},
        )

        echoed = response.headers[CORRELATION_ID_HEADER]

        assert echoed.strip()
        assert echoed != "   "


# ---------------------------------------------------------------------------
# Correlation in the logs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_the_failure_log_shares_the_access_log_id(
    mode: str, logs: pytest.LogCaptureFixture
) -> None:
    """The join the Backend needs to trace one request across both records.

    Before the fix ``request.failed`` recorded ``correlation_id: unset``,
    because the handler runs after the middleware has reset the ContextVar.
    """
    _, body, _, _ = CASES[mode]

    for client in client_for(mode):
        client.post(SEARCH, json=body, headers={CORRELATION_ID_HEADER: SUPPLIED_ID})

    found = correlation_ids(logs)

    assert found["http.request"] == SUPPLIED_ID
    for event, value in found.items():
        assert value == SUPPLIED_ID, f"{event} lost the correlation id"


#: Modes whose handler returns the envelope without a failure record. Only
#: request validation: a 422 is the caller's mistake, the access record
#: already carries the status, and the response names the offending fields.
#: A classified ``AiServiceError`` is *not* in this set -- see
#: ``test_a_classified_service_error_is_recorded_with_its_code``.
UNLOGGED_MODES = ("validation",)

#: Every mode that must produce a ``request.failed`` record.
LOGGED_FAILURE_MODES = [mode for mode in MODES if mode not in UNLOGGED_MODES]


@pytest.mark.parametrize("mode", LOGGED_FAILURE_MODES)
def test_the_failure_is_recorded_at_all(mode: str, logs: pytest.LogCaptureFixture) -> None:
    """Guards every correlation assertion above from passing vacuously.

    Without this, a regression that stopped logging failures entirely would
    leave those tests green -- they would find no ``request.failed`` record
    to disagree with.
    """
    _, body, _, _ = CASES[mode]

    for client in client_for(mode):
        client.post(SEARCH, json=body, headers={CORRELATION_ID_HEADER: SUPPLIED_ID})

    assert "request.failed" in correlation_ids(logs)


def test_an_unexpected_failure_is_always_recorded(logs: pytest.LogCaptureFixture) -> None:
    """Stated separately because it is the one that matters most.

    An unclassified exception is a defect. If it stopped being logged, the
    service would return 500 with nothing to diagnose it by.
    """
    _, body, _, _ = CASES["unexpected"]

    for client in client_for("unexpected"):
        client.post(SEARCH, json=body, headers={CORRELATION_ID_HEADER: SUPPLIED_ID})

    failures = [
        record
        for record in logs.records
        if getattr(record, "memovo_event", None) == "request.failed"
    ]

    assert len(failures) == 1
    assert failures[0].levelno == logging.ERROR
    assert getattr(failures[0], "memovo_fields", {})["error_code"] == "INTERNAL_ERROR"


@pytest.mark.parametrize("mode", UNLOGGED_MODES)
def test_an_unlogged_mode_is_still_correlated(mode: str, logs: pytest.LogCaptureFixture) -> None:
    """Documents the exclusion above instead of leaving it implicit.

    The response is still correlated; only the extra failure record is
    absent, because these handlers return the envelope directly.
    """
    _, body, _, _ = CASES[mode]

    for client in client_for(mode):
        response = client.post(SEARCH, json=body, headers={CORRELATION_ID_HEADER: SUPPLIED_ID})

    assert response.headers[CORRELATION_ID_HEADER] == SUPPLIED_ID
    assert correlation_ids(logs)["http.request"] == SUPPLIED_ID
    assert "request.failed" not in correlation_ids(logs)


def test_a_classified_service_error_is_recorded_with_its_code(
    logs: pytest.LogCaptureFixture,
) -> None:
    """An ``AiServiceError`` leaves one explained, correlated failure record.

    ``CHUNKING_FAILED`` and ``EMBEDDING_FAILED`` reach the caller through
    this exception and its own handler. Before this record existed, the
    access log showed a 500 with no ``error_code`` to explain it. The record
    is a WARNING without a traceback: the failure is already classified, and
    a classified failure has no unknown worth a stack trace.
    """
    _, body, _, _ = CASES["chunking_failed"]

    for client in client_for("chunking_failed"):
        response = client.post(SEARCH, json=body, headers={CORRELATION_ID_HEADER: SUPPLIED_ID})

    failures = [
        record
        for record in logs.records
        if getattr(record, "memovo_event", None) == "request.failed"
    ]

    assert response.json()["error"]["code"] == "CHUNKING_FAILED"
    assert len(failures) == 1
    assert failures[0].levelno == logging.WARNING
    assert not failures[0].exc_info, "a classified failure must not carry a traceback"

    fields = getattr(failures[0], "memovo_fields", {})
    assert fields["correlation_id"] == SUPPLIED_ID
    assert fields["error_code"] == "CHUNKING_FAILED"
    assert fields["error_type"] == "AiServiceError"
    assert fields["status"] == 500
    assert fields["route"] == SEARCH


@pytest.mark.parametrize("mode", MODES)
def test_no_exception_message_reaches_the_logs(mode: str, logs: pytest.LogCaptureFixture) -> None:
    """Two of these providers embed the user's query in the message."""
    _, body, _, _ = CASES[mode]

    for client in client_for(mode):
        client.post(SEARCH, json=body, headers={CORRELATION_ID_HEADER: SUPPLIED_ID})

    assert PRIVATE not in rendered(logs)


@pytest.mark.parametrize("mode", MODES)
def test_the_failure_log_records_only_the_exception_type(
    mode: str, logs: pytest.LogCaptureFixture
) -> None:
    _, body, _, _ = CASES[mode]

    for client in client_for(mode):
        client.post(SEARCH, json=body, headers={CORRELATION_ID_HEADER: SUPPLIED_ID})

    for record in logs.records:
        fields = getattr(record, "memovo_fields", None)
        if getattr(record, "memovo_event", None) == "request.failed" and isinstance(fields, dict):
            assert set(fields) == {
                "correlation_id",
                "route",
                "error_type",
                "error_code",
                "status",
            }
