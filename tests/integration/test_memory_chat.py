"""``POST /ai/chat/memories`` and ``POST /ai/memories/prepare-note`` over HTTP.

A fake vector index and a scripted generation provider; no model, no
network.
"""

import json
from collections.abc import Iterator, Sequence

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from memovo_ai.api.dependencies import (
    get_memory_chat_service,
    get_note_preparation_service,
)
from memovo_ai.core.config import GenerationSettings, SearchSettings
from memovo_ai.embeddings import EMBEDDING_DIMENSION, Embedding
from memovo_ai.generation import FakeGenerationProvider, InvalidGenerationOutputError
from memovo_ai.main import create_app
from memovo_ai.providers.vector_search import FakeVectorSearchProvider, VectorRecord
from memovo_ai.services import MemoryChatService, MemorySearchService, NotePreparationService

pytestmark = pytest.mark.integration

CHAT = "/ai/chat/memories"
NOTE = "/ai/memories/prepare-note"
USER = "user_123"


class StubEmbeddings:
    async def embed_documents(self, texts: Sequence[str]) -> list[Embedding]:
        return [[0.1] * EMBEDDING_DIMENSION for _ in texts]

    async def embed_query(self, query: str) -> Embedding:
        return [0.1] * EMBEDDING_DIMENSION


MEMORY = VectorRecord(
    user_id=USER,
    memory_id="memory_1",
    chunk_id="memory_1_chunk_0",
    chunk_index=0,
    content="Run one worker per container.",
    title="Deployment notes",
    tags=("deployment",),
    score=0.9,
)


def answer(text: str, used: list[str]) -> str:
    return json.dumps({"answer": text, "usedSources": used})


def build_app(
    records: Sequence[VectorRecord] = (),
    responses: Sequence[object] = (),
    *,
    generation_enabled: bool = True,
) -> FastAPI:
    fake = FakeGenerationProvider(list(responses))  # type: ignore[arg-type]
    generation = fake if generation_enabled else None
    settings = GenerationSettings(_env_file=None)  # type: ignore[call-arg]
    search = MemorySearchService(
        embedding_provider=StubEmbeddings(),
        vector_search=FakeVectorSearchProvider(records),
        settings=SearchSettings(_env_file=None),  # type: ignore[call-arg]
    )
    application = create_app(load_services=False)
    application.dependency_overrides[get_memory_chat_service] = lambda: MemoryChatService(
        search=search, generation=generation, settings=settings
    )
    application.dependency_overrides[get_note_preparation_service] = lambda: NotePreparationService(
        generation=generation, settings=settings
    )
    return application


def client_of(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


# --------------------------------------------------------------------------
# Routes exist
# --------------------------------------------------------------------------
def test_all_four_contract_endpoints_are_published() -> None:
    for client in client_of(build_app()):
        paths = client.get("/openapi.json").json()["paths"]

    assert {
        "/ai/memories/process",
        "/ai/memories/search",
        "/ai/chat/memories",
        "/ai/memories/prepare-note",
    } <= set(paths)


# --------------------------------------------------------------------------
# Chat
# --------------------------------------------------------------------------
def test_a_grounded_answer_has_the_contract_shape() -> None:
    app = build_app([MEMORY], [answer("You run one worker per container.", ["S1"])])

    for client in client_of(app):
        response = client.post(CHAT, json={"userId": USER, "message": "deployment?"})

    assert response.status_code == 200
    assert response.json() == {
        "found": True,
        "answer": "You run one worker per container.",
        "sources": [
            {
                "memoryId": "memory_1",
                "score": 0.9,
                "title": "Deployment notes",
                "chunks": [
                    {"chunkId": "memory_1_chunk_0", "content": "Run one worker per container."}
                ],
            }
        ],
    }


def test_no_match_is_found_false_with_no_sources_and_status_200() -> None:
    for client in client_of(build_app([], [answer("never", [])])):
        response = client.post(CHAT, json={"userId": USER, "message": "anything"})

    assert response.status_code == 200
    body = response.json()
    assert body["found"] is False
    assert body["sources"] == []
    assert isinstance(body["answer"], str) and body["answer"]


def test_the_documented_request_with_history_and_conversation_id_is_accepted() -> None:
    app = build_app([MEMORY], [answer("ok", ["S1"])])
    body = {
        "userId": USER,
        "message": "and the second one?",
        "conversationId": "conv-1",
        "history": [
            {"role": "user", "content": "What did I save about deployment?"},
            {"role": "assistant", "content": "One note about workers."},
        ],
    }

    for client in client_of(app):
        assert client.post(CHAT, json=body).status_code == 200


@pytest.mark.parametrize(
    "body",
    [
        {"message": "m"},
        {"userId": USER},
        {"userId": USER, "message": "m", "history": None},
        {"userId": USER, "message": "m", "conversationId": None},
        {"userId": USER, "message": "m", "history": [{"role": "system", "content": "x"}]},
        {"userId": USER, "message": "m", "history": [{"role": "user"}]},
        {"userId": USER, "message": "m", "type": "note"},
        {"user_id": USER, "message": "m"},
    ],
)
def test_invalid_chat_requests_are_422(body: dict[str, object]) -> None:
    for client in client_of(build_app([MEMORY], [answer("ok", ["S1"])])):
        response = client.post(CHAT, json=body)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_INPUT"


def test_disabled_generation_is_503_generation_unavailable() -> None:
    for client in client_of(build_app([MEMORY], generation_enabled=False)):
        response = client.post(CHAT, json={"userId": USER, "message": "m"})

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "GENERATION_UNAVAILABLE"
    assert response.json()["error"]["retryable"] is True


def test_malformed_model_output_is_502_and_not_retryable() -> None:
    for client in client_of(build_app([MEMORY], ["not json"])):
        response = client.post(CHAT, json={"userId": USER, "message": "m"})

    assert response.status_code == 502
    assert response.json()["error"] == {
        "code": "AI_INVALID_RESPONSE",
        "message": "The generated output was invalid",
        "retryable": False,
    }


def test_an_unknown_source_reference_is_502() -> None:
    for client in client_of(build_app([MEMORY], [answer("x", ["S9"])])):
        response = client.post(CHAT, json={"userId": USER, "message": "m"})

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "AI_INVALID_RESPONSE"


def test_a_startup_without_services_is_503_model_unavailable() -> None:
    for client in client_of(create_app(load_services=False)):
        response = client.post(CHAT, json={"userId": USER, "message": "m"})

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "MODEL_UNAVAILABLE"


def test_the_response_never_carries_the_user_id_or_handles() -> None:
    for client in client_of(build_app([MEMORY], [answer("ok [S1]", ["S1"])])):
        text = client.post(CHAT, json={"userId": USER, "message": "m"}).text

    assert USER not in text
    assert "S1" not in text
    assert "userId" not in text


def test_the_chat_route_contains_no_generation_or_retrieval_logic() -> None:
    import ast
    from pathlib import Path

    from memovo_ai.api.routes import memory_chat

    tree = ast.parse(Path(memory_chat.__file__ or "").read_text(encoding="utf-8"))
    imported = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module
    }

    for forbidden in ("memovo_ai.generation", "memovo_ai.retrieval", "memovo_ai.embeddings"):
        assert not any(module.startswith(forbidden) for module in imported)


# --------------------------------------------------------------------------
# Prepare note
# --------------------------------------------------------------------------
def test_a_prepared_note_has_the_contract_shape() -> None:
    scripted = json.dumps({"title": "Prepared note title", "content": "Cleaned content"})

    for client in client_of(build_app([], [scripted])):
        response = client.post(NOTE, json={"content": "raw content"})

    assert response.status_code == 200
    assert response.json() == {"title": "Prepared note title", "content": "Cleaned content"}


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"content": None},
        {"content": 1},
        {"content": "c", "userId": "u"},
        {"content": "c", "tags": []},
    ],
)
def test_invalid_note_requests_are_422(body: dict[str, object]) -> None:
    for client in client_of(build_app([], ["{}"])):
        response = client.post(NOTE, json=body)

    assert response.status_code == 422


def test_blank_note_content_is_422_without_a_model_call() -> None:
    for client in client_of(build_app([], [])):
        response = client.post(NOTE, json={"content": "   "})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_INPUT"


def test_note_preparation_when_disabled_is_503() -> None:
    for client in client_of(build_app([], generation_enabled=False)):
        response = client.post(NOTE, json={"content": "c"})

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "GENERATION_UNAVAILABLE"


def test_malformed_note_output_is_502() -> None:
    for client in client_of(build_app([], [InvalidGenerationOutputError("bad")])):
        response = client.post(NOTE, json={"content": "c"})

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "AI_INVALID_RESPONSE"


def test_note_response_never_carries_tags_or_identifiers() -> None:
    scripted = json.dumps({"title": "t", "content": "c", "tags": ["x"], "id": "abc"})

    for client in client_of(build_app([], [scripted])):
        body = client.post(NOTE, json={"content": "raw"}).json()

    assert set(body) == {"title", "content"}
