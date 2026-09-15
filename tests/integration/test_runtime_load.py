"""Runtime behaviour under mixed load (plan P6).

A slow generation call must not stall processing, search or the probes, and
the process must keep answering liveness while an answer is being generated.
Driven through the ASGI app with an async client so requests genuinely
overlap on one event loop; the generation provider is a fake that sleeps.
"""

import asyncio
import time
from collections.abc import AsyncIterator, Sequence

import httpx2 as httpx
import pytest
from fastapi import FastAPI

from memovo_ai.api.dependencies import Services
from memovo_ai.core.config import GenerationSettings, SearchSettings
from memovo_ai.embeddings import EMBEDDING_DIMENSION, Embedding
from memovo_ai.generation import BoundedGenerationProvider, GenerationRequest, GenerationResult
from memovo_ai.main import create_app, wait_for_initialization
from memovo_ai.providers.vector_search import FakeVectorSearchProvider, VectorRecord
from memovo_ai.services import (
    MemoryChatService,
    MemoryProcessorService,
    MemorySearchService,
    NotePreparationService,
)

pytestmark = pytest.mark.integration

USER = "user_123"
CHAT = "/ai/chat/memories"
SEARCH = "/ai/memories/search"
PROCESS = "/ai/memories/process"
NOTE = "/ai/memories/prepare-note"

ANSWER = '{"answer": "One worker per container.", "usedSources": ["S1"]}'
PREPARED = '{"title": "Deployment", "content": "One worker per container."}'


class StubEmbeddings:
    async def embed_documents(self, texts: Sequence[str]) -> list[Embedding]:
        return [[0.1] * EMBEDDING_DIMENSION for _ in texts]

    async def embed_query(self, query: str) -> Embedding:
        return [0.1] * EMBEDDING_DIMENSION


class SlowGeneration:
    """Sleeps on the event loop; never blocks it."""

    def __init__(self, delay: float) -> None:
        self.delay = delay
        self.in_flight = 0
        self.peak = 0

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        try:
            await asyncio.sleep(self.delay)
        finally:
            self.in_flight -= 1
        text = PREPARED if "<<<CONTENT>>>" in request.messages[-1].content else ANSWER
        return GenerationResult(text=text, finish_reason="stop", model="slow")


MEMORY = VectorRecord(
    user_id=USER,
    memory_id="memory_1",
    chunk_id="memory_1_chunk_0",
    chunk_index=0,
    content="Run one worker per container.",
    title="Deployment notes",
    tags=(),
    score=0.9,
)


def services_with(generation: SlowGeneration, *, slots: int) -> Services:
    embeddings = StubEmbeddings()
    vector_search = FakeVectorSearchProvider([MEMORY])
    bounded = BoundedGenerationProvider(generation, max_concurrency=slots, timeout_seconds=5)
    settings = GenerationSettings(_env_file=None)  # type: ignore[call-arg]
    search = MemorySearchService(
        embedding_provider=embeddings,
        vector_search=vector_search,
        settings=SearchSettings(_env_file=None),  # type: ignore[call-arg]
    )
    return Services(
        memory_search=search,
        memory_processor=MemoryProcessorService(embedding_provider=embeddings),
        vector_search=vector_search,
        generation=bounded,
        memory_chat=MemoryChatService(search=search, generation=bounded, settings=settings),
        note_preparation=NotePreparationService(generation=bounded, settings=settings),
    )


async def running_app(
    monkeypatch: pytest.MonkeyPatch, generation: SlowGeneration, *, slots: int = 2
) -> AsyncIterator[tuple[FastAPI, httpx.AsyncClient]]:
    """The real startup path with working services, over an ASGI transport.

    Startup yields before the build finishes, so the services are waited for
    explicitly: these tests are about load on a *ready* instance.
    """
    monkeypatch.setattr(
        "memovo_ai.main.build_services", lambda: services_with(generation, slots=slots)
    )
    app = create_app()

    async with app.router.lifespan_context(app):
        await wait_for_initialization(app)
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://ai") as client:
            yield app, client


def chat_body() -> dict[str, object]:
    return {"userId": USER, "message": "deployment?"}


def process_body() -> dict[str, object]:
    return {"type": "note", "memoryId": "m", "title": "t", "content": "c" * 200, "tags": []}


async def test_liveness_answers_while_an_answer_is_being_generated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    slow = SlowGeneration(0.4)
    async for _, client in running_app(monkeypatch, slow):
        chat = asyncio.create_task(client.post(CHAT, json=chat_body()))
        await asyncio.sleep(0.05)
        assert slow.in_flight == 1, "the chat request should be inside generation by now"

        started = time.perf_counter()
        health = await client.get("/health")
        ready = await client.get("/ready")
        probe_ms = (time.perf_counter() - started) * 1000

        assert health.status_code == 200
        assert ready.status_code == 200
        assert probe_ms < 200, f"probes waited {probe_ms:.0f} ms behind generation"

        response = await chat
        assert response.status_code == 200
        assert response.json()["found"] is True


async def test_processing_and_search_are_not_queued_behind_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Generation slots bound generation only. Retrieval work never waits for
    them."""
    slow = SlowGeneration(0.4)
    async for _, client in running_app(monkeypatch, slow, slots=1):
        first = asyncio.create_task(client.post(CHAT, json=chat_body()))
        second = asyncio.create_task(client.post(CHAT, json=chat_body()))
        await asyncio.sleep(0.05)

        started = time.perf_counter()
        search = await client.post(SEARCH, json={"userId": USER, "query": "deployment"})
        process = await client.post(PROCESS, json=process_body())
        retrieval_ms = (time.perf_counter() - started) * 1000

        assert search.status_code == 200
        assert search.json()["found"] is True
        assert process.status_code == 200
        assert retrieval_ms < 300, f"retrieval waited {retrieval_ms:.0f} ms behind generation"

        responses = await asyncio.gather(first, second)
        assert [r.status_code for r in responses] == [200, 200]
        assert slow.peak == 1, "the single generation slot was not honoured"


async def test_mixed_load_completes_with_bounded_generation_concurrency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    slow = SlowGeneration(0.1)
    async for _, client in running_app(monkeypatch, slow, slots=2):
        tasks = [
            *(client.post(CHAT, json=chat_body()) for _ in range(4)),
            *(client.post(NOTE, json={"content": "raw " * 20}) for _ in range(2)),
            *(client.post(SEARCH, json={"userId": USER, "query": "q"}) for _ in range(3)),
            *(client.post(PROCESS, json=process_body()) for _ in range(3)),
            client.get("/health"),
            client.get("/ready"),
        ]

        responses = await asyncio.gather(*tasks)

        assert all(r.status_code == 200 for r in responses), [r.status_code for r in responses]
        assert slow.peak <= 2
        assert slow.in_flight == 0


async def test_a_disconnected_chat_client_releases_its_generation_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    slow = SlowGeneration(0.5)
    async for _, client in running_app(monkeypatch, slow, slots=1):
        task = asyncio.create_task(client.post(CHAT, json=chat_body()))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        # Give the server side a moment to unwind, then the slot must be free
        # for the next caller without waiting out the original sleep.
        await asyncio.sleep(0.05)
        started = time.perf_counter()
        response = await client.post(CHAT, json=chat_body())
        elapsed = time.perf_counter() - started

        assert response.status_code == 200
        assert elapsed < 1.5


async def test_readiness_does_not_depend_on_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retrieval is what readiness certifies. A disabled generator leaves the
    instance in rotation; chat answers 503 on its own."""
    embeddings = StubEmbeddings()
    vector_search = FakeVectorSearchProvider([MEMORY])
    search = MemorySearchService(
        embedding_provider=embeddings,
        vector_search=vector_search,
        settings=SearchSettings(_env_file=None),  # type: ignore[call-arg]
    )
    settings = GenerationSettings(_env_file=None)  # type: ignore[call-arg]
    monkeypatch.setattr(
        "memovo_ai.main.build_services",
        lambda: Services(
            memory_search=search,
            memory_processor=MemoryProcessorService(embedding_provider=embeddings),
            vector_search=vector_search,
            generation=None,
            memory_chat=MemoryChatService(search=search, generation=None, settings=settings),
            note_preparation=NotePreparationService(generation=None, settings=settings),
        ),
    )
    app = create_app()

    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=transport, base_url="http://ai") as client,
    ):
        await wait_for_initialization(app)
        assert (await client.get("/ready")).status_code == 200
        chat = await client.post(CHAT, json=chat_body())
        assert chat.status_code == 503
        assert chat.json()["error"]["code"] == "GENERATION_UNAVAILABLE"
        search = await client.post(SEARCH, json={"userId": USER, "query": "q"})
        assert search.status_code == 200
