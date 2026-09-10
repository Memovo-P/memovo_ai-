"""Runs a dataset through the real retrieval pipeline.

The whole point of this harness is that it scores **chunks**, exactly as
production does:

    memory -> canonical content -> chunks -> embeddings -> vector index
    query  -> embedding -> user pre-filter -> Top-K -> threshold
           -> dedupe by memoryId -> max chunk score -> ranked memories

Scoring whole memories instead would measure a system nobody ships. A query
usually matches one chunk, not an entire document, so whole-document
similarity understates what the real pipeline sees -- and a threshold derived
that way would be wrong.

Embeddings are computed once and reused across every threshold, so a sweep
costs one pass over the model rather than one pass per threshold. Queries go
through a cache for the same reason: the search service embeds on every call,
which would otherwise mean re-encoding the whole query set once per threshold.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from memovo_ai.core.config import SearchSettings
from memovo_ai.embeddings.base import EmbeddingProvider
from memovo_ai.embeddings.models import Embedding
from memovo_ai.evaluation.metrics import QueryOutcome, RetrievalMetrics, summarize
from memovo_ai.providers.vector_search.fake import FakeVectorSearchProvider, VectorRecord
from memovo_ai.schemas.process import ProcessMemoryRequest
from memovo_ai.schemas.search import SearchMemoryRequest, SearchMemorySuccessResponse
from memovo_ai.services.memory_processor import MemoryProcessorService
from memovo_ai.services.memory_search import MemorySearchService

__all__ = [
    "Dataset",
    "EvaluationCase",
    "evaluate_thresholds",
    "index_memories",
    "load_dataset",
]

#: Every memory in an evaluation belongs to one synthetic owner. Isolation is
#: covered by its own tests; mixing owners here would only shrink the corpus.
EVAL_USER = "eval_user"


class _CachingEmbeddings:
    """Memoizes query embeddings across a threshold sweep.

    The search service embeds the query on every call, which is right for
    production and wasteful here: the same 60-odd queries are re-encoded for
    every threshold. Caching turns a sweep of N thresholds from N passes over
    the model into one. It is confined to evaluation -- nothing in the request
    path caches.
    """

    __slots__ = ("_provider", "_queries")

    def __init__(self, provider: EmbeddingProvider) -> None:
        self._provider = provider
        self._queries: dict[str, Embedding] = {}

    async def embed_documents(self, texts: Sequence[str]) -> list[Embedding]:
        return await self._provider.embed_documents(texts)

    async def embed_query(self, query: str) -> Embedding:
        cached = self._queries.get(query)
        if cached is None:
            cached = await self._provider.embed_query(query)
            self._queries[query] = cached

        return list(cached)


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    query_id: str
    kind: str
    query: str
    expected: frozenset[str]


@dataclass(frozen=True, slots=True)
class Dataset:
    name: str
    memories: tuple[ProcessMemoryRequest, ...]
    cases: tuple[EvaluationCase, ...]


def load_dataset(path: Path) -> Dataset:
    """Read a dataset file, validating memories against the real request schema."""
    raw = json.loads(path.read_text(encoding="utf-8"))

    return Dataset(
        name=str(raw["name"]),
        memories=tuple(ProcessMemoryRequest.model_validate(m) for m in raw["memories"]),
        cases=tuple(
            EvaluationCase(
                query_id=str(q["id"]),
                kind=str(q["kind"]),
                query=str(q["query"]),
                expected=frozenset(q["expectedMemoryIds"]),
            )
            for q in raw["queries"]
        ),
    )


async def index_memories(
    dataset: Dataset, *, processor: MemoryProcessorService
) -> list[VectorRecord]:
    """Chunk and embed every memory, as the Backend would before indexing."""
    records: list[VectorRecord] = []

    for memory in dataset.memories:
        response = await processor.process(memory)
        records.extend(
            VectorRecord(
                user_id=EVAL_USER,
                memory_id=memory.memory_id,
                chunk_id=chunk.chunk_id,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                title=memory.title,
                tags=tuple(memory.tags),
                embedding=tuple(chunk.embedding),
            )
            for chunk in response.chunks
        )

    return records


async def _run_at(
    threshold: float,
    *,
    dataset: Dataset,
    embeddings: EmbeddingProvider,
    records: Sequence[VectorRecord],
    top_k: int,
) -> tuple[RetrievalMetrics, list[QueryOutcome]]:
    service = MemorySearchService(
        embedding_provider=embeddings,
        vector_search=FakeVectorSearchProvider(records),
        settings=SearchSettings(
            _env_file=None,
            top_k=top_k,
            similarity_threshold=threshold,
        ),
    )

    outcomes: list[QueryOutcome] = []
    for case in dataset.cases:
        response = await service.search(
            SearchMemoryRequest.model_validate({"query": case.query, "userId": EVAL_USER})
        )
        returned = (
            tuple(r.memory_id for r in response.results)
            if isinstance(response, SearchMemorySuccessResponse)
            else ()
        )
        top = (
            response.results[0].score
            if isinstance(response, SearchMemorySuccessResponse) and response.results
            else None
        )
        outcomes.append(
            QueryOutcome(
                query_id=case.query_id,
                kind=case.kind,
                expected=case.expected,
                returned=returned,
                top_score=top,
            )
        )

    return summarize(threshold, outcomes), outcomes


async def evaluate_thresholds(
    dataset: Dataset,
    *,
    embeddings: EmbeddingProvider,
    thresholds: Sequence[float],
    top_k: int = 5,
) -> tuple[list[RetrievalMetrics], dict[float, list[QueryOutcome]]]:
    """Index once, then sweep thresholds over the same vectors.

    Returns the aggregate metrics per threshold and the raw per-query outcomes,
    so a caller can inspect which queries a threshold actually loses.
    """
    cached = _CachingEmbeddings(embeddings)
    processor = MemoryProcessorService(embedding_provider=cached)
    records = await index_memories(dataset, processor=processor)

    results: list[RetrievalMetrics] = []
    outcomes: dict[float, list[QueryOutcome]] = {}

    for threshold in thresholds:
        metrics, per_query = await _run_at(
            threshold,
            dataset=dataset,
            embeddings=cached,
            records=records,
            top_k=top_k,
        )
        results.append(metrics)
        outcomes[threshold] = per_query

    return results, outcomes
