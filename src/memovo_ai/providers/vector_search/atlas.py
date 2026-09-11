"""MongoDB Atlas Vector Search adapter.

Read-only, like every provider on this side of the boundary. It issues one
``$vectorSearch`` aggregation and nothing else -- no insert, upsert, update or
delete. The Backend owns every write to the collection (doc 02 section 5,
doc 06 section 4).

The user pre-filter
-------------------
``userId`` goes inside the ``$vectorSearch`` stage as its ``filter``, not into
a later ``$match``. That is the difference between a pre-filter and a
post-filter: Atlas applies it while traversing the index, so another user's
chunks are never candidates and never consume a Top-K slot. A ``$match`` after
the stage would leave the user with fewer than ``top_k`` results and no
indication why -- and is exactly what doc 06 section 3 forbids.

Connection
----------
``pymongo`` is imported lazily inside :meth:`MongoAtlasVectorSearchProvider.connect`,
so this module can be imported, type-checked and unit-tested without the
optional ``atlas`` extra installed. Credentials come from the environment and
are held in a ``SecretStr``; nothing here logs or formats them.
"""

import inspect
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any, Protocol, cast

from memovo_ai.core.config import AtlasSettings
from memovo_ai.providers.vector_search.base import (
    VectorSearchUnavailableError,
    validate_user_scope,
)
from memovo_ai.retrieval.models import VectorSearchHit

__all__ = ["MongoAtlasVectorSearchProvider", "VectorCollection", "build_search_pipeline"]

#: Document fields the Backend must write alongside each vector. Named exactly
#: as the public contract names them, so an operator can compare a stored
#: document against the API payload without translating.
FIELD_USER_ID = "userId"
FIELD_MEMORY_ID = "memoryId"
FIELD_CHUNK_ID = "chunkId"
FIELD_CHUNK_INDEX = "chunkIndex"
FIELD_CONTENT = "content"
FIELD_TITLE = "title"
FIELD_TAGS = "tags"
FIELD_SCORE = "score"

_REQUIRED_FIELDS = (
    FIELD_USER_ID,
    FIELD_MEMORY_ID,
    FIELD_CHUNK_ID,
    FIELD_CHUNK_INDEX,
    FIELD_CONTENT,
)


class VectorCollection(Protocol):
    """The slice of a Mongo collection this adapter uses.

    One method, so unit tests can substitute a stub without pymongo. Narrowing
    it here is what keeps the driver out of the test path.
    """

    def aggregate(self, pipeline: Sequence[Mapping[str, Any]], /) -> object:
        """Run an aggregation, yielding documents asynchronously.

        Returns ``object`` because the driver's async client returns a
        coroutine wrapping the cursor while simpler doubles return the
        iterator directly; :meth:`MongoAtlasVectorSearchProvider._aggregate`
        normalizes both. Positional-only, since that is how it is called.
        """
        ...


def build_search_pipeline(
    *,
    user_id: str,
    query_embedding: Sequence[float],
    top_k: int,
    settings: AtlasSettings,
) -> list[dict[str, Any]]:
    """The ``$vectorSearch`` aggregation, as a plain value.

    Separated from execution so the pipeline -- and above all the placement of
    the ``userId`` filter -- can be asserted in tests without a database.
    """
    return [
        {
            "$vectorSearch": {
                "index": settings.index,
                "path": settings.path,
                "queryVector": list(query_embedding),
                "numCandidates": settings.num_candidates(top_k),
                "limit": top_k,
                # Pre-filter. Inside the stage, evaluated during the index
                # traversal -- not a $match afterwards.
                "filter": {FIELD_USER_ID: {"$eq": user_id}},
            }
        },
        {
            "$project": {
                "_id": 0,
                FIELD_USER_ID: 1,
                FIELD_MEMORY_ID: 1,
                FIELD_CHUNK_ID: 1,
                FIELD_CHUNK_INDEX: 1,
                FIELD_CONTENT: 1,
                FIELD_TITLE: 1,
                FIELD_TAGS: 1,
                FIELD_SCORE: {"$meta": "vectorSearchScore"},
            }
        },
    ]


def _to_hit(document: Mapping[str, Any]) -> VectorSearchHit:
    """Convert one Atlas document, refusing anything malformed.

    A missing field means the Backend wrote an incomplete record. Skipping it
    would silently return fewer results than the index actually holds, so this
    fails instead. The message names the field only -- never the value, which
    could be a user's memory content.
    """
    missing = [field for field in _REQUIRED_FIELDS if document.get(field) is None]
    if missing:
        message = f"vector document is missing required field(s): {', '.join(missing)}"
        raise VectorSearchUnavailableError(message)

    tags = document.get(FIELD_TAGS) or ()
    if isinstance(tags, str):
        message = f"vector document field {FIELD_TAGS!r} must be a list, not a string"
        raise VectorSearchUnavailableError(message)

    try:
        return VectorSearchHit(
            user_id=str(document[FIELD_USER_ID]),
            memory_id=str(document[FIELD_MEMORY_ID]),
            chunk_id=str(document[FIELD_CHUNK_ID]),
            chunk_index=int(document[FIELD_CHUNK_INDEX]),
            content=str(document[FIELD_CONTENT]),
            score=float(document.get(FIELD_SCORE, 0.0)),
            title=str(document.get(FIELD_TITLE) or ""),
            tags=tuple(str(tag) for tag in tags),
        )
    except (TypeError, ValueError) as error:
        message = "vector document could not be read as a search hit"
        raise VectorSearchUnavailableError(message) from error


class MongoAtlasVectorSearchProvider:
    """Reads the nearest chunks for one user from Atlas Vector Search."""

    __slots__ = ("_client", "_collection", "_settings")

    def __init__(
        self,
        collection: VectorCollection,
        *,
        settings: AtlasSettings,
        client: object | None = None,
    ) -> None:
        self._collection = collection
        self._settings = settings
        self._client = client

    @classmethod
    def connect(cls, settings: AtlasSettings | None = None) -> "MongoAtlasVectorSearchProvider":
        """Open a client from configuration.

        Raises:
            VectorSearchUnavailableError: if the driver is missing or no
                connection string is configured. Both are startup problems,
                and both fail loudly rather than leaving a provider that
                answers every query with nothing.
        """
        resolved = settings if settings is not None else AtlasSettings()

        if not resolved.is_configured:
            message = (
                "Atlas connection string is not configured; "
                "set MEMOVO_ATLAS_URI to use the atlas vector provider"
            )
            raise VectorSearchUnavailableError(message)

        try:
            from pymongo import AsyncMongoClient
        except ImportError as error:
            message = (
                "MongoDB driver is not installed; "
                "install the 'atlas' extra to use the atlas vector provider"
            )
            raise VectorSearchUnavailableError(message) from error

        client: Any = AsyncMongoClient(
            resolved.uri.get_secret_value(),
            serverSelectionTimeoutMS=resolved.timeout_ms,
            connectTimeoutMS=resolved.timeout_ms,
            timeoutMS=resolved.timeout_ms,
        )
        collection = client[resolved.database][resolved.collection]

        return cls(cast("VectorCollection", collection), settings=resolved, client=client)

    @property
    def settings(self) -> AtlasSettings:
        return self._settings

    async def search(
        self,
        *,
        user_id: str,
        query_embedding: Sequence[float],
        top_k: int,
    ) -> list[VectorSearchHit]:
        """Return at most ``top_k`` hits for ``user_id``, best score first."""
        if top_k < 1:
            message = f"top_k must be at least 1, got {top_k}"
            raise ValueError(message)

        scoped = validate_user_scope(user_id)
        pipeline = build_search_pipeline(
            user_id=scoped,
            query_embedding=query_embedding,
            top_k=top_k,
            settings=self._settings,
        )

        documents = await self._aggregate(pipeline)

        # Atlas returns them ranked; the ordering contract is restated here so
        # a change in driver behaviour cannot silently reorder results.
        hits = [_to_hit(document) for document in documents]
        hits.sort(key=lambda hit: (-hit.score, hit.chunk_id))

        return hits

    async def _aggregate(self, pipeline: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
        try:
            cursor = self._collection.aggregate(pipeline)
            # pymongo's async client returns a coroutine; simpler doubles
            # return the iterator directly.
            if inspect.isawaitable(cursor):
                cursor = await cursor

            return [document async for document in cast("AsyncIterator[Mapping[str, Any]]", cursor)]
        except VectorSearchUnavailableError:
            raise
        except TimeoutError:
            # Already the right shape; Phase 15 maps it to TIMEOUT.
            raise
        except Exception as error:
            if _is_timeout(error):
                message = "vector search timed out"
                raise TimeoutError(message) from error

            # Deliberately generic: a driver exception can carry the
            # connection string, and it must never reach a response or a log
            # line (doc 06, section 6).
            message = "vector search failed"
            raise VectorSearchUnavailableError(message) from error

    async def aclose(self) -> None:
        """Close the underlying client, if this provider opened one."""
        close = getattr(self._client, "close", None)
        if close is None:
            return

        result = close()
        if inspect.isawaitable(result):
            await result


def _is_timeout(error: BaseException) -> bool:
    """Recognise a driver timeout without importing pymongo.

    Importing the driver here would defeat the lazy import and force the
    optional extra on every install, so the check is by type name.
    """
    names = {type(cause).__name__ for cause in (error, error.__cause__) if cause is not None}

    return any(
        name
        in {
            "ServerSelectionTimeoutError",
            "NetworkTimeout",
            "ExecutionTimeout",
            "WTimeoutError",
            "TimeoutError",
        }
        for name in names
    )
