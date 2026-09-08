"""The vector search provider boundary.

This interface is **read-only**. It exposes ``search`` and nothing else: no
``insert``, ``upsert``, ``update`` or ``delete`` (doc 02 section 5, doc 06
section 4). The Backend owns vector persistence; the AI Service only reads.

``user_id`` is a required keyword argument rather than an optional filter so
that no implementation can accidentally search across users. It is a
**pre-filter**: an adapter must hand it to the vector engine as part of the
query, so the engine never ranks another user's chunks. Searching globally and
discarding foreign results afterwards is forbidden -- Top-K would already have
been consumed by chunks the user may not see, and cross-user retrieval is a
critical security failure (doc 06, section 3).

Application services depend on this protocol, never on a concrete vector
database. The production adapter is written once the storage technology is
confirmed (doc 01, section 8).
"""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from memovo_ai.retrieval.models import VectorSearchHit

__all__ = ["VECTOR_WRITE_OPERATIONS", "VectorSearchProvider"]

#: Operation names a read-only provider must never expose. Asserted by tests
#: against both the protocol and every implementation.
VECTOR_WRITE_OPERATIONS = ("insert", "upsert", "update", "delete", "add", "remove", "index")


@runtime_checkable
class VectorSearchProvider(Protocol):
    """Reads the nearest chunks for one user's query."""

    async def search(
        self,
        *,
        user_id: str,
        query_embedding: Sequence[float],
        top_k: int,
    ) -> list[VectorSearchHit]:
        """Return at most ``top_k`` hits for ``user_id``, best score first.

        Implementations must apply ``user_id`` as a pre-filter inside the
        vector engine. No similarity threshold is applied here: filtering by
        score is Phase 11's job, and a provider that dropped low-scoring hits
        would make the threshold untestable.
        """
        ...
