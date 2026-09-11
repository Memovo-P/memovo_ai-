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

__all__ = [
    "VECTOR_WRITE_OPERATIONS",
    "InvalidUserScopeError",
    "UserIsolationError",
    "UserScopedVectorSearchProvider",
    "VectorSearchProvider",
    "VectorSearchUnavailableError",
    "validate_user_scope",
]

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


class VectorSearchUnavailableError(Exception):
    """The vector engine could not answer.

    Covers connection failures, driver errors and malformed index documents.
    Phase 15 maps it to ``VECTOR_SEARCH_FAILED`` (503, retryable): the query
    itself was fine, so repeating it may well succeed.

    Messages are generic by construction. A driver exception can carry the
    connection string, and that must never reach a response or a log line
    (doc 06, section 6); the original is chained instead.
    """


class UserIsolationError(Exception):
    """A provider returned a chunk belonging to someone other than the caller.

    This is never an expected condition. It means the ``user_id`` pre-filter
    did not reach the vector engine, or reached it wrongly -- a critical
    security failure (doc 06, section 3), not a recoverable one.

    Phase 15 decides the public mapping. It is a service defect rather than a
    bad request, so ``INTERNAL_ERROR`` fits better than ``VECTOR_SEARCH_FAILED``;
    either way it must not be presented as retryable, because retrying a
    misconfigured filter leaks the same rows again.
    """


class InvalidUserScopeError(ValueError):
    """The supplied ``user_id`` cannot scope a search.

    Subclasses :class:`ValueError` so callers that catch the broader type
    still behave as before. It exists as its own type so the error contract
    can map a bad request to ``INVALID_INPUT`` without also sweeping up
    unrelated ``ValueError``s, such as a misconfigured provider name, which
    are service defects and belong on ``INTERNAL_ERROR``.
    """


def validate_user_scope(user_id: str) -> str:
    """Return ``user_id`` unchanged after checking it can scope a search.

    A blank identifier is rejected outright: an adapter handed an empty filter
    may match every row in a shared collection, which is precisely the
    cross-user leak the pre-filter exists to prevent.

    The value is returned verbatim, never trimmed or normalized. Silently
    altering an identity would search as somebody else.
    """
    if not isinstance(user_id, str) or not user_id.strip():
        message = "user_id must be a non-blank string to scope a vector search"
        raise InvalidUserScopeError(message)

    return user_id


class UserScopedVectorSearchProvider:
    """Enforces the user-isolation invariant around any provider.

    Two guarantees, in order:

    1. The search is refused unless ``user_id`` can actually scope it.
    2. Every returned hit is verified to belong to that user.

    Step 2 **raises** on a foreign hit; it does not drop it. That distinction
    is the whole point. Dropping would be post-filtering -- it would mask a
    broken pre-filter, silently return fewer than ``top_k`` results, and leave
    the leak in place for whatever code forgot to wrap the provider. Raising
    fails closed and surfaces the defect.

    This is defence in depth, not the isolation mechanism. Correctness still
    comes from the adapter passing ``user_id`` to the vector engine as a
    pre-filter, so that another user's chunks are never ranked at all.
    """

    __slots__ = ("_provider",)

    def __init__(self, provider: VectorSearchProvider) -> None:
        self._provider = provider

    async def search(
        self,
        *,
        user_id: str,
        query_embedding: Sequence[float],
        top_k: int,
    ) -> list[VectorSearchHit]:
        scoped = validate_user_scope(user_id)

        hits = await self._provider.search(
            user_id=scoped,
            query_embedding=query_embedding,
            top_k=top_k,
        )

        foreign = sum(1 for hit in hits if hit.user_id != scoped)
        if foreign:
            # Counts only: no identifiers, no content. This message reaches
            # logs, and CLAUDE.md forbids logging retrieved chunk text.
            message = (
                f"vector search returned {foreign} of {len(hits)} chunks owned by "
                "another user; the user pre-filter is not being applied"
            )
            raise UserIsolationError(message)

        return hits
