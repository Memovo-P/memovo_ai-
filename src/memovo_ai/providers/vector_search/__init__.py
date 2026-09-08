"""Read-only vector search providers.

The interface exposes search only. The Backend owns every write to the vector
database (doc 06, section 4). The production adapter is added once the storage
technology is confirmed; until then the in-memory fake serves development and
tests.
"""

from memovo_ai.providers.vector_search.base import (
    VECTOR_WRITE_OPERATIONS,
    InvalidUserScopeError,
    UserIsolationError,
    UserScopedVectorSearchProvider,
    VectorSearchProvider,
    validate_user_scope,
)
from memovo_ai.providers.vector_search.fake import (
    FakeVectorSearchProvider,
    SearchCall,
    VectorRecord,
)

__all__ = [
    "VECTOR_WRITE_OPERATIONS",
    "FakeVectorSearchProvider",
    "InvalidUserScopeError",
    "SearchCall",
    "UserIsolationError",
    "UserScopedVectorSearchProvider",
    "VectorRecord",
    "VectorSearchProvider",
    "validate_user_scope",
]
