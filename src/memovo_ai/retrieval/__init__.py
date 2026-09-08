"""Retrieval: vector hit models, filtering, ranking and orchestration.

Sprint 1 adds these per phase. Threshold filtering and the hit model exist so
far; ranking and deduplication (Phase 12) and the search service (Phase 13)
follow.
"""

from memovo_ai.retrieval.filtering import (
    DEFAULT_SIMILARITY_THRESHOLD,
    filter_by_threshold,
    meets_threshold,
)
from memovo_ai.retrieval.models import VectorSearchHit

__all__ = [
    "DEFAULT_SIMILARITY_THRESHOLD",
    "VectorSearchHit",
    "filter_by_threshold",
    "meets_threshold",
]
