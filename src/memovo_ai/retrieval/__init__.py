"""Retrieval: vector hit models, filtering, ranking and orchestration.

Sprint 1 adds these per phase. Only the hit model exists so far; threshold
filtering (Phase 11), ranking and deduplication (Phase 12) and the search
service (Phase 13) follow.
"""

from memovo_ai.retrieval.models import VectorSearchHit

__all__ = ["VectorSearchHit"]
