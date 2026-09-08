"""Similarity threshold filtering.

Applied after the vector engine has returned its Top-K for the user, and
before grouping and ranking (doc 01, Flow B)::

    userId pre-filter -> Top-K -> threshold -> deduplicate -> rank

The threshold is **inclusive**: every source document writes it as
``score >= 0.75`` (doc 01 Flow B, doc 02 section 3, doc 03 Phase 11), so a hit
scoring exactly at the threshold is relevant.

For a personal-memory product the threshold exists to prefer returning nothing
over returning something unrelated (doc 05, section 12). Filtering here never
truncates: Top-K was already applied by the provider, and dropping further hits
would silently change the result count the ranking phase sees.
"""

import math
from collections.abc import Sequence

from memovo_ai.retrieval.models import VectorSearchHit

__all__ = ["DEFAULT_SIMILARITY_THRESHOLD", "filter_by_threshold", "meets_threshold"]

#: Locked by doc 01, decision 9. Doc 05 section 13 evaluates 0.60 through 0.85,
#: but evaluation informs a later product decision -- it does not silently
#: replace this value.
#:
#: 0.75 is exactly representable in binary floating point, so the boundary
#: comparison is exact rather than approximate.
DEFAULT_SIMILARITY_THRESHOLD = 0.75


def meets_threshold(
    hit: VectorSearchHit,
    *,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> bool:
    """Whether a single hit is relevant enough to return."""
    return hit.score >= threshold


def filter_by_threshold(
    hits: Sequence[VectorSearchHit],
    *,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> list[VectorSearchHit]:
    """Keep the hits that meet ``threshold``, preserving their order.

    No score range is assumed. The similarity metric is recommended but not
    locked (doc 01, section 7), so a threshold outside ``[0, 1]`` is accepted:
    a metric such as inner product is not bounded that way. Only a
    non-comparable threshold is rejected.

    Raises:
        ValueError: if ``threshold`` is NaN or infinite.
    """
    if not math.isfinite(threshold):
        message = f"threshold must be a finite number, got {threshold}"
        raise ValueError(message)

    return [hit for hit in hits if hit.score >= threshold]
