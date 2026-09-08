"""Cross-cutting utilities.

Sprint 1 adds these as their phases arrive. Deterministic hashing lands here
first; configuration, error definitions and logging follow in later phases.
"""

from memovo_ai.core.hashing import CHUNK_ID_PREFIX, chunk_id

__all__ = ["CHUNK_ID_PREFIX", "chunk_id"]
