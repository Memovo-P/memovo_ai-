"""Memovo AI Service.

Sprint 1 scope: memory processing (``/ai/memories/process``) and memory
retrieval (``/ai/memories/search``).

This service performs AI processing and retrieval only. It never writes to
MongoDB or to the Vector DB, and it does not implement authentication or
authorization. See ``memovo-ai-docs/`` for the authoritative specification.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
