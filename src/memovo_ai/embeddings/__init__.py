"""Embedding provider interfaces and output validation.

Sprint 1 keeps this layer model-free: the interface, the embedding invariants,
and a validating wrapper. The Qwen3 implementation arrives at Phase 06.
"""

from memovo_ai.embeddings.base import EmbeddingProvider, ValidatedEmbeddingProvider
from memovo_ai.embeddings.models import (
    EMBEDDING_DIMENSION,
    Embedding,
    EmbeddingError,
    EmbeddingInferenceError,
    InvalidEmbeddingError,
    ModelUnavailableError,
    l2_norm,
    validate_embedding,
    validate_embeddings,
)
from memovo_ai.embeddings.qwen3 import Qwen3EmbeddingProvider, TextEncoder

__all__ = [
    "EMBEDDING_DIMENSION",
    "Embedding",
    "EmbeddingError",
    "EmbeddingInferenceError",
    "EmbeddingProvider",
    "InvalidEmbeddingError",
    "ModelUnavailableError",
    "Qwen3EmbeddingProvider",
    "TextEncoder",
    "ValidatedEmbeddingProvider",
    "l2_norm",
    "validate_embedding",
    "validate_embeddings",
]
