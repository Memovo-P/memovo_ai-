"""Qwen3 embedding provider.

The only module in the service that knows a model runtime exists. Everything
upstream depends on :class:`~memovo_ai.embeddings.base.EmbeddingProvider`, so
swapping the runtime touches this file alone.

Loading
-------
``sentence-transformers`` is imported lazily inside :meth:`Qwen3EmbeddingProvider.load`,
never at module import. Unit tests inject a fake encoder and therefore neither
import the library nor download weights.

The model is loaded once and shared (doc 02, section 7). ``load()`` is called
once, from the background build that runs after the HTTP process has started;
nothing loads per request.

Threading
---------
``SentenceTransformer.encode`` is synchronous and compute-bound. Calling it
directly from an async method would block the event loop for the whole forward
pass, stalling every concurrent request, so it runs via :func:`asyncio.to_thread`.

Errors
------
Failures surface as :class:`~memovo_ai.embeddings.models.ModelUnavailableError`
and :class:`~memovo_ai.embeddings.models.EmbeddingInferenceError`. Their
messages are deliberately generic and carry no model path, repository id or
underlying exception text -- the original is chained with ``raise ... from``,
so it stays available in tracebacks and never in a serialized message. That way
even a naive mapping at Phase 15 cannot leak internals (doc 06, section 6).
"""

import asyncio
from collections.abc import Sequence
from typing import Protocol, cast

from memovo_ai.core.config import EmbeddingSettings
from memovo_ai.embeddings.models import (
    Embedding,
    EmbeddingInferenceError,
    InvalidEmbeddingError,
    ModelUnavailableError,
)

__all__ = ["Qwen3EmbeddingProvider", "TextEncoder"]


class TextEncoder(Protocol):
    """The slice of ``SentenceTransformer`` this provider actually uses.

    Narrowing the dependency to one method is what lets unit tests substitute
    a fake without importing the real library.
    """

    def encode(self, sentences: list[str], /, **kwargs: object) -> object:
        """Return one vector per sentence, in order.

        Positional-only: this provider always calls it positionally, and
        ``SentenceTransformer`` names the parameter ``inputs``. Without the
        marker the protocol would not match the real encoder on a parameter
        name that is never used.
        """
        ...


def _to_embeddings(raw: object) -> list[Embedding]:
    """Normalize an encoder's output into plain lists of floats.

    ``sentence-transformers`` returns a numpy array by default and a list of
    tensors in other configurations. Both expose ``tolist()``; anything else is
    treated as an ordinary nested sequence. Structural validation happens
    afterwards in the embeddings layer, so this only converts.
    """
    if hasattr(raw, "tolist"):
        raw = raw.tolist()

    if isinstance(raw, str | bytes | dict) or not isinstance(raw, Sequence):
        message = f"encoder returned {type(raw).__name__}, expected a sequence of embeddings"
        raise InvalidEmbeddingError(message)

    embeddings: list[Embedding] = []
    for row in raw:
        vector = row.tolist() if hasattr(row, "tolist") else row
        if isinstance(vector, str | bytes | dict) or not isinstance(vector, Sequence):
            message = f"encoder returned a {type(vector).__name__} row, expected a sequence"
            raise InvalidEmbeddingError(message)
        embeddings.append([float(value) for value in vector])

    return embeddings


class Qwen3EmbeddingProvider:
    """Embeds documents and queries with a Qwen3-Embedding model.

    Implements :class:`~memovo_ai.embeddings.base.EmbeddingProvider`. Wrap it
    in :class:`~memovo_ai.embeddings.base.ValidatedEmbeddingProvider` to
    enforce the dimension and finiteness invariants on every output.
    """

    __slots__ = ("_encoder", "_settings")

    def __init__(self, encoder: TextEncoder, *, settings: EmbeddingSettings | None = None) -> None:
        self._encoder = encoder
        self._settings = settings if settings is not None else EmbeddingSettings()

    @classmethod
    def load(cls, settings: EmbeddingSettings | None = None) -> "Qwen3EmbeddingProvider":
        """Load the model once and return a ready provider.

        Call at startup, not per request.

        Raises:
            ModelUnavailableError: if the library is missing or loading fails.
        """
        resolved = settings if settings is not None else EmbeddingSettings()

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            message = (
                "embedding runtime is not installed; "
                "install the 'embeddings' extra to enable the model"
            )
            raise ModelUnavailableError(message) from error

        try:
            encoder = SentenceTransformer(resolved.model, device=resolved.resolved_device)
        except Exception as error:
            message = "embedding model could not be loaded"
            raise ModelUnavailableError(message) from error

        # SentenceTransformer.encode is overloaded across a large union of
        # input types, so it does not match TextEncoder structurally even
        # though the call this provider makes is valid. The cast is confined
        # to this boundary; everything above it sees the narrow protocol.
        return cls(cast("TextEncoder", encoder), settings=resolved)

    @property
    def settings(self) -> EmbeddingSettings:
        return self._settings

    async def embed_documents(self, texts: Sequence[str]) -> list[Embedding]:
        """Embed chunk texts in batches, preserving input order."""
        if not texts:
            return []

        return await self._encode(list(texts), prompt_name=None)

    async def embed_query(self, query: str) -> Embedding:
        """Embed a single search query.

        Kept separate from document embedding because Qwen3-Embedding applies
        an instruction prompt to queries but not to documents.
        """
        prompt_name = self._settings.query_prompt_name or None
        embeddings = await self._encode([query], prompt_name=prompt_name)

        if len(embeddings) != 1:
            message = f"expected 1 embedding for the query, got {len(embeddings)}"
            raise InvalidEmbeddingError(message)

        return embeddings[0]

    async def _encode(self, texts: list[str], *, prompt_name: str | None) -> list[Embedding]:
        kwargs: dict[str, object] = {
            "batch_size": self._settings.batch_size,
            # Model output is preserved as-is. Normalization is not locked
            # (doc 01, section 7); the decision is made before production
            # indexing, not silently here.
            "normalize_embeddings": False,
        }
        if prompt_name is not None:
            kwargs["prompt_name"] = prompt_name

        try:
            raw = await asyncio.to_thread(self._encoder.encode, texts, **kwargs)
        except InvalidEmbeddingError:
            raise
        except Exception as error:
            message = "embedding inference failed"
            raise EmbeddingInferenceError(message) from error

        return _to_embeddings(raw)
