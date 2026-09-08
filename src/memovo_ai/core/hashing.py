"""Deterministic chunk identifiers.

The guarantee (doc 03, Phase 04; CLAUDE.md)::

    same memoryId + same chunkIndex + same final chunk content
        -> same chunkId

    any change to final chunk content -> different chunkId

Retries with unchanged input therefore reproduce identical chunk IDs, which is
what lets the Backend reconcile a reprocessed Memory against the stored one.
The AI Service holds no state: nothing here consults previous chunk sets.

Framing
-------
Doc 03 sketches the concept as ``SHA256(memory_id + sep + chunk_index + sep +
content)``. Plain separator concatenation is ambiguous when a ``memoryId``
contains the separator -- ``("memory", 1, "2:2:text")`` and
``("memory:1", 2, "2:text")`` both render to ``memory:1:2:2:text`` and would
share an ID despite being different chunks of different Memories.

Each field is therefore length-prefixed before hashing, so exactly one triple
maps to any given hash input. The specific hashing implementation is internal
to the AI Service (CLAUDE.md), and this satisfies the documented behavioural
guarantee exactly.
"""

import hashlib

__all__ = ["CHUNK_ID_PREFIX", "chunk_id"]

#: Human-readable marker, matching the ``chunk_...`` form shown in doc 03.
CHUNK_ID_PREFIX = "chunk_"

#: Bytes used to encode each field's length in the hash input.
_LENGTH_BYTES = 8

#: Text encoding for every field. Fixed so that Arabic, CJK and emoji content
#: hash identically regardless of platform or locale.
_ENCODING = "utf-8"


def chunk_id(*, memory_id: str, chunk_index: int, content: str) -> str:
    """Derive the deterministic identifier for one chunk.

    ``content`` must be the *final* chunk content, exactly as it will appear
    in the response -- including any overlap prefix. Hashing anything else
    would break the guarantee that changed content changes the ID.

    Raises:
        ValueError: if ``chunk_index`` is negative.
    """
    if chunk_index < 0:
        message = f"chunk_index must not be negative, got {chunk_index}"
        raise ValueError(message)

    digest = hashlib.sha256()
    for field in (memory_id, str(chunk_index), content):
        encoded = field.encode(_ENCODING)
        digest.update(len(encoded).to_bytes(_LENGTH_BYTES, "big"))
        digest.update(encoded)

    return f"{CHUNK_ID_PREFIX}{digest.hexdigest()}"
