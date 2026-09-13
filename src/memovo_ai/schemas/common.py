"""Shared building blocks for the public AI <-> Backend API contract.

The public JSON contract uses camelCase field names (``memoryId``,
``chunkId``, ``chunkIndex``, ``userId``). Python attributes stay snake_case
and carry an explicit ``Field(alias=...)``; ``serialize_by_alias`` makes the
camelCase form the default on the wire in both directions.

``populate_by_name`` is left at its default of ``False``, so snake_case input is
*not* accepted. Combined with ``extra="forbid"`` that means a request sending
``memory_id`` is rejected rather than silently accepted alongside ``memoryId``.
"""

from pydantic import BaseModel, ConfigDict

__all__ = ["MemovoBaseModel"]


class MemovoBaseModel(BaseModel):
    """Base for every public request and response model.

    - ``extra="forbid"``: undocumented fields are rejected, never ignored.
    - ``strict=True``: no silent coercion (``"0"`` is not ``0``, ``1.0`` is not
      ``1``, ``True`` is not ``1.0``). Widening ``int`` to ``float`` remains
      allowed, so an embedding value of ``0`` is valid.
    - ``serialize_by_alias=True``: dumps emit the public camelCase names.
    """

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        serialize_by_alias=True,
    )
