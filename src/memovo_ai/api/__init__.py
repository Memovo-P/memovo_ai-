"""HTTP transport.

Route registration, request dependencies and response serialization. Core AI
logic belongs to the service and domain layers, never here.
"""

from memovo_ai.api.router import api_router

__all__ = ["api_router"]
