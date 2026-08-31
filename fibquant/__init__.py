"""FibQuant KV-cache compression for Qwen3.5 full-attention layers.

The package interface exposes the runtime modules callers need. Offline
construction and codec primitives live in their named modules rather than
being duplicated as package-root aliases.
"""

from .cache import FibQuantCache, FibQuantLayer
from .runtime import FibQuantRuntime, active_specs, enable_fibquant
from .spec import FibQuantSpec

__all__ = [
    "FibQuantCache",
    "FibQuantLayer",
    "FibQuantRuntime",
    "FibQuantSpec",
    "enable_fibquant",
    "active_specs",
]
