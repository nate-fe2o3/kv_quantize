"""FibQuant KV-cache compression for Qwen3.5 full-attention layers.

See fibquant/codebook.py for the offline codebook construction (arXiv:2605.11478),
fibquant/scoring.py for the shared chunked nearest-codeword scorer,
fibquant/spec.py for the operating point, fibquant/payload.py for the
compressed KV storage format, and fibquant/cache.py for the transformers
DynamicCache integration.
"""

from .cache import FibQuantCache, FibQuantLayer, enable_fibquant
from .codebook import (
    build_codebook,
    build_directions,
    build_radii,
    build_rotation,
)
from .quantize import bytes_per_token, decode, encode, index_dtype, pack_indices, unpack_indices
from .runtime import FibQuantRuntime, active_specs
from .scoring import min_pairwise_distance, nearest
from .spec import FibQuantSpec, default_spec_path, load_spec, spec_path

__all__ = [
    "FibQuantCache",
    "FibQuantLayer",
    "FibQuantRuntime",
    "FibQuantSpec",
    "enable_fibquant",
    "active_specs",
    "build_codebook",
    "build_directions",
    "build_radii",
    "build_rotation",
    "default_spec_path",
    "load_spec",
    "spec_path",
    "bytes_per_token",
    "decode",
    "encode",
    "index_dtype",
    "pack_indices",
    "unpack_indices",
    "nearest",
    "min_pairwise_distance",
]
