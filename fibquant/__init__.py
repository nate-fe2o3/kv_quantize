"""FibQuant KV-cache compression for Qwen3.5 full-attention layers.

See fibquant/codebook.py for the offline codebook construction (arXiv:2605.11478).
"""

from .cache import FibQuantCache, FibQuantLayer, FibQuantSpec, enable_fibquant
from .codebook import (
    build_codebook,
    build_directions,
    build_radii,
    build_rotation,
    default_spec_path,
    load_spec,
    save_spec,
)
from .quantize import bytes_per_token, decode, encode, index_dtype

__all__ = [
    "FibQuantCache",
    "FibQuantLayer",
    "FibQuantSpec",
    "enable_fibquant",
    "build_codebook",
    "build_directions",
    "build_radii",
    "build_rotation",
    "default_spec_path",
    "load_spec",
    "save_spec",
    "bytes_per_token",
    "decode",
    "encode",
    "index_dtype",
]
