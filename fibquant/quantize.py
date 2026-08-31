"""Batched FibQuant encode/decode for KV cache tensors.

Tensors are (batch, heads, seq, d) with d = head dim. Each head-vector is
independently encoded: fp16 norm header + one uint8/uint16 container element
(block index) per k-block.

`encode`/`decode` below are compatibility adapters: they keep the original
bare-tensor call shape (codebook, rotation passed in directly) by building a
throwaway `codec.PreparedCodec` per call. That throwaway has nowhere to
amortize a device move or the augmented-codebook precompute across calls, so
callers on a hot path (KVPayload, in payload.py) build one `PreparedCodec`
once and reuse it via `codec.prepared_codec_for` instead of calling these
functions -- see codec.py for the deep primitive and the reasoning.
"""

from __future__ import annotations

import torch

from .codec import PreparedCodec
from .scoring import DEFAULT_SCORE_BYTES


def index_dtype(n_levels: int) -> torch.dtype:
    """Smallest container dtype that can hold codeword indices in [0, n_levels)."""
    if n_levels <= 2**8:
        return torch.uint8
    if n_levels <= 2**16:
        return torch.uint16
    raise ValueError(f"n_levels={n_levels} exceeds uint16; not supported")


_BITS_PACKED = 12  # 12-bit indices (N=4096) pair-pack: 2 x 12 bits = 3 bytes exactly


def pack_indices(indices: torch.Tensor, n_levels: int) -> torch.Tensor:
    """Pack codeword indices into the compact per-row storage form.

    8-bit (N <= 256) and 16-bit (N <= 65536) containers are already minimum
    size and returned as-is. 12-bit indices (N = 4096) are pair-packed: two
    consecutive indices per 3 bytes, stored flat as (..., 1.5 * blocks) uint8
    so storage ops (cat/crop/reorder/select) operate on unchanged dims.

    Byte layout per pair (even index e, odd index o):
        b0 = e % 256,  b1 = (e // 256) * 16 + (o % 16),  b2 = o // 16
    """
    bits = int(n_levels - 1).bit_length()
    if bits != _BITS_PACKED:
        return indices
    blocks = indices.shape[-1]
    if blocks % 2:
        raise ValueError(f"pair-packing requires an even number of blocks, got {blocks}")
    idx = indices.long().view(*indices.shape[:-1], blocks // 2, 2)
    even, odd = idx[..., 0], idx[..., 1]  # each in [0, n_levels)
    b0 = (even % 256).to(torch.uint8)
    b1 = ((even // 256) * 16 + (odd % 16)).to(torch.uint8)
    b2 = (odd // 16).to(torch.uint8)
    return torch.stack([b0, b1, b2], dim=-1).reshape(*indices.shape[:-1], -1)


def unpack_indices(packed: torch.Tensor, n_levels: int) -> torch.Tensor:
    """Inverse of pack_indices; returns (..., blocks) uint16 codeword indices.

    Non-12-bit payloads are already in logical form and returned as-is.
    """
    bits = int(n_levels - 1).bit_length()
    if bits != _BITS_PACKED:
        return packed
    if packed.numel() == 0:
        # Empty cache (e.g. after crop(0)): reshape(..., -1, 3) of 0 elements
        # is ambiguous (any -1 satisfies "0 elements"), so derive the logical
        # block count directly from the packed layout instead -- every 3
        # packed bytes hold 2 logical indices (see pack_indices).
        blocks = packed.shape[-1] * 2 // 3
        return torch.empty(*packed.shape[:-1], blocks, dtype=torch.uint16, device=packed.device)
    x = packed.long().reshape(*packed.shape[:-1], -1, 3)
    b0, b1, b2 = x[..., 0], x[..., 1], x[..., 2]
    even = b0 + (b1 // 16) * 256
    odd = (b1 % 16) + b2 * 16
    return torch.stack([even, odd], dim=-1).reshape(*packed.shape[:-1], -1).to(torch.uint16)


@torch.no_grad()
def encode(
    x: torch.Tensor,
    codebook: torch.Tensor,
    rotation: torch.Tensor,
    k: int,
    max_score_bytes: int = DEFAULT_SCORE_BYTES,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize x -> (indices, norms).

    indices: (B, H, S, d/k) uint8/uint16, one codeword index per k-block
             (dtype chosen from codebook size; see index_dtype).
    norms:   (B, H, S) fp16, per-vector L2 norms.

    Compatibility adapter over codec.PreparedCodec: builds one for this call
    (moving codebook/rotation to x.device if needed) and delegates to it.
    Nearest-codeword search is delegated to the shared chunked scorer
    (scoring.nearest) — the same implementation offline codebook construction
    uses, so the training and deployment geometry cannot drift. Peak memory
    stays ~max_score_bytes regardless of N.
    """
    n_levels = codebook.shape[0]
    codec = PreparedCodec(codebook, rotation, k, n_levels)
    if codec.codebook.device != x.device:
        codec = codec.to(x.device)
    return codec.encode(x, max_score_bytes=max_score_bytes)


@torch.no_grad()
def decode(
    indices: torch.Tensor,
    norms: torch.Tensor,
    codebook: torch.Tensor,
    rotation: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize (indices, norms) back to a (B, H, S, d) tensor.

    Compatibility adapter over codec.PreparedCodec; see `encode` above.
    """
    n_levels = codebook.shape[0]
    k = codebook.shape[1]
    codec = PreparedCodec(codebook, rotation, k, n_levels)
    if codec.codebook.device != indices.device:
        codec = codec.to(indices.device)
    return codec.decode(indices, norms, dtype=dtype)


def bytes_per_token(d: int, k: int, n_levels: int) -> dict[str, float]:
    """Payload bytes per token per (K or V) head vector, plus the fp16 reference.

    Two payload figures are reported:
      - packed:    actual storage (pair-packed bitstream for 12-bit indices)
      - container: one uint8/uint16 element per k-block (no bit-packing)

    At b=3 (N=4096, 12-bit indices) pair-packing stores two indices per 3
    bytes, so packed < container: 96 B vs 128 B per head vector. At b=2 and
    b=4 the two figures coincide. Norms are fp16 (2 B per head vector).
    """
    bits = int(n_levels - 1).bit_length()  # bits per block index
    blocks = d // k
    container_bytes = torch.empty(1, dtype=index_dtype(n_levels)).element_size()
    payload_packed = blocks * bits / 8  # bytes (pair-packed for 12-bit)
    payload_container = blocks * container_bytes  # bytes
    norm = 2  # fp16
    return {
        "bits_per_block": bits,
        "bytes_per_block_container": container_bytes,
        "payload_bytes_per_head_vector": payload_packed,
        "payload_bytes_per_head_vector_container": payload_container,
        "norm_bytes_per_head_vector": norm,
        "total_bytes_per_head_vector": payload_packed + norm,
        "fp16_bytes_per_head_vector": d * 2,
    }
