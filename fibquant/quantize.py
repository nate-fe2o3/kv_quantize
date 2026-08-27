"""Batched FibQuant encode/decode for KV cache tensors.

Tensors are (batch, heads, seq, d) with d = head dim. Each head-vector is
independently encoded: fp16 norm header + one uint8/uint16 container element
(block index) per k-block.
"""

from __future__ import annotations

import torch


def index_dtype(n_levels: int) -> torch.dtype:
    """Smallest container dtype that can hold codeword indices in [0, n_levels)."""
    if n_levels <= 2**8:
        return torch.uint8
    if n_levels <= 2**16:
        return torch.uint16
    raise ValueError(f"n_levels={n_levels} exceeds uint16; not supported")


_DEFAULT_SCORE_BYTES = 1 << 30  # 1 GiB budget for the fp32 (chunk, blocks, N) scores


@torch.no_grad()
def encode(
    x: torch.Tensor,
    codebook: torch.Tensor,
    rotation: torch.Tensor,
    k: int,
    max_score_bytes: int = _DEFAULT_SCORE_BYTES,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize x -> (indices, norms).

    indices: (B, H, S, d/k) uint8/uint16, one codeword index per k-block
             (dtype chosen from codebook size; see index_dtype).
    norms:   (B, H, S) fp16, per-vector L2 norms.

    The nearest-codeword search avoids materializing the full (B,H,S,d/k,N)
    distance tensor. Since ||y||^2 is constant per block, minimizing
    ||y - c_j||^2 is equivalent to maximizing <[2y, -1], [c_j, ||c_j||^2]>,
    i.e. a single matmul in R^{k+1} followed by argmax. The score tensor is
    computed row-chunked so peak memory stays ~max_score_bytes regardless of
    N; the results are identical to the previous one-shot computation.
    """
    norms = x.norm(dim=-1)
    safe = norms.clamp(min=1e-6)
    y = x.to(torch.float32) / safe.unsqueeze(-1)
    y = y @ rotation.to(x.device)  # (B, H, S, d)
    y = y.view(*y.shape[:-1], -1, k)  # (B, H, S, d/k, k)

    codebook_aug = torch.cat(
        [codebook, codebook.square().sum(-1, keepdim=True)], dim=-1
    ).to(x.device)  # (N, k+1) -> [c, ||c||^2]
    y_a = torch.cat([2 * y, -torch.ones_like(y[..., :1])], dim=-1)  # (..., d/k, k+1)

    n_levels = codebook.shape[0]
    dtype = index_dtype(n_levels)
    blocks = y_a.shape[-2]
    rows = y_a.numel() // (blocks * y_a.shape[-1])
    y_a = y_a.reshape(rows, blocks, -1)  # (rows, d/k, k+1)
    indices = torch.empty(rows, blocks, dtype=dtype, device=x.device)
    chunk_rows = max(1, max_score_bytes // (blocks * n_levels * 4))
    for start in range(0, rows, chunk_rows):
        end = min(start + chunk_rows, rows)
        scores = y_a[start:end] @ codebook_aug.t()  # (chunk, d/k, N)
        indices[start:end] = scores.argmax(-1).to(dtype)

    indices = indices.view(*x.shape[:-1], blocks)
    norms_f16 = safe.to(torch.float16)
    return indices, norms_f16


@torch.no_grad()
def decode(
    indices: torch.Tensor,
    norms: torch.Tensor,
    codebook: torch.Tensor,
    rotation: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize (indices, norms) back to a (B, H, S, d) tensor."""
    y = codebook.to(indices.device)[indices.long()]  # (B, H, S, d/k, k)
    y = y.reshape(*y.shape[:-2], -1)  # (B, H, S, d)
    x = y @ rotation.to(indices.device).t()
    x = x * norms.to(x.dtype).unsqueeze(-1)
    return x.to(dtype)


def bytes_per_token(d: int, k: int, n_levels: int) -> dict[str, float]:
    """Payload bytes per token per (K or V) head vector, plus the fp16 reference.

    Two payload figures are reported:
      - packed:    idealized bitstream size (bits may not fill a container)
      - container: actual storage (one uint8/uint16 element per k-block)

    At b=2 (N=256) and b=4 (N=65536) container == packed. At b=3 (N=4096,
    12-bit indices) the uint16 container makes storage identical to b=4
    unless true bit-packing is implemented.
    """
    bits = int(n_levels - 1).bit_length()  # bits per block index
    blocks = d // k
    container_bytes = torch.empty(1, dtype=index_dtype(n_levels)).element_size()
    payload_packed = blocks * bits / 8  # bytes
    payload_container = blocks * container_bytes  # bytes
    norm = 2  # fp16
    return {
        "bits_per_block": bits,
        "bytes_per_block_container": container_bytes,
        "payload_bytes_per_head_vector": payload_packed,
        "payload_bytes_per_head_vector_container": payload_container,
        "norm_bytes_per_head_vector": norm,
        "total_bytes_per_head_vector": payload_container + norm,
        "fp16_bytes_per_head_vector": d * 2,
    }
