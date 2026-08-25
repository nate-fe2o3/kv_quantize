"""Batched FibQuant encode/decode for KV cache tensors.

Tensors are (batch, heads, seq, d) with d = head dim. Each head-vector is
independently encoded: fp16 norm header + one uint8 block index per k-block.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def encode(
    x: torch.Tensor,
    codebook: torch.Tensor,
    rotation: torch.Tensor,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize x -> (indices, norms).

    indices: (B, H, S, d/k) uint8, one codeword index per k-block.
    norms:   (B, H, S) fp16, per-vector L2 norms.

    The nearest-codeword search avoids materializing the full (B,H,S,d/k,N)
    distance tensor. Since ||y||^2 is constant per block, minimizing
    ||y - c_j||^2 is equivalent to maximizing <[2y, -1], [c_j, ||c_j||^2]>,
    i.e. a single matmul in R^{k+1} followed by argmax.
    """
    norms = x.norm(dim=-1)
    safe = norms.clamp(min=1e-6)
    y = x.to(torch.float32) / safe.unsqueeze(-1)
    y = y @ rotation.to(x.device)  # (B, H, S, d)
    y = y.view(*y.shape[:-1], -1, k)  # (B, H, S, d/k, k)

    codebook_aug = torch.cat(
        [codebook, codebook.square().sum(-1, keepdim=True)], dim=-1
    )  # (N, k+1) -> [c, ||c||^2]
    y_a = torch.cat([2 * y, -torch.ones_like(y[..., :1])], dim=-1)  # (..., d/k, k+1)
    scores = y_a @ codebook_aug.t().to(x.device)  # (B, H, S, d/k, N)
    indices = scores.argmax(-1).to(torch.uint8)
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
    """Payload bytes per token per (K or V) head vector, plus the fp16 fp16 ratio."""
    bits = int(n_levels - 1).bit_length()  # bits per block index
    payload = (d // k) * bits / 8  # bytes
    norm = 2  # fp16
    return {
        "bits_per_block": bits,
        "payload_bytes_per_head_vector": payload,
        "norm_bytes_per_head_vector": norm,
        "total_bytes_per_head_vector": payload + norm,
        "fp16_bytes_per_head_vector": d * 2,
    }
