"""Build and save the shared FibQuant codebook + rotation matrix.

Default: d=256 (Qwen3.5 full-attn head dim), k=4, N=256 -> b=2 bits/coord.
Higher-fidelity operating points: --n-levels 4096 (b=3) or --n-levels 65536
(b=4). Indices are stored as uint8/uint16 container elements according to the
codebook size (see fibquant.quantize.index_dtype).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Make the repo root importable even when run as "python scripts/foo.py".
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fibquant import build_codebook, build_rotation, default_spec_path, save_spec


def min_pairwise_distance(codebook: torch.Tensor, score_mb: int = 1024) -> torch.Tensor:
    """Minimum pairwise codeword distance via chunked Gram (no (N, N, k))."""
    n_levels, _ = codebook.shape
    norm2 = codebook.square().sum(-1)  # (N,)
    chunk = max(1, int(score_mb * 2**20) // (n_levels * 4))
    mins = []
    for start in range(0, n_levels, chunk):
        end = min(start + chunk, n_levels)
        rows = codebook[start:end]
        # d2[i, j] = ||c_i||^2 + ||c_j||^2 - 2 <c_i, c_j>
        d2 = norm2[start:end].unsqueeze(-1) + norm2.unsqueeze(0) - 2 * (rows @ codebook.t())
        d2.fill_diagonal_(float("inf"))
        d2 = d2.clamp_min(0.0)
        mins.append(d2.min(-1).values)
    return torch.cat(mins).sqrt()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type=int, default=256, help="head dim")
    parser.add_argument("--k", type=int, default=4, help="block size")
    parser.add_argument("--n-levels", type=int, default=256, help="codewords per block")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--restarts", type=int, default=4)
    parser.add_argument("--lloyd-iters", type=int, default=25)
    parser.add_argument("--m-factor", type=int, default=30, help="samples per codeword (samples = m_factor * N)")
    parser.add_argument("--score-mb", type=int, default=1024, help="approx MB per (chunk, N) score matrix")
    parser.add_argument("--out", type=str, default=None, help="output checkpoint path")
    args = parser.parse_args()

    bpc = (args.n_levels - 1).bit_length() / args.k
    print(f"building codebook d={args.d} k={args.k} N={args.n_levels} -> b={bpc} bits/coord")

    codebook, counts = build_codebook(
        args.d,
        args.k,
        args.n_levels,
        seed=args.seed,
        restarts=args.restarts,
        lloyd_iters=args.lloyd_iters,
        m_factor=args.m_factor,
        score_mb=args.score_mb,
        return_counts=True,
    )
    rotation = build_rotation(args.d, seed=args.seed)

    min_dists = min_pairwise_distance(codebook, score_mb=args.score_mb)
    dead = int((counts == 0).sum())
    dead_frac = dead / args.n_levels

    mse = (codebook**2).mean().item()
    out = args.out or str(default_spec_path(args.d, args.k, args.n_levels))
    save_spec(
        out,
        codebook=codebook,
        rotation=rotation,
        d=args.d,
        k=args.k,
        n_levels=args.n_levels,
        seed=args.seed,
        mse=mse,
    )
    print(f"codebook radius range: [{codebook.norm(dim=-1).min():.4f}, {codebook.norm(dim=-1).max():.4f}]")
    print(f"min pairwise codeword distance: {min_dists.min():.4f}, mean: {min_dists.mean():.4f}")
    print(f"codeword mean-squared norm: {mse:.4f}")
    print(f"dead codewords: {dead}/{args.n_levels} ({dead_frac:.2%})")
    if dead_frac > 0.01:
        print(
            "WARNING: >1% dead codewords -- consider a larger --m-factor or more --lloyd-iters"
        )
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
