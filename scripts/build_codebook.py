"""Build and save the shared FibQuant codebook + rotation matrix.

Default: d=256 (Qwen3.5 full-attn head dim), k=4, N=256 -> b=2 bits/coord.
"""

from __future__ import annotations

import argparse

from fibquant import build_codebook, build_rotation, default_spec_path, save_spec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type=int, default=256, help="head dim")
    parser.add_argument("--k", type=int, default=4, help="block size")
    parser.add_argument("--n-levels", type=int, default=256, help="codewords per block")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--restarts", type=int, default=4)
    parser.add_argument("--lloyd-iters", type=int, default=25)
    parser.add_argument("--out", type=str, default=None, help="output checkpoint path")
    args = parser.parse_args()

    bpc = (args.n_levels - 1).bit_length() / args.k
    print(f"building codebook d={args.d} k={args.k} N={args.n_levels} -> b={bpc} bits/coord")

    codebook = build_codebook(
        args.d,
        args.k,
        args.n_levels,
        seed=args.seed,
        restarts=args.restarts,
        lloyd_iters=args.lloyd_iters,
    )
    rotation = build_rotation(args.d, seed=args.seed)

    diff = codebook.unsqueeze(0) - codebook.unsqueeze(1)
    d2 = (diff * diff).sum(-1)
    d2.diagonal().fill_(float("inf"))
    min_dists = d2.min(-1).values.sqrt()

    import torch

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
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
