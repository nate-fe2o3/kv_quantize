"""Build and save the shared FibQuant codebook + rotation matrix.

Databricks-only (checkpoints are written to the UC volume next to the model
checkpoints); run as a notebook or `python scripts/build_codebook.py` on the
cluster. Configure via the constants below — no CLI arguments.

Default: d=256 (Qwen3.5 full-attn head dim), k=4, N=256 -> b=2 bits/coord.
Higher-fidelity operating points: N_LEVELS=4096 (b=3) or N_LEVELS=65536
(b=4). Indices are stored as uint8/uint16 container elements according to the
codebook size (see fibquant.quantize.index_dtype).

Codebook quality diagnostics (min pairwise distance, dead-codeword fraction)
come from the shared scorer module, so their chunking heuristic matches the
build's.
"""

from __future__ import annotations

import torch

from fibquant import FibQuantSpec, build_codebook, build_rotation
from fibquant.scoring import min_pairwise_distance

# --- build configuration --------------------------------------------------
D = 256  # head dim
K = 4  # block size
N_LEVELS = 256  # codewords per block (b = log2(N) / k bits/coord)
SEED = 0
RESTARTS = 4
LLOYD_ITERS = 25
M_FACTOR = 30  # samples per codeword (samples = M_FACTOR * N_LEVELS)
SCORE_MB = 1024  # approx MB per (chunk, N) score matrix
OUT = f"/Volumes/security_engineering/nbutton/q34b/models/fibquant/fibquant_d{D}_k{K}_N{N_LEVELS}.pt"


def main() -> None:
    bpc = (N_LEVELS - 1).bit_length() / K
    print(f"building codebook d={D} k={K} N={N_LEVELS} -> b={bpc} bits/coord")
    max_score_bytes = SCORE_MB * 2**20

    codebook, counts = build_codebook(
        D,
        K,
        N_LEVELS,
        seed=SEED,
        restarts=RESTARTS,
        lloyd_iters=LLOYD_ITERS,
        m_factor=M_FACTOR,
        max_score_bytes=max_score_bytes,
        return_counts=True,
    )
    rotation = build_rotation(D, seed=SEED)

    min_dists = min_pairwise_distance(codebook, max_score_bytes=max_score_bytes)
    dead = int((counts == 0).sum())
    dead_frac = dead / N_LEVELS

    mse = (codebook**2).mean().item()
    spec = FibQuantSpec(codebook=codebook, rotation=rotation, d=D, k=K, n_levels=N_LEVELS)
    spec.save(OUT, seed=SEED, mse=mse)
    print(f"codebook radius range: [{codebook.norm(dim=-1).min():.4f}, {codebook.norm(dim=-1).max():.4f}]")
    print(f"min pairwise codeword distance: {min_dists.min():.4f}, mean: {min_dists.mean():.4f}")
    print(f"codeword mean-squared norm: {mse:.4f}")
    print(f"dead codewords: {dead}/{N_LEVELS} ({dead_frac:.2%})")
    if dead_frac > 0.01:
        print(
            "WARNING: >1% dead codewords -- consider a larger M_FACTOR or more LLOYD_ITERS"
        )
    print(f"saved to {OUT}")


if __name__ == "__main__":
    main()
