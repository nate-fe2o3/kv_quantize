"""Offline codebook and rotation construction for FibQuant.

Implements the radial-angular codebook of Lee & Kim, "FibQuant: Universal Vector
Quantization for Random-Access KV-Cache Compression" (arXiv:2605.11478):

  - radii: Beta-quantile companding of the spherical-Beta source f_{d,k},
           beta_{d,k} = (k/(k+2)) * (d-k-2)/2 + 1
  - directions: Roberts-Kronecker rank-one sequence (Fibonacci generalization)
  - polish: multi-restart Lloyd-Max on samples from f_{d,k}
  - rotation: one shared Haar-random orthogonal matrix (d x d)

All of it is deterministic given the seed and shared across layers/heads/prompts.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy
import torch


def fib_phi(k: int) -> float:
    """Positive root of x^(k+1) = x + 1 (golden ratio generalized)."""
    if k == 1:
        return (1.0 + math.sqrt(5.0)) / 2.0
    x = 1.3
    for _ in range(200):
        f = x ** (k + 1) - x - 1.0
        fp = (k + 1.0) * x**k - 1.0
        x = x - f / fp
    return x


def build_rotation(d: int, seed: int = 0) -> torch.Tensor:
    """Haar-random orthogonal matrix of shape (d, d)."""
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(d, d, generator=g)
    q, _ = torch.linalg.qr(z)
    if torch.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def sample_spherical_beta(d: int, k: int, n: int, seed: int) -> torch.Tensor:
    """Sample n points from f_{d,k}: R^2 ~ Beta(k/2, (d-k)/2), U uniform on S^{k-1}."""
    torch.manual_seed(seed)
    r2 = torch.distributions.Beta(k / 2, (d - k) / 2).sample((n,))
    z = torch.randn(n, k)
    u = z / z.norm(dim=-1, keepdim=True)
    return r2.sqrt().unsqueeze(-1) * u


def build_radii(d: int, k: int, n_levels: int) -> torch.Tensor:
    """Beta-quantile radii, r_n = sqrt(BetaInv((n-1/2)/N; k/2, beta_{d,k}))."""
    import scipy.stats

    q = (numpy.arange(n_levels, dtype=numpy.float64) + 0.5) / n_levels
    beta = k / (k + 2) * (d - k - 2) / 2 + 1
    r2 = scipy.stats.beta.ppf(q, k / 2, beta)
    return torch.from_numpy(numpy.sqrt(r2).astype(numpy.float32))


def build_directions(k: int, n_levels: int) -> torch.Tensor:
    """Roberts-Kronecker rank-one directions on S^{k-1}."""
    phi = fib_phi(k)
    n = torch.arange(1, n_levels + 1, dtype=torch.float64)
    j = torch.arange(1, k + 1, dtype=torch.float64)
    frac = torch.frac((n - 0.5).unsqueeze(-1) * (phi ** (-j)))
    z = torch.distributions.Normal(0, 1).icdf(frac)
    u = z / z.norm(dim=-1, keepdim=True)
    return u.to(torch.float32)


_DEFAULT_SCORE_MB = 1024  # approx MB of one (chunk, N) score matrix


def _chunk_rows(n_levels: int, score_mb: int = _DEFAULT_SCORE_MB) -> int:
    """Sample rows per score chunk so the (chunk, N) fp32 matrix is ~score_mb."""
    return max(1, int(score_mb * 2**20) // (n_levels * 4))


def _assign_chunked(
    sample_aug: torch.Tensor,
    sample_norm2: torch.Tensor,
    codebook_aug: torch.Tensor,
    chunk_rows: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Nearest-codeword assignment without materializing (n, N) or (n, N, k).

    With score(s, c) = 2<s, c> - ||c||^2, argmin_j ||s - c_j||^2 == argmax_j
    score(s, c_j), and the assigned distance is exactly ||s||^2 - max_score.
    score(c) is computed over row chunks so peak memory is O(chunk_rows x N).
    Returns (assign int64, assigned d2 fp32).
    """
    n = sample_aug.shape[0]
    assign = torch.empty(n, dtype=torch.int64)
    max_score = torch.empty(n, dtype=torch.float32)
    for start in range(0, n, chunk_rows):
        end = min(start + chunk_rows, n)
        scores = sample_aug[start:end] @ codebook_aug.t()  # (chunk, N)
        a = scores.argmax(-1)
        assign[start:end] = a
        max_score[start:end] = scores.gather(1, a.unsqueeze(-1)).squeeze(-1)
    return assign, sample_norm2 - max_score


def _mean_sq_err(
    samples: torch.Tensor,
    codebook: torch.Tensor,
    chunk_rows: int,
) -> float:
    """Mean squared distance to the nearest codeword (chunked, no (n, N, k))."""
    sample_aug = torch.cat([2.0 * samples, -torch.ones_like(samples[..., :1])], dim=-1)
    sample_norm2 = samples.square().sum(-1)
    codebook_aug = torch.cat([codebook, codebook.square().sum(-1, keepdim=True)], dim=-1)
    _, assigned_d2 = _assign_chunked(sample_aug, sample_norm2, codebook_aug, chunk_rows)
    return assigned_d2.mean().item()


def _lloyd_max(
    codebook: torch.Tensor,
    samples: torch.Tensor,
    iters: int,
    seed: int,
    chunk_rows: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Lloyd-Max polish with empty-cell repair (split highest-distortion cell).

    Uses the augmented-score identity so assignment needs no (n, N, k)
    distance tensor: only (chunk_rows, N) score matrices plus
    index_add_/bincount/scatter_add_ accumulators. Returns (codebook, counts),
    where counts = samples assigned to each codeword after the last iteration
    (zeros are dead codewords).
    """
    n, k = samples.shape
    n_levels = codebook.shape[0]
    g = torch.Generator().manual_seed(seed)
    c = codebook.clone()
    sample_aug = torch.cat([2.0 * samples, -torch.ones_like(samples[..., :1])], dim=-1)
    sample_norm2 = samples.square().sum(-1)
    counts = None
    for _ in range(iters):
        codebook_aug = torch.cat([c, c.square().sum(-1, keepdim=True)], dim=-1)
        assign, d2_assigned = _assign_chunked(sample_aug, sample_norm2, codebook_aug, chunk_rows)

        centroids = torch.zeros(n_levels, k, dtype=samples.dtype)
        centroids.index_add_(0, assign, samples)  # sum of samples per cell
        counts = torch.bincount(assign, minlength=n_levels)
        centroids = centroids / counts.clamp_min(1).unsqueeze(-1)

        cell_mse = torch.zeros(n_levels, dtype=samples.dtype)
        cell_mse.scatter_add_(0, assign, d2_assigned)

        empty = counts == 0
        if empty.any():
            for j in empty.nonzero(as_tuple=True)[0]:
                i_split = int(cell_mse.argmax())
                perturb = torch.randn(k, generator=g, dtype=c.dtype) * 0.02
                centroids[j] = centroids[i_split] + perturb
        c = centroids
    return c, counts


def build_codebook(
    d: int,
    k: int,
    n_levels: int,
    seed: int = 0,
    restarts: int = 4,
    lloyd_iters: int = 25,
    m_factor: int = 30,
    score_mb: int = _DEFAULT_SCORE_MB,
    return_counts: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Build the shared FibQuant codebook for the spherical-Beta source f_{d,k}.

    With return_counts=True, also return the per-codeword sample counts of the
    best restart (entries of 0 are dead codewords).
    """
    radii = build_radii(d, k, n_levels)
    directions = build_directions(k, n_levels)
    init = radii.unsqueeze(-1) * directions  # (N, k)

    samples = sample_spherical_beta(d, k, m_factor * n_levels, seed)
    chunk_rows = _chunk_rows(n_levels, score_mb)

    best_cb, best_mse, best_counts = None, float("inf"), None
    for restart in range(restarts):
        g = torch.Generator().manual_seed(seed + 1 + restart)
        rot = torch.linalg.qr(torch.randn(k, k, generator=g))[0]
        c, counts = _lloyd_max(init @ rot, samples, lloyd_iters, seed + 100 + restart, chunk_rows)
        mse = _mean_sq_err(samples, c, chunk_rows)
        if mse < best_mse:
            best_cb, best_mse, best_counts = c, mse, counts
    return (best_cb, best_counts) if return_counts else best_cb


def save_spec(
    path: str | Path,
    codebook: torch.Tensor,
    rotation: torch.Tensor,
    d: int,
    k: int,
    n_levels: int,
    seed: int,
    mse: float,
) -> None:
    """Persist codebook + rotation to disk as a single checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "codebook": codebook,
            "rotation": rotation,
            "d": d,
            "k": k,
            "n_levels": n_levels,
            "seed": seed,
            "mse": mse,
        },
        path,
    )


def load_spec(path: str | Path) -> dict:
    """Load a FibQuant spec checkpoint (see save_spec)."""
    return torch.load(path, map_location="cpu", weights_only=False)


def default_spec_path(d: int, k: int, n_levels: int) -> Path:
    """models/fibquant/fibquant_d{d}_k{k}_N{n_levels}.pt"""
    return Path(__file__).resolve().parent.parent / "models" / "fibquant" / f"fibquant_d{d}_k{k}_N{n_levels}.pt"
