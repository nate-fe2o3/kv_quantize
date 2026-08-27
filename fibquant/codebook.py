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

import numpy
import torch

from .scoring import DEFAULT_SCORE_BYTES, nearest



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


def _mean_sq_err(
    samples: torch.Tensor,
    codebook: torch.Tensor,
    max_score_bytes: int = DEFAULT_SCORE_BYTES,
) -> float:
    """Mean squared distance to the nearest codeword (chunked via scoring.nearest)."""
    _, assigned_d2 = nearest(samples, codebook, max_score_bytes=max_score_bytes)
    return assigned_d2.mean().item()


def _lloyd_max(
    codebook: torch.Tensor,
    samples: torch.Tensor,
    iters: int,
    seed: int,
    max_score_bytes: int = DEFAULT_SCORE_BYTES,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Lloyd-Max polish with empty-cell repair (split highest-distortion cell).

    Assignment is delegated to scoring.nearest (augmented-score identity,
    chunked), so no (n, N, k) distance tensor is materialized; centroids use
    index_add_/bincount accumulators. Returns (codebook, counts), where counts
    = samples assigned to each codeword after the last iteration (zeros are
    dead codewords).
    """
    n, k = samples.shape
    n_levels = codebook.shape[0]
    g = torch.Generator().manual_seed(seed)
    c = codebook.clone()
    counts = None
    for _ in range(iters):
        assign, d2_assigned = nearest(samples, c, max_score_bytes=max_score_bytes)

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
    max_score_bytes: int = DEFAULT_SCORE_BYTES,
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

    best_cb, best_mse, best_counts = None, float("inf"), None
    for restart in range(restarts):
        g = torch.Generator().manual_seed(seed + 1 + restart)
        rot = torch.linalg.qr(torch.randn(k, k, generator=g))[0]
        c, counts = _lloyd_max(
            init @ rot, samples, lloyd_iters, seed + 100 + restart, max_score_bytes
        )
        mse = _mean_sq_err(samples, c, max_score_bytes)
        if mse < best_mse:
            best_cb, best_mse, best_counts = c, mse, counts
    return (best_cb, best_counts) if return_counts else best_cb
