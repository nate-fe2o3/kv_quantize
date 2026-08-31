"""Offline codebook and rotation construction for FibQuant.

Implements the radial-angular codebook of Lee & Kim, "FibQuant: Universal Vector
Quantization for Random-Access KV-Cache Compression" (arXiv:2605.11478):

  - radii: Beta-quantile companding of the spherical-Beta source f_{d,k},
           beta_{d,k} = (k/(k+2)) * (d-k-2)/2 + 1
  - directions: Roberts-Kronecker rank-one sequence (Fibonacci generalization)
  - polish: multi-restart Lloyd-Max on samples from f_{d,k}
  - rotation: one shared Haar-random orthogonal matrix (d x d)

All of it is deterministic given the seed and shared across layers/heads/prompts.

Every builder below takes an explicit `device` (default "cpu") and is CUDA-
capable: generators, samples, and index/accumulator tensors are all placed on
that device, so `build_codebook(..., device="cuda")` trains without any
CPU<->GPU traffic. Randomness is drawn from local `torch.Generator` instances
(or, where the underlying distribution has no `generator=` parameter,
`torch.random.fork_rng` around a `torch.manual_seed`, which snapshots and
restores the global RNG state) rather than bare `torch.manual_seed` calls, so
building a codebook does not perturb whatever the caller's global RNG state
was doing.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy
import torch

from .scoring import DEFAULT_SCORE_BYTES, nearest


class CodebookResult(NamedTuple):
    """Outcome of one `build_codebook()` call.

    Replaces the old boolean-shaped return (either a tensor or a bare
    `(codebook, counts)` tuple) with a typed, self-describing object:

      - codebook: (N, k) fp32, the winning restart's polished codebook.
      - counts: (N,) int64, per-codeword sample counts from the winning
        restart's last Lloyd-Max iteration (0 = dead codeword).
      - mse: the true quantization MSE -- mean squared distance from the
        training samples to their nearest codeword under the winning
        codebook (`_mean_sq_err`) -- NOT the mean squared codeword *norm*,
        which `scripts/build_codebook.py` used to save under the same "mse"
        name by mistake.

    A plain NamedTuple so `codebook, counts, mse = build_codebook(...)` still
    works positionally alongside the named `.codebook` / `.counts` / `.mse`
    access.
    """

    codebook: torch.Tensor
    counts: torch.Tensor
    mse: float


def fib_phi(k: int) -> float:
    """Positive root of x^(k+1) = x + 1 (golden ratio generalized).

    Newton's method via scipy.optimize.root_scalar: same update rule
    (f, f') and initial guess as the original hand-rolled loop, so the root
    (and everything derived from it) is unchanged; scipy just owns the
    convergence bookkeeping. No randomness involved, so this is deterministic
    regardless of device.
    """
    if k == 1:
        return (1.0 + math.sqrt(5.0)) / 2.0
    from scipy.optimize import root_scalar

    sol = root_scalar(
        lambda x: x ** (k + 1) - x - 1.0,
        fprime=lambda x: (k + 1.0) * x**k - 1.0,
        x0=1.3,
        method="newton",
        xtol=1e-14,
        maxiter=200,
    )
    return sol.root


def build_rotation(d: int, seed: int = 0, device: torch.device | str = "cpu") -> torch.Tensor:
    """Haar-random orthogonal matrix of shape (d, d), SO(d) (det = +1).

    Uses torch.nn.init.orthogonal_ (QR with the Mezzadri sign correction for
    uniform Haar measure) driven by a local Generator, so no global RNG state
    is touched and the result is reproducible per (seed, device). orthogonal_
    does not itself guarantee det = +1 (it produces O(d), not SO(d)), so a
    column flip enforces the same SO(d) orientation the original
    randn+qr+det-flip implementation guaranteed.
    """
    device = torch.device(device)
    g = torch.Generator(device=device).manual_seed(seed)
    q = torch.empty(d, d, device=device)
    torch.nn.init.orthogonal_(q, generator=g)
    if torch.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def sample_spherical_beta(
    d: int, k: int, n: int, seed: int, device: torch.device | str = "cpu"
) -> torch.Tensor:
    """Sample n points from f_{d,k}: R^2 ~ Beta(k/2, (d-k)/2), U uniform on S^{k-1}.

    torch.distributions.Beta has no `generator=` argument, so determinism
    still routes through `torch.manual_seed`; `torch.random.fork_rng` saves
    and restores the global RNG state around it, so callers' global RNG is
    left exactly as they found it once this returns.
    """
    device = torch.device(device)
    fork_devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(seed)
        r2 = torch.distributions.Beta(k / 2, (d - k) / 2).sample((n,)).to(device)
        z = torch.randn(n, k, device=device)
    u = z / z.norm(dim=-1, keepdim=True)
    return r2.sqrt().unsqueeze(-1) * u


def build_radii(d: int, k: int, n_levels: int, device: torch.device | str = "cpu") -> torch.Tensor:
    """Beta-quantile radii, r_n = sqrt(BetaInv((n-1/2)/N; k/2, beta_{d,k})).

    The quantile function itself is numpy/scipy (CPU-only); `.to(device)`
    places the result on the requested device once, no per-call recompute.
    """
    import scipy.stats

    q = (numpy.arange(n_levels, dtype=numpy.float64) + 0.5) / n_levels
    beta = k / (k + 2) * (d - k - 2) / 2 + 1
    r2 = scipy.stats.beta.ppf(q, k / 2, beta)
    return torch.from_numpy(numpy.sqrt(r2).astype(numpy.float32)).to(device)


def build_directions(k: int, n_levels: int, device: torch.device | str = "cpu") -> torch.Tensor:
    """Roberts-Kronecker rank-one directions on S^{k-1}.

    Purely deterministic (a low-discrepancy sequence, no sampling), so only a
    device placement is needed -- no generator/seed involved.
    """
    device = torch.device(device)
    phi = fib_phi(k)
    n = torch.arange(1, n_levels + 1, dtype=torch.float64, device=device)
    j = torch.arange(1, k + 1, dtype=torch.float64, device=device)
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
    index_add_/bincount accumulators, all placed on samples.device (CUDA-
    capable). Returns (codebook, counts), where counts = samples assigned to
    each codeword after the last iteration (zeros are dead codewords).
    """
    n, k = samples.shape
    n_levels = codebook.shape[0]
    device = samples.device
    g = torch.Generator(device=device).manual_seed(seed)
    c = codebook.clone()
    counts = None
    for _ in range(iters):
        assign, d2_assigned = nearest(samples, c, max_score_bytes=max_score_bytes)

        centroids = torch.zeros(n_levels, k, dtype=samples.dtype, device=device)
        centroids.index_add_(0, assign, samples)  # sum of samples per cell
        counts = torch.bincount(assign, minlength=n_levels)
        centroids = centroids / counts.clamp_min(1).unsqueeze(-1)

        cell_mse = torch.zeros(n_levels, dtype=samples.dtype, device=device)
        cell_mse.scatter_add_(0, assign, d2_assigned)

        empty = counts == 0
        if empty.any():
            for j in empty.nonzero(as_tuple=True)[0]:
                i_split = int(cell_mse.argmax())
                perturb = torch.randn(k, generator=g, device=device, dtype=c.dtype) * 0.02
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
    device: torch.device | str = "cpu",
) -> CodebookResult:
    """Build the shared FibQuant codebook for the spherical-Beta source f_{d,k}.

    Returns a CodebookResult (codebook, counts, mse); see CodebookResult for
    what each field means. `device` places every generator/sample/accumulator
    tensor used during training (default "cpu"; CUDA-capable).
    """
    device = torch.device(device)
    radii = build_radii(d, k, n_levels, device=device)
    directions = build_directions(k, n_levels, device=device)
    init = radii.unsqueeze(-1) * directions  # (N, k)

    samples = sample_spherical_beta(d, k, m_factor * n_levels, seed, device=device)

    best_cb, best_mse, best_counts = None, float("inf"), None
    for restart in range(restarts):
        g = torch.Generator(device=device).manual_seed(seed + 1 + restart)
        rot = torch.empty(k, k, device=device)
        torch.nn.init.orthogonal_(rot, generator=g)  # diversifies restarts; orientation irrelevant here
        c, counts = _lloyd_max(
            init @ rot, samples, lloyd_iters, seed + 100 + restart, max_score_bytes
        )
        mse = _mean_sq_err(samples, c, max_score_bytes)
        if mse < best_mse:
            best_cb, best_mse, best_counts = c, mse, counts
    return CodebookResult(best_cb, best_counts, best_mse)
