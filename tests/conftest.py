"""Shared fixtures: small trained operating points, no model or GPU needed."""

import pytest
import torch

from fibquant import build_codebook, build_rotation
from fibquant.spec import FibQuantSpec


@pytest.fixture(scope="session")
def small_spec() -> FibQuantSpec:
    """d=8, k=2, N=16 (uint8 containers; 2 bits/coord)."""
    torch.manual_seed(0)
    codebook = build_codebook(8, 2, 16, seed=0, restarts=1, lloyd_iters=4, m_factor=10)
    return FibQuantSpec(
        codebook=codebook,
        rotation=build_rotation(8, seed=0),
        d=8,
        k=2,
        n_levels=16,
    )


@pytest.fixture(scope="session")
def packed_spec() -> FibQuantSpec:
    """d=8, k=2, N=4096 (12-bit pair-packed uint8 storage; 6 bits/coord).

    Blocks = 4 (even), so the pair-packing path is exercised end to end.
    """
    torch.manual_seed(0)
    codebook = build_codebook(8, 2, 4096, seed=0, restarts=1, lloyd_iters=2, m_factor=8)
    return FibQuantSpec(
        codebook=codebook,
        rotation=build_rotation(8, seed=0),
        d=8,
        k=2,
        n_levels=4096,
    )


@pytest.fixture(scope="session")
def random_wide_spec() -> FibQuantSpec:
    """d=8, k=2, N=65536 (uint16 containers) — random codebook, untrained.

    Only used for the container/pack identity paths; quality is irrelevant.
    """
    torch.manual_seed(0)
    codebook = torch.randn(65536, 2)
    codebook = codebook / codebook.norm(dim=-1, keepdim=True)
    return FibQuantSpec(
        codebook=codebook,
        rotation=build_rotation(8, seed=0),
        d=8,
        k=2,
        n_levels=65536,
    )
