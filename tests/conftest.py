"""Shared fixtures: small trained operating points, no model or GPU needed."""

import pytest
import torch

from fibquant.codebook import build_codebook, build_directions, build_radii, build_rotation
from fibquant.spec import FibQuantSpec


def _untrained_codebook(d: int, k: int, n_levels: int) -> torch.Tensor:
    """The radii*directions init, with no Lloyd-Max polish.

    Fixtures that only exercise storage/packing (not quantization quality)
    use this instead of build_codebook, so a 4096- or 65536-level codebook
    never allocates a (samples, N) training score matrix merely to test
    storage -- at these N, k the init alone is already highly accurate (see
    packed_spec below), so no polish is needed for the fixtures that use it.
    """
    radii = build_radii(d, k, n_levels)
    directions = build_directions(k, n_levels)
    return radii.unsqueeze(-1) * directions


@pytest.fixture(scope="session")
def small_spec() -> FibQuantSpec:
    """d=8, k=2, N=16 (uint8 containers; 2 bits/coord)."""
    codebook = build_codebook(
        8, 2, 16, seed=0, restarts=1, lloyd_iters=4, m_factor=10
    ).codebook
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
    Codebook is the untrained radii/directions init (see
    _untrained_codebook): this fixture exists to exercise 12-bit
    storage/pack correctness, not codebook training quality (small_spec
    already covers real Lloyd-Max training, at a size that keeps it cheap),
    so building it must not train a 4096-level codebook nor allocate a
    (samples, 4096) training score matrix.
    """
    codebook = _untrained_codebook(8, 2, 4096)
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
