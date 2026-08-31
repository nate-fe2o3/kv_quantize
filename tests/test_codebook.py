"""Offline codebook construction: fib_phi, rotation SO(d)/device, build MSE, RNG hygiene."""

import math

import pytest
import torch

from fibquant.codebook import (
    CodebookResult,
    build_codebook,
    build_rotation,
    fib_phi,
    sample_spherical_beta,
)
from fibquant.scoring import nearest


def test_fib_phi_root_of_defining_equation():
    for k in range(1, 6):
        phi = fib_phi(k)
        assert math.isclose(phi ** (k + 1), phi + 1.0, rel_tol=1e-9)


def test_fib_phi_k1_is_golden_ratio():
    assert math.isclose(fib_phi(1), (1 + math.sqrt(5)) / 2)


def test_build_rotation_is_so_d_and_deterministic():
    q = build_rotation(8, seed=0)
    assert q.shape == (8, 8)
    assert torch.allclose(q.t() @ q, torch.eye(8), atol=1e-5)
    assert torch.linalg.det(q).item() > 0  # SO(d), not just O(d)
    q_again = build_rotation(8, seed=0)
    assert torch.equal(q, q_again)
    q_other_seed = build_rotation(8, seed=1)
    assert not torch.equal(q, q_other_seed)


def test_build_rotation_default_device_is_cpu():
    q = build_rotation(8, seed=0)
    assert q.device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_build_rotation_works_on_cuda():
    q = build_rotation(8, seed=0, device="cuda")
    assert q.device.type == "cuda"
    assert torch.allclose(q.t() @ q, torch.eye(8, device="cuda"), atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_build_codebook_works_on_cuda():
    result = build_codebook(
        8, 2, 16, seed=0, restarts=1, lloyd_iters=2, m_factor=8, device="cuda"
    )
    assert result.codebook.device.type == "cuda"
    assert result.counts.device.type == "cuda"


def test_sample_spherical_beta_does_not_mutate_global_rng():
    torch.manual_seed(123)
    before = torch.rand(4)

    torch.manual_seed(123)
    sample_spherical_beta(8, 2, 32, seed=999)
    after_call = torch.rand(4)

    assert torch.equal(before, after_call), "global RNG state leaked out of sample_spherical_beta"


def test_build_codebook_returns_typed_result_with_true_mse():
    result = build_codebook(
        8, 2, 16, seed=0, restarts=1, lloyd_iters=4, m_factor=20
    )
    assert isinstance(result, CodebookResult)
    assert result.codebook.shape == (16, 2)
    assert result.counts.shape == (16,)
    assert result.counts.sum().item() == 20 * 16  # m_factor * n_levels training samples

    # the true quantization MSE: mean squared distance from a fresh sample of
    # the same source to its nearest codeword under the returned codebook --
    # not the mean squared codeword *norm* the build script used to save
    # under the same name.
    probe = sample_spherical_beta(8, 2, 4096, seed=1234)
    _, dist2 = nearest(probe, result.codebook)
    expected_mse = dist2.mean().item()
    assert result.mse == pytest.approx(expected_mse, rel=0.35)
    assert result.mse >= 0.0

    # unpacking still works positionally: codebook, counts, mse = result
    codebook, counts, mse = result
    assert torch.equal(codebook, result.codebook)
    assert mse == result.mse
