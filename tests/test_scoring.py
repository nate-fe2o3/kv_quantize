"""Chunked nearest-codeword scorer: identity, chunking equivalence, budgets."""

import pytest
import torch

from fibquant.scoring import chunk_rows_for, min_pairwise_distance, nearest


def test_matches_brute_force(small_spec):
    torch.manual_seed(1)
    samples = torch.randn(37, 2)
    indices, dist2 = nearest(samples, small_spec.codebook)
    brute_dist2 = torch.cdist(samples, small_spec.codebook).pow(2).min(-1)
    assert torch.equal(indices, brute_dist2.indices)
    assert torch.allclose(dist2, brute_dist2.values, atol=1e-5)


def test_chunk_size_does_not_change_result(small_spec):
    torch.manual_seed(2)
    samples = torch.randn(53, 2)
    i_full, d_full = nearest(samples, small_spec.codebook)
    i_1, d_1 = nearest(samples, small_spec.codebook, chunk_rows=1)
    i_7, d_7 = nearest(samples, small_spec.codebook, chunk_rows=7)
    assert torch.equal(i_full, i_1)
    assert torch.equal(i_full, i_7)
    assert torch.allclose(d_full, d_1, atol=1e-6)
    assert torch.allclose(d_full, d_7, atol=1e-6)


def test_dist2_is_norm2_minus_max_score(small_spec):
    torch.manual_seed(3)
    samples = torch.randn(9, 2)
    indices, dist2 = nearest(samples, small_spec.codebook)
    # augmented identity: ||s - c||^2 = ||s||^2 - max_c(2<s,c> - ||c||^2)
    scores = 2 * (samples @ small_spec.codebook.t()) - small_spec.codebook.square().sum(-1)
    manual = samples.square().sum(-1) - scores.max(-1).values
    assert torch.allclose(dist2, manual, atol=1e-6)


def test_tiny_budget_still_correct_and_bounded(small_spec):
    torch.manual_seed(4)
    samples = torch.randn(100, 2)
    i_big, _ = nearest(samples, small_spec.codebook, max_score_bytes=1 << 30)
    i_tiny, _ = nearest(samples, small_spec.codebook, max_score_bytes=64)  # ~1 row/chunk
    assert torch.equal(i_big, i_tiny)


def test_chunk_rows_for_budget():
    # (chunk, N) fp32 matrix must stay ~max_score_bytes
    assert chunk_rows_for(4096, 1 << 30) == (1 << 30) // (4096 * 4)
    assert chunk_rows_for(4096, 4096 * 4) == 1
    assert chunk_rows_for(4096, 0) == 1  # never below one row


def test_rejects_codebook_beyond_uint16():
    samples = torch.randn(3, 2)
    codebook = torch.randn(1 << 17, 2)
    with pytest.raises(ValueError, match="exceeds uint16"):
        nearest(samples, codebook)


def test_min_pairwise_distance_matches_brute_force(small_spec):
    torch.manual_seed(5)
    kn = small_spec.codebook[:16]
    d = min_pairwise_distance(kn)
    norm2 = kn.square().sum(-1)
    full = norm2.unsqueeze(-1) + norm2.unsqueeze(0) - 2 * (kn @ kn.t())
    full.fill_diagonal_(float("inf"))
    expected = full.sqrt().min(-1).values
    assert torch.allclose(d, expected, atol=1e-5)
    # bounded memory: chunked form never builds the (N, N, k) tensor
    d_chunked = min_pairwise_distance(kn, max_score_bytes=64)
    assert torch.allclose(d, d_chunked, atol=1e-5)
