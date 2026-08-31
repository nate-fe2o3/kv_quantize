"""Chunked nearest-codeword scoring shared by codebook construction and runtime encode.

The core identity (augmented inner product): for a sample s and codeword c,

    ||s - c||^2 = ||s||^2 - (2<s, c> - ||c||^2)

so argmin_j ||s - c_j||^2 == argmax_j score(s, c_j) with
score(s, c) = 2<s, c> - ||c||^2, and the assigned distance^2 is exactly
||s||^2 - max_score. Score matrices are computed over row chunks so peak
memory stays at O(chunk_rows x N) fp32 regardless of the number of samples.

This is the single implementation of that identity: offline construction
(Lloyd-Max assignment, final-MSE evaluation) and runtime encode all call
`nearest`, so the geometry used to train the codebook can never drift from
the geometry used to query it.

The module also owns the chunked pairwise-distance diagnostics used by
codebook-quality checks, since they share the same budget heuristic.

`augment_codebook` is split out so callers that query the *same* codebook
repeatedly (runtime encode via a prepared codec, see codec.py) can build the
`[c, ||c||^2]` augmentation once and pass it back into `nearest` via
`codebook_aug`, instead of paying for the concat + squared-norm reduction on
every call. Offline codebook construction, where the codebook changes every
Lloyd-Max iteration, keeps calling `nearest` without `codebook_aug` and gets
the identical (recomputed-each-time) behavior it always had.
"""

from __future__ import annotations

import torch

# Default fp32 bytes budget for one (chunk_rows, N) score matrix, ~1 GiB.
DEFAULT_SCORE_BYTES = 1 << 30


def chunk_rows_for(n_levels: int, max_score_bytes: int) -> int:
    """Sample rows per score chunk so the (chunk, N) fp32 matrix is ~max_score_bytes."""
    bytes_per_row = n_levels * 4  # fp32
    return max(1, max_score_bytes // bytes_per_row)


def augment_codebook(codebook: torch.Tensor) -> torch.Tensor:
    """Precompute the `nearest` augmentation `[c, ||c||^2]`, shape (N, k + 1).

    Pure function of the codebook alone (not the queried samples), so callers
    with a fixed codebook across many `nearest` calls (runtime encode) should
    compute this once and pass it back in as `nearest(..., codebook_aug=...)`.
    """
    return torch.cat([codebook, codebook.square().sum(-1, keepdim=True)], dim=-1)


def nearest(
    samples: torch.Tensor,
    codebook: torch.Tensor,
    max_score_bytes: int = DEFAULT_SCORE_BYTES,
    chunk_rows: int | None = None,
    codebook_aug: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Nearest-codeword assignment of each sample row against a codebook.

    Args:
        samples: (n, k) fp32.
        codebook: (N, k) fp32, any radius (need not be on the unit ball).
        max_score_bytes: fp32 budget for one (chunk, N) score matrix.
        chunk_rows: explicit chunk size override (tests force chunk=1).
            Defaults to the budget-derived size.
        codebook_aug: precomputed `augment_codebook(codebook)`, i.e. (N, k+1)
            with the squared-norm column already appended. Pass this in when
            the codebook is unchanged across many calls (see codec.py's
            PreparedCodec) to skip re-deriving it every time. Left as None
            (the default), it is derived from `codebook` as before -- this is
            what offline codebook construction does, since its codebook
            changes every Lloyd-Max iteration and there is nothing to reuse.

    Returns:
        (indices, dist2): indices int64 (n,) with argmin codeword per row;
        dist2 fp32 (n,) with the assigned squared distance. Always both —
        callers that only need indices discard dist2.
    """
    n, k = samples.shape
    n_levels, _ = codebook.shape
    if n_levels > 2**16:
        raise ValueError(f"n_levels={n_levels} exceeds uint16; not supported")

    # Augmented identity: score(s, c) = [2s, -1] . [c, ||c||^2].
    sample_aug = torch.cat([2.0 * samples, -torch.ones_like(samples[..., :1])], dim=-1)
    sample_norm2 = samples.square().sum(-1)
    codebook_aug = codebook_aug if codebook_aug is not None else augment_codebook(codebook)
    if codebook_aug.device != samples.device:
        codebook_aug = codebook_aug.to(samples.device)

    if chunk_rows is None:
        chunk_rows = chunk_rows_for(n_levels, max_score_bytes)
    chunk_rows = max(1, chunk_rows)

    indices = torch.empty(n, dtype=torch.int64, device=samples.device)
    max_score = torch.empty(n, dtype=torch.float32, device=samples.device)
    for start in range(0, n, chunk_rows):
        end = min(start + chunk_rows, n)
        scores = sample_aug[start:end] @ codebook_aug.t()  # (chunk, N)
        a = scores.argmax(-1)
        indices[start:end] = a
        max_score[start:end] = scores.gather(1, a.unsqueeze(-1)).squeeze(-1)
    return indices, sample_norm2 - max_score


def min_pairwise_distance(
    codebook: torch.Tensor,
    max_score_bytes: int = DEFAULT_SCORE_BYTES,
) -> torch.Tensor:
    """Per-codeword minimum distance to another codeword, via chunked Gram.

    Materializes nothing larger than (chunk, N) fp32, so it scales to
    N = 65536 (the O(N^2, k) one-shot form does not).
    """
    n_levels, _ = codebook.shape
    norm2 = codebook.square().sum(-1)  # (N,)
    chunk = chunk_rows_for(n_levels, max_score_bytes)
    mins = []
    for start in range(0, n_levels, chunk):
        end = min(start + chunk, n_levels)
        rows = codebook[start:end]
        # d2[i, j] = ||c_i||^2 + ||c_j||^2 - 2 <c_i, c_j>
        d2 = norm2[start:end].unsqueeze(-1) + norm2.unsqueeze(0) - 2 * (rows @ codebook.t())
        # Mask the self-distance: chunk row r is codeword start + r, at column
        # start + r — NOT at the matrix diagonal (which only aligns when
        # start == 0). fill_diagonal_ here would mask the wrong element.
        r = torch.arange(end - start)
        d2[r, start + r] = float("inf")
        d2 = d2.clamp_min(0.0)
        mins.append(d2.min(-1).values)
    return torch.cat(mins).sqrt()
