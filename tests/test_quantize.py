"""Codec: encode/decode roundtrips, index dtypes, pair-packing, byte accounting."""

import pytest
import torch

from fibquant.quantize import (
    bytes_per_token,
    decode,
    encode,
    index_dtype,
    pack_indices,
    unpack_indices,
)


def test_index_dtype_boundaries():
    assert index_dtype(256) is torch.uint8
    assert index_dtype(4096) is torch.uint16  # 12-bit indices still need uint16 containers
    assert index_dtype(65536) is torch.uint16
    with pytest.raises(ValueError, match="exceeds uint16"):
        index_dtype(65537)


def test_encode_decode_roundtrip(small_spec):
    torch.manual_seed(0)
    x = torch.randn(2, 2, 16, 8)
    indices, norms = encode(x, small_spec.codebook, small_spec.rotation, small_spec.k)
    assert indices.shape == (2, 2, 16, 4)  # d/k = 4 blocks
    assert indices.dtype is torch.uint8
    x_hat = decode(indices, norms, small_spec.codebook, small_spec.rotation)
    cos = torch.nn.functional.cosine_similarity(x, x_hat, dim=-1)
    assert cos.mean() > 0.9


def test_decode_empty_output_shape_is_blocks_times_k(small_spec):
    """Regression: an empty cache (e.g. after crop(0)) must decode to a
    logical last dim of blocks * k (== d), not codebook.shape[1] (== k
    alone) -- the latter silently truncated every empty decode to k
    coordinates instead of d."""
    blocks = small_spec.d // small_spec.k
    indices = torch.empty(2, 3, 0, blocks, dtype=torch.uint8)
    norms = torch.empty(2, 3, 0, dtype=torch.float16)
    x_hat = decode(indices, norms, small_spec.codebook, small_spec.rotation, dtype=torch.float32)
    assert x_hat.shape == (2, 3, 0, small_spec.d)


def test_unpack_indices_empty_packed_shape(packed_spec):
    """Regression: unpacking an empty 12-bit packed tensor must not crash
    (reshape(..., -1, 3) of 0 elements is ambiguous) and must recover the
    logical block count (packed_dim * 2 // 3), not silently drop it."""
    blocks = packed_spec.d // packed_spec.k  # 4
    packed_dim = blocks * 3 // 2  # 6, pair-packed
    packed = torch.empty(2, 3, 0, packed_dim, dtype=torch.uint8)
    unpacked = unpack_indices(packed, packed_spec.n_levels)
    assert unpacked.shape == (2, 3, 0, blocks)
    assert unpacked.dtype is torch.uint16


def test_encode_packed_roundtrip(packed_spec):
    torch.manual_seed(0)
    x = torch.randn(2, 1, 8, 8)
    indices, norms = encode(x, packed_spec.codebook, packed_spec.rotation, packed_spec.k)
    assert indices.dtype is torch.uint16  # logical indices before packing
    packed = pack_indices(indices, packed_spec.n_levels)
    assert packed.dtype is torch.uint8
    assert packed.shape[-1] == indices.shape[-1] * 3 // 2  # 2 x 12-bit per 3 bytes
    assert torch.equal(unpack_indices(packed, packed_spec.n_levels), indices.to(torch.uint16))


def test_pack_unpack_roundtrip_all_indices(packed_spec):
    """Every index value must survive the pair-packing bitstream.

    One row of 4096 consecutive indices covers all byte positions: each pair
    (e, o) spans the full b0 = e % 256, b1 bytes (e//256 * 16 + o % 16), and
    b2 = o // 16 ranges.
    """
    indices = torch.arange(4096, dtype=torch.long).to(torch.uint16).reshape(1, 1, 1, 4096)
    packed = pack_indices(indices, packed_spec.n_levels)
    un = unpack_indices(packed, packed_spec.n_levels).reshape(-1)
    assert torch.equal(indices.reshape(-1), un), "pack roundtrip corrupts indices"


def test_pack_identity_for_container_sizes(small_spec, random_wide_spec):
    """8-bit and 16-bit containers are already minimal; pack is the identity."""
    idx8 = torch.randint(0, 16, (1, 1, 4, 4), dtype=torch.uint8)
    assert pack_indices(idx8, 16) is idx8
    assert unpack_indices(idx8, 16) is idx8
    idx16 = torch.randint(0, 65536, (1, 1, 4, 4), dtype=torch.uint16)
    assert pack_indices(idx16, 65536) is idx16
    assert unpack_indices(idx16, 65536) is idx16


def test_pack_rejects_odd_blocks():
    idx = torch.zeros(1, 1, 1, 3, dtype=torch.uint16)
    with pytest.raises(ValueError, match="even number of blocks"):
        pack_indices(idx, 4096)


def test_bytes_per_token_consistent_with_storage():
    # b=2: 1 byte per 4-coord block, 64 blocks -> 64 B payload + 2 B norm (d=256)
    b2 = bytes_per_token(d=256, k=4, n_levels=256)
    assert b2["bytes_per_block_container"] == 1
    assert b2["payload_bytes_per_head_vector"] == 64
    assert b2["total_bytes_per_head_vector"] == 66
    # b=3: 12-bit indices, pair-packed -> 96 B vs 128 B container
    b3 = bytes_per_token(d=256, k=4, n_levels=4096)
    assert b3["bits_per_block"] == 12
    assert b3["payload_bytes_per_head_vector"] == 96
    assert b3["payload_bytes_per_head_vector_container"] == 128
    # b=4: 16-bit exact
    b4 = bytes_per_token(d=256, k=4, n_levels=65536)
    assert b4["payload_bytes_per_head_vector"] == 128
    assert b4["payload_bytes_per_head_vector_container"] == 128
