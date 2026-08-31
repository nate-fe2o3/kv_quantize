"""KV payload: storage format, protocol verbs, byte accounting, crop semantics."""

import torch

from fibquant.payload import KVPayload


def _states(batch=2, heads=3, seq=8, d=8):
    torch.manual_seed(0)
    return torch.randn(batch, heads, seq, d), torch.randn(batch, heads, seq, d)


def test_update_decode_roundtrip(small_spec):
    pl = KVPayload(small_spec)
    k, v = _states()
    pl.update(k, v)
    keys, values = pl.decode_all(dtype=torch.float32)
    assert pl.seq_length == 8
    assert torch.nn.functional.cosine_similarity(k, keys, dim=-1).mean() > 0.9
    assert torch.nn.functional.cosine_similarity(v, values, dim=-1).mean() > 0.9


def test_crop_matches_dynamic_layer_semantics(small_spec):
    pl = KVPayload(small_spec)
    k, v = _states()
    pl.update(k, v)
    before, _ = pl.decode_all()

    pl.crop(5)  # positive: keep first 5 (transformers DynamicLayer contract)
    assert pl.seq_length == 5
    assert torch.equal(pl.decode_all()[0], before[..., :5, :])

    pl.crop(-3)  # negative: remove 3 from the end -> 2 remain
    assert pl.seq_length == 2
    assert torch.equal(pl.decode_all()[0], before[..., :2, :])

    pl.crop(100)  # beyond seq: no-op
    assert pl.seq_length == 2
    pl.crop(0)  # base DynamicLayer.crop(0) empties the cache -- must match
    assert pl.seq_length == 0
    empty_keys, empty_values = pl.decode_all()
    # regression: an empty decode's last dim must be d (blocks * k), not k
    # alone (see fibquant.codec.PreparedCodec.decode).
    assert empty_keys.shape == (2, 3, 0, small_spec.d)
    assert empty_values.shape == (2, 3, 0, small_spec.d)


def test_crop_on_empty_is_noop(small_spec):
    pl = KVPayload(small_spec)
    pl.crop(3)
    assert pl.seq_length == 0


def test_append_after_crop(small_spec):
    pl = KVPayload(small_spec)
    k, v = _states(seq=4)
    pl.update(k, v)
    pl.crop(2)
    pl.update(k, v)  # encode of new tokens appends after the cropped prefix
    assert pl.seq_length == 6


def test_reorder_select_repeat_preserve_decode(small_spec):
    pl = KVPayload(small_spec)
    k, v = _states()
    pl.update(k, v)
    before, _ = pl.decode_all()

    beam = torch.tensor([1, 0])
    pl.reorder_cache(beam)
    assert torch.equal(pl.decode_all()[0], before[beam])

    pl.batch_select_indices(torch.tensor([0]))
    assert pl.decode_all()[0].shape[0] == 1

    pl.batch_repeat_interleave(2)
    got = pl.decode_all()[0]
    assert got.shape == (2, 3, 8, 8)
    assert torch.equal(got[0], got[1])


def test_storage_and_fp16_accounting(small_spec):
    pl = KVPayload(small_spec)
    k, v = _states(batch=2, heads=3, seq=8)
    pl.update(k, v)
    # indices: 2*3*8 rows x 4 blocks x 1 byte (uint8) per K and V
    assert pl.stored_bytes() == 2 * (2 * 3 * 8 * 4 * 1 + 2 * 3 * 8 * 2)
    assert pl.fp16_bytes() == 2 * 8 * 8 * 2 * 2 * 3
    pl.reset()
    assert pl.stored_bytes() == 0 and pl.seq_length == 0


def test_packed_storage_accounting(packed_spec):
    pl = KVPayload(packed_spec)
    k, v = _states(batch=2, heads=3, seq=8)
    pl.update(k, v)
    # 12-bit pair-packed: 4 blocks -> 6 bytes per head vector, fp16 norm 2 B
    assert pl.stored_bytes() == 2 * (2 * 3 * 8 * 6 + 2 * 3 * 8 * 2)
    assert pl.key_indices.dtype is torch.uint8
    assert pl.key_indices.shape[-1] == 6
    keys, _ = pl.decode_all(dtype=torch.float32)
    assert torch.nn.functional.cosine_similarity(k, keys, dim=-1).mean() > 0.99


def test_packed_crop_zero_decode_empty_shape(packed_spec):
    """Regression: cropping a 12-bit pair-packed payload to empty must decode
    to (batch, heads, 0, d), not crash. unpack_indices used to reshape the
    packed bytes into (..., -1, 3), which is ambiguous for 0 elements."""
    pl = KVPayload(packed_spec)
    k, v = _states(batch=2, heads=3, seq=8)
    pl.update(k, v)
    pl.crop(0)
    assert pl.seq_length == 0
    keys, values = pl.decode_all(dtype=torch.float32)
    assert keys.shape == (2, 3, 0, packed_spec.d)
    assert values.shape == (2, 3, 0, packed_spec.d)
