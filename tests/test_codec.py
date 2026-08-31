"""Prepared codec: buffer-copy caching, per-device reuse, adapter equivalence."""

import torch

from fibquant.codec import PreparedCodec, prepared_codec_for, prepared_codec_for_spec
from fibquant.payload import KVPayload
from fibquant.quantize import decode as bare_decode
from fibquant.quantize import encode as bare_encode
from fibquant.spec import FibQuantSpec


def test_prepared_codec_matches_bare_tensor_adapters(small_spec):
    """codec.PreparedCodec must be exactly the encode/decode compatibility
    adapters in quantize.py delegate to -- same numbers, not just same shape."""
    torch.manual_seed(0)
    x = torch.randn(2, 2, 16, small_spec.d)
    codec = PreparedCodec.from_spec(small_spec)

    idx_codec, norms_codec = codec.encode(x)
    idx_bare, norms_bare = bare_encode(x, small_spec.codebook, small_spec.rotation, small_spec.k)
    assert torch.equal(idx_codec, idx_bare)
    assert torch.equal(norms_codec, norms_bare)

    x_hat_codec = codec.decode(idx_codec, norms_codec, dtype=torch.float32)
    x_hat_bare = bare_decode(idx_bare, norms_bare, small_spec.codebook, small_spec.rotation, dtype=torch.float32)
    assert torch.equal(x_hat_codec, x_hat_bare)


def test_prepared_codec_decode_empty_shape(small_spec):
    """Same empty-decode regression as test_quantize.py, at the codec level."""
    codec = PreparedCodec.from_spec(small_spec)
    blocks = small_spec.d // small_spec.k
    indices = torch.empty(2, 3, 0, blocks, dtype=torch.uint8)
    norms = torch.empty(2, 3, 0, dtype=torch.float16)
    out = codec.decode(indices, norms, dtype=torch.float32)
    assert out.shape == (2, 3, 0, small_spec.d)


def test_prepared_codec_for_reuses_same_device_instance(small_spec):
    """Fetching the codec for the same device twice must return the exact
    same object (no repeated deep-copy/`.to()`, no repeated augmented-codebook
    build) -- this is the "move tensors once per device" contract."""
    base = PreparedCodec.from_spec(small_spec)
    cache: dict[torch.device, PreparedCodec] = {}
    first = prepared_codec_for(base, "cpu", cache)
    second = prepared_codec_for(base, "cpu", cache)
    assert first is second
    assert len(cache) == 1
    # the base already lives on cpu, so no copy was even needed
    assert first is base


def test_prepared_codec_for_deep_copies_augmented_codebook_by_value():
    """A fresh device entry must carry the *same* augmented codebook values as
    the base (no recompute), while being an independent tensor (a real copy,
    not aliasing the base's buffers)."""
    import copy

    codebook = torch.randn(16, 2)
    base = PreparedCodec(codebook, torch.eye(2), k=2, n_levels=16)
    other = copy.deepcopy(base)
    assert torch.equal(base.codebook_aug, other.codebook_aug)
    assert base.codebook_aug.data_ptr() != other.codebook_aug.data_ptr()


def test_kvpayload_reuses_prepared_codec_across_calls(small_spec):
    """KVPayload must not rebuild its PreparedCodec (device copy or augmented
    codebook) on every update()/decode_all() call -- the cache entry for a
    device, once populated, is reused."""
    pl = KVPayload(small_spec)
    torch.manual_seed(0)
    k = torch.randn(2, 3, 4, small_spec.d)
    v = torch.randn(2, 3, 4, small_spec.d)
    pl.update(k, v)
    codec_after_first = pl._codec_for(torch.device("cpu"))
    pl.update(k, v)
    codec_after_second = pl._codec_for(torch.device("cpu"))
    pl.decode_all()
    codec_after_decode = pl._codec_for(torch.device("cpu"))
    assert codec_after_first is codec_after_second is codec_after_decode


def test_kvpayloads_sharing_one_spec_share_one_prepared_codec(small_spec):
    """Regression: a FibQuantCache hands the *same* spec instance to every
    per-layer KVPayload, so N full-attention layers must share one
    PreparedCodec per device, not clone one each (the memory a shared spec
    is meant to save). Two independent spec instances, even with identical
    field values, must never be conflated."""
    p1 = KVPayload(small_spec)
    p2 = KVPayload(small_spec)
    assert p1._codec_for("cpu") is p2._codec_for("cpu")
    assert prepared_codec_for_spec(small_spec, "cpu") is p1._codec_for("cpu")

    other_spec = FibQuantSpec(
        codebook=small_spec.codebook.clone(),
        rotation=small_spec.rotation.clone(),
        d=small_spec.d,
        k=small_spec.k,
        n_levels=small_spec.n_levels,
    )
    p3 = KVPayload(other_spec)
    assert p3._codec_for("cpu") is not p1._codec_for("cpu")


def test_kvpayload_construction_is_still_spec_only(small_spec):
    """Public construction contract KVPayload(spec) must be preserved
    (cache.py, owned by another agent, constructs KVPayload this way)."""
    pl = KVPayload(small_spec)
    assert pl.spec is small_spec
    assert pl.seq_length == 0
