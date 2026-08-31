"""Prepared codec: per-device cached encode/decode state for one FibQuantSpec.

The original module-level `encode()`/`decode()` (still in quantize.py) take
bare codebook/rotation tensors and re-derive everything the query needs on
every call: `.to(device)` on the codebook and rotation, and (inside
scoring.nearest) the augmented codebook `[c, ||c||^2]` used by the argmax
identity. Both are invariant to the query -- they only depend on the (spec,
device) pair -- but KVPayload.update()/decode_all() call that path once per
generation step, so device copies and the augmented-codebook concat were
being redone on every token.

`PreparedCodec` is the deep primitive that removes that waste: an
`nn.Module` holding the codebook, rotation (and its transpose, for decode),
and the precomputed augmented codebook as buffers, so a single `.to(device)`
moves all four together and the augmented codebook is derived exactly once
per physical copy, not once per call. `prepared_codec_for` is the small
per-device cache on top of it: a long-lived base instance plus a dict of
device -> already-moved copies, built lazily the first time a given device is
seen and reused after that.

`prepared_codec_for_spec` is the entry point KVPayload actually calls: it
keys that (base, per-device cache) pair off the FibQuantSpec instance itself
(see its docstring), so every KVPayload sharing one spec -- the normal case,
one spec and N full-attention layers per FibQuantCache -- shares one base and
one per-device cache instead of each cloning its own.

Offline codebook construction (codebook.py) has fundamentally different
requirements -- the codebook changes every Lloyd-Max iteration, so there is
nothing to prepare or cache -- and keeps calling `scoring.nearest` directly on
bare tensors. `quantize.encode`/`quantize.decode` remain as thin compatibility
adapters over a throwaway `PreparedCodec`, for callers (and the existing
test suite) that still want the original bare-tensor call shape; they do not
get the caching benefit, since there is nothing to cache across a single call.
"""

from __future__ import annotations

import copy

import torch
from torch import nn

from .scoring import DEFAULT_SCORE_BYTES, augment_codebook, nearest


class PreparedCodec(nn.Module):
    """Device-resident encode/decode state for one (codebook, rotation) pair.

    Buffers:
      - codebook (N, k): the codeword table.
      - rotation (d, d): the shared block-decomposition rotation.
      - rotation_t (d, d): rotation's transpose, precomputed once for decode
        (`x = y @ rotation_t`) instead of transposing on every call.
      - codebook_aug (N, k+1): `scoring.augment_codebook(codebook)`, the
        identity's `[c, ||c||^2]` augmentation, precomputed once so
        `scoring.nearest` never has to rebuild it per encode call.

    All four buffers move together under nn.Module's `.to(device)`/`.cpu()`/
    `.cuda()`, so a PreparedCodec is the unit of "things that must live on the
    same device to encode or decode".
    """

    codebook: torch.Tensor
    rotation: torch.Tensor
    rotation_t: torch.Tensor
    codebook_aug: torch.Tensor

    def __init__(self, codebook: torch.Tensor, rotation: torch.Tensor, k: int, n_levels: int):
        super().__init__()
        self.k = k
        self.n_levels = n_levels
        self.register_buffer("codebook", codebook.detach().clone())
        self.register_buffer("rotation", rotation.detach().clone())
        self.register_buffer("rotation_t", rotation.detach().clone().t().contiguous())
        self.register_buffer("codebook_aug", augment_codebook(self.codebook))

    @classmethod
    def from_spec(cls, spec) -> "PreparedCodec":
        """Build from a FibQuantSpec (typed loosely to avoid a spec.py<->codec.py cycle)."""
        return cls(spec.codebook, spec.rotation, spec.k, spec.n_levels)

    @torch.no_grad()
    def encode(
        self, x: torch.Tensor, max_score_bytes: int = DEFAULT_SCORE_BYTES
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize x -> (indices, norms); same contract as quantize.encode.

        Callers are responsible for `x` already living on this codec's
        device (see `prepared_codec_for`) -- this method does not itself
        move anything, since doing so on every call is exactly the cost this
        module exists to remove.
        """
        from .quantize import index_dtype  # local import: quantize.py imports codec.py, not vice versa

        norms = x.norm(dim=-1)
        safe = norms.clamp(min=1e-6)
        y = x.to(torch.float32) / safe.unsqueeze(-1)
        y = y @ self.rotation  # (B, H, S, d)
        y = y.view(*y.shape[:-1], -1, self.k)  # (B, H, S, d/k, k)

        dtype = index_dtype(self.n_levels)
        blocks = y.shape[-2]
        rows = y.numel() // self.k  # = B * H * S * blocks
        idx, _ = nearest(
            y.reshape(rows, self.k),
            self.codebook,
            max_score_bytes=max_score_bytes,
            codebook_aug=self.codebook_aug,
        )
        indices = idx.view(*y.shape[:-2], blocks).to(dtype)
        norms_f16 = safe.to(torch.float16)
        return indices, norms_f16

    @torch.no_grad()
    def decode(
        self,
        indices: torch.Tensor,
        norms: torch.Tensor,
        dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        """Dequantize (indices, norms) -> (..., d); same contract as quantize.decode."""
        if indices.numel() == 0:
            # Empty cache (e.g. after crop(0)): reshape of 0 elements is
            # ambiguous, so the logical output dim is derived explicitly as
            # blocks * k (indices' last dim is the block count), not
            # `codebook.shape[1]` (== k alone -- the original bug).
            d = indices.shape[-1] * self.k
            return torch.empty(*indices.shape[:-1], d, dtype=dtype, device=indices.device)
        y = self.codebook[indices.long()]  # (B, H, S, d/k, k)
        y = y.reshape(*y.shape[:-2], -1)  # (B, H, S, d)
        x = y @ self.rotation_t
        x = x * norms.to(x.dtype).unsqueeze(-1)
        return x.to(dtype)


def prepared_codec_for(
    base: PreparedCodec,
    device: torch.device | str,
    cache: dict[torch.device, PreparedCodec],
) -> PreparedCodec:
    """Fetch-or-build the (base, device) prepared codec, memoized in `cache`.

    `base` is a long-lived instance (typically built once on the spec's
    native/CPU device); this is what gives "move tensors once per device":
    the first call for a new device deep-copies `base` (buffers only --
    `codebook_aug` copies by value, no recompute of the augmented identity)
    and moves the copy, every later call for the same device reuses the
    cached copy instead of repeating the transfer.
    """
    device = torch.device(device)
    codec = cache.get(device)
    if codec is None:
        codec = base if base.codebook.device == device else copy.deepcopy(base).to(device)
        cache[device] = codec
    return codec


# Attribute names under which prepared_codec_for_spec stashes its base
# instance and per-device cache directly on a FibQuantSpec object (see that
# function's docstring for why identity, not id()/weakref bookkeeping, is
# the right lifetime anchor here).
_BASE_ATTR = "_fibquant_prepared_codec_base"
_CACHE_ATTR = "_fibquant_prepared_codec_cache"


def prepared_codec_for_spec(spec, device: torch.device | str) -> PreparedCodec:
    """The (spec, device) PreparedCodec, shared by every caller holding this
    exact spec instance -- not one clone per caller.

    A FibQuantCache builds one FibQuantSpec and hands that same object to
    every per-layer FibQuantLayer/KVPayload it constructs (one spec, N
    full-attention layers). Without sharing, each KVPayload calling
    `PreparedCodec.from_spec(spec)` independently would clone the codebook,
    rotation, rotation transpose, and augmented codebook once per layer --
    at N=65536 that duplicates real memory N times over and defeats the
    point of one spec shared across layers.

    The base instance and its per-device cache are attached directly to the
    spec object as plain attributes, rather than kept in a module-level
    registry keyed by `id(spec)`: FibQuantSpec is a mutable, `eq`-defined
    dataclass and so is unhashable (cannot be a WeakKeyDictionary key), and
    an id-keyed registry would need extra bookkeeping to avoid a garbage
    -collected spec's id being silently reused by an unrelated later object.
    Attaching the cache to the spec instead ties its lifetime exactly to the
    spec's own lifetime -- it is garbage collected together with the spec,
    never confused with a different (even equal-valued) spec instance, and
    every KVPayload sharing that spec object shares this same cache.
    """
    base = getattr(spec, _BASE_ATTR, None)
    if base is None:
        base = PreparedCodec.from_spec(spec)
        setattr(spec, _BASE_ATTR, base)
    cache = getattr(spec, _CACHE_ATTR, None)
    if cache is None:
        cache = {}
        setattr(spec, _CACHE_ATTR, cache)
    return prepared_codec_for(base, device, cache)
