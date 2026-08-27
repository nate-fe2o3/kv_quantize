"""Compressed KV payload: the storage format of one FibQuant layer.

Owns everything about how compressed keys/values are stored for one layer:
the codec steps (encode / pack / decode), the container dtype policy, the
storage dim conventions, byte accounting, and the in-place storage verbs
(append, crop, reorder, select, repeat, reset) that transformers
DynamicLayer protocol operations translate to.

Storage layout (both keys and values, identically):

  - indices: (batch, heads, seq, blocks) or (batch, heads, seq, 1.5 * blocks)
    for the 12-bit pair-packed form (see quantize.pack_indices). Seq is the
    second-to-last dim.
  - norms:   (batch, heads, seq) fp16. Seq is the last dim.

The mixed dim convention is written once here; callers never touch it.

Invariants pinned by design (see CONTEXT.md):

  - indices are always appended in complete head-vector rows, and packing is
    within-row (adjacent k-blocks), never across tokens — so the sequence-dim
    verbs (crop / reorder / select / repeat / reset) never need to unpack or
    repack.
  - FibQuantSpec validation already rejects d not divisible by k and 12-bit
    operating points with an odd block count; pack_indices remains as defense
    in depth.
"""

from __future__ import annotations

import torch

from .quantize import decode, encode, pack_indices, unpack_indices
from .spec import FibQuantSpec


class KVPayload:
    """Compressed key/value storage for one layer, decoded on read."""

    def __init__(self, spec: FibQuantSpec):
        self.spec = spec
        self.key_indices: torch.Tensor | None = None
        self.key_norms: torch.Tensor | None = None
        self.value_indices: torch.Tensor | None = None
        self.value_norms: torch.Tensor | None = None

    # -- append / read ----------------------------------------------------

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        """Encode, pack, and append the incoming KV states."""
        spec = self.spec
        k_idx, k_norm = encode(key_states, spec.codebook, spec.rotation, spec.k)
        v_idx, v_norm = encode(value_states, spec.codebook, spec.rotation, spec.k)
        k_idx = pack_indices(k_idx, spec.n_levels)
        v_idx = pack_indices(v_idx, spec.n_levels)

        if self.key_indices is None:
            self.key_indices, self.key_norms = k_idx, k_norm
            self.value_indices, self.value_norms = v_idx, v_norm
        else:
            self.key_indices = torch.cat([self.key_indices, k_idx], dim=-2)
            self.key_norms = torch.cat([self.key_norms, k_norm], dim=-1)
            self.value_indices = torch.cat([self.value_indices, v_idx], dim=-2)
            self.value_norms = torch.cat([self.value_norms, v_norm], dim=-1)

    def decode_all(self, dtype: torch.dtype = torch.bfloat16) -> tuple[torch.Tensor, torch.Tensor]:
        """Dequantize the full cache: (keys, values) in storage order."""
        spec = self.spec
        key_ids = unpack_indices(self.key_indices, spec.n_levels)
        value_ids = unpack_indices(self.value_indices, spec.n_levels)
        keys = decode(key_ids, self.key_norms, spec.codebook, spec.rotation, dtype=dtype)
        values = decode(value_ids, self.value_norms, spec.codebook, spec.rotation, dtype=dtype)
        return keys, values

    # -- sequence-dim protocol verbs (in place) ---------------------------

    @property
    def seq_length(self) -> int:
        if self.key_indices is None:
            return 0
        return self.key_indices.shape[-2]

    def crop(self, max_length: int) -> None:
        """Match transformers DynamicLayer.crop exactly: keep first max_length
        tokens; negative max_length removes that many from the end."""
        if self.key_indices is None:
            return
        seq = self.seq_length
        if max_length < 0:
            max_length = seq - abs(max_length)
        if seq <= max_length:
            return
        self.key_indices = self.key_indices[..., :max_length, :]
        self.key_norms = self.key_norms[..., :max_length]
        self.value_indices = self.value_indices[..., :max_length, :]
        self.value_norms = self.value_norms[..., :max_length]

    def reorder_cache(self, beam_idx: torch.Tensor) -> None:
        if self.key_indices is None:
            return
        beam_idx = beam_idx.to(self.key_indices.device)
        for name in ("key_indices", "key_norms", "value_indices", "value_norms"):
            setattr(self, name, getattr(self, name).index_select(0, beam_idx))

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        if self.key_indices is None:
            return
        for name in ("key_indices", "key_norms", "value_indices", "value_norms"):
            setattr(self, name, getattr(self, name)[indices, ...])

    def batch_repeat_interleave(self, repeats: int) -> None:
        if self.key_indices is None:
            return
        for name in ("key_indices", "key_norms", "value_indices", "value_norms"):
            setattr(self, name, getattr(self, name).repeat_interleave(repeats, dim=0))

    def reset(self) -> None:
        self.key_indices = None
        self.key_norms = None
        self.value_indices = None
        self.value_norms = None

    # -- memory accounting ------------------------------------------------

    def stored_bytes(self) -> int:
        if self.key_indices is None:
            return 0
        return sum(
            t.numel() * t.element_size()
            for t in (self.key_indices, self.key_norms, self.value_indices, self.value_norms)
        )

    def fp16_bytes(self) -> int:
        """Bytes the same cache would occupy as fp16 (K + V)."""
        if self.key_indices is None:
            return 0
        seq = self.key_indices.shape[-2]
        batch, heads = self.key_indices.shape[0], self.key_indices.shape[1]
        return 2 * seq * self.spec.d * 2 * batch * heads
