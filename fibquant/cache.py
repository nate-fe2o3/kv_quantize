"""FibQuant integration with transformers DynamicCache.

`FibQuantLayer` replaces the per-layer KV storage for full-attention layers:
keys/values are stored as a compressed KV payload (packed block indices +
fp16 norms — see payload.py) and decoded on read, so the attention interface
is unchanged while persistent memory drops by ~8x at b=2 (k=4, N=256), ~5.2x
at b=3 (k=4, N=4096, 12-bit indices pair-packed two per 3 bytes), and ~3.9x
at b=4 (k=4, N=65536). The index container dtype follows the codebook size
(see quantize.index_dtype / pack_indices).

FibQuantLayer is pure DynamicLayer-protocol translation: every storage-format
decision (dims, dtype policy, packing, byte accounting) lives in KVPayload,
and every operating-point decision lives in FibQuantSpec (see spec.py).

Linear-attention (Gated DeltaNet) layers keep their recurrent-state machinery,
which is inherited from `DynamicCache` untouched.

Usage:
    from fibquant import FibQuantSpec, FibQuantRuntime
    spec = FibQuantSpec.from_path(spec_path)   # or from_bits(d=256, k=4, bits=2)
    FibQuantRuntime(spec).install()            # forward() + generate() + lm-eval now compress
    # per-instance install (does not touch other models):
    FibQuantRuntime(spec).install(model=model)
"""

from __future__ import annotations

import torch
from transformers.cache_utils import DynamicCache, DynamicLayer

from .payload import KVPayload
from .runtime import FibQuantRuntime
from .spec import FibQuantSpec

__all__ = ["FibQuantCache", "FibQuantLayer", "FibQuantSpec", "FibQuantRuntime", "enable_fibquant"]


class FibQuantLayer(DynamicLayer):
    """Per-layer compressed KV cache: a KVPayload behind the DynamicLayer protocol."""

    def __init__(self, spec: FibQuantSpec):
        super().__init__()
        self.spec = spec
        self.payload = KVPayload(spec)

    # -- storage -----------------------------------------------------------

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        self.dtype, self.device = key_states.dtype, key_states.device
        # Empty placeholders so base DynamicCache bookkeeping that peeks at
        # .keys/.values does not crash; all real data lives in the payload.
        self.keys = torch.tensor([], dtype=self.dtype, device=self.device)
        self.values = torch.tensor([], dtype=self.dtype, device=self.device)
        self.is_initialized = True

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, *args, **kwargs):
        """Encode the incoming KV and return the decoded full-cache tensors."""
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        self.payload.update(key_states, value_states)
        return self.payload.decode_all(dtype=self.dtype)

    def decode_all(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.payload.decode_all(dtype=self.dtype)

    # -- protocol surface --------------------------------------------------

    def get_seq_length(self) -> int:
        if not self.is_initialized:
            return 0
        return self.payload.seq_length

    def get_max_length(self) -> int:
        return -1

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        if self.get_seq_length() == 0:
            return
        self.payload.reorder_cache(beam_idx)

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        if self.get_seq_length() == 0:
            return
        self.payload.batch_select_indices(indices)

    def batch_repeat_interleave(self, repeats: int) -> None:
        if self.get_seq_length() == 0:
            return
        self.payload.batch_repeat_interleave(repeats)

    def crop(self, max_length: int) -> None:
        self.payload.crop(max_length)

    def reset(self) -> None:
        self.payload.reset()
        if self.is_initialized:
            self.keys.zero_()
            self.values.zero_()

    # -- memory accounting -------------------------------------------------

    def stored_bytes(self) -> int:
        return self.payload.stored_bytes()

    def fp16_bytes(self) -> int:
        return self.payload.fp16_bytes()


class FibQuantCache(DynamicCache):
    """DynamicCache whose full-attention layers store FibQuant payloads."""

    def __init__(self, config=None, spec: FibQuantSpec | None = None, **kwargs):
        super().__init__(config=config, **kwargs)
        self.spec = spec
        if config is not None and spec is not None:
            text_config = config.get_text_config(decoder=True)
            layer_types = text_config.layer_types
            for layer_idx, layer_type in enumerate(layer_types):
                if layer_type == "full_attention":
                    self.layers[layer_idx] = FibQuantLayer(spec)

    def stored_bytes(self) -> int:
        return sum(
            layer.stored_bytes() if isinstance(layer, FibQuantLayer) else 0 for layer in self.layers
        )

    def fp16_bytes(self) -> int:
        return sum(
            layer.fp16_bytes() if isinstance(layer, FibQuantLayer) else 0 for layer in self.layers
        )


def enable_fibquant(
    model: torch.nn.Module | None = None,
    spec: FibQuantSpec | None = None,
) -> FibQuantRuntime:
    """Backwards-compatible install wrapper; prefer FibQuantRuntime.

    Preserves the old (model, spec) call shape. Unlike the original one-shot
    patch, the runtime install is idempotent per operating point, re-patches
    (with a warning) when a different spec is installed, covers generate()
    via the cache-factory patch, and offers uninstall()/active_spec.
    """
    if spec is None:
        raise ValueError("enable_fibquant requires a spec (pass spec=...)")
    return FibQuantRuntime(spec).install(model=model)
