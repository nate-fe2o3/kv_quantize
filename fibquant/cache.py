"""FibQuant integration with transformers DynamicCache.

`FibQuantLayer` replaces the per-layer KV storage for full-attention layers:
keys/values are stored as (uint8/uint16 block indices, fp16 norms) and decoded
on read, so the attention interface is unchanged while persistent memory drops
by ~8x at b=2 (k=4, N=256) and ~3.9x at b=4 (k=4, N=65536). The index
container dtype follows the codebook size (see quantize.index_dtype).

Linear-attention (Gated DeltaNet) layers keep their recurrent-state machinery,
which is inherited from `DynamicCache` untouched.

Usage:
    from fibquant import enable_fibquant, load_spec
    spec = load_spec(spec_path)
    enable_fibquant(model, spec)          # forward() + lm-eval now compress
    model.generate(..., cache=FibQuantCache(config, spec))  # generate() must pass the cache
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers.cache_utils import DynamicCache, DynamicLayer

from .quantize import decode, encode

_FIBQUANT_PATCHED = False


@dataclass
class FibQuantSpec:
    """Shared codebook + rotation metadata for one (d, k, N) operating point."""

    codebook: torch.Tensor  # (N, k) fp32 on unit ball
    rotation: torch.Tensor  # (d, d) fp32 orthogonal
    d: int
    k: int
    n_levels: int

    @property
    def bits_per_coord(self) -> float:
        return (self.n_levels - 1).bit_length() / self.k

    @classmethod
    def from_checkpoint(cls, ckpt: dict) -> "FibQuantSpec":
        codebook = ckpt["codebook"]
        rotation = ckpt["rotation"]
        if codebook.shape[0] != ckpt["n_levels"]:
            raise ValueError(
                f"checkpoint n_levels={ckpt['n_levels']} != codebook rows {codebook.shape[0]}"
            )
        if codebook.shape[1] != ckpt["k"]:
            raise ValueError(
                f"checkpoint k={ckpt['k']} != codebook dim {codebook.shape[1]}"
            )
        if tuple(rotation.shape) != (ckpt["d"], ckpt["d"]):
            raise ValueError(
                f"checkpoint d={ckpt['d']} != rotation shape {tuple(rotation.shape)}"
            )
        return cls(
            codebook=codebook,
            rotation=rotation,
            d=ckpt["d"],
            k=ckpt["k"],
            n_levels=ckpt["n_levels"],
        )


class FibQuantLayer(DynamicLayer):
    """Per-layer compressed KV cache: uint8/uint16 block indices + fp16 norms."""

    def __init__(self, spec: FibQuantSpec):
        super().__init__()
        self.spec = spec
        self.num_blocks = spec.d // spec.k
        self.key_indices: torch.Tensor | None = None
        self.key_norms: torch.Tensor | None = None
        self.value_indices: torch.Tensor | None = None
        self.value_norms: torch.Tensor | None = None

    # -- storage -----------------------------------------------------------

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        self.dtype, self.device = key_states.dtype, key_states.device
        # Empty placeholders so base DynamicCache bookkeeping that peeks at
        # .keys/.values does not crash; all real data lives in *_indices/_norms.
        self.keys = torch.tensor([], dtype=self.dtype, device=self.device)
        self.values = torch.tensor([], dtype=self.dtype, device=self.device)
        self.is_initialized = True

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, *args, **kwargs):
        """Encode the incoming KV and return the decoded full-cache tensors."""
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        spec = self.spec
        k_idx, k_norm = encode(key_states, spec.codebook, spec.rotation, spec.k)
        v_idx, v_norm = encode(value_states, spec.codebook, spec.rotation, spec.k)

        if self.key_indices is None:
            self.key_indices, self.key_norms = k_idx, k_norm
            self.value_indices, self.value_norms = v_idx, v_norm
        else:
            self.key_indices = torch.cat([self.key_indices, k_idx], dim=-2)
            self.key_norms = torch.cat([self.key_norms, k_norm], dim=-1)
            self.value_indices = torch.cat([self.value_indices, v_idx], dim=-2)
            self.value_norms = torch.cat([self.value_norms, v_norm], dim=-1)

        return self.decode_all()

    def decode_all(self) -> tuple[torch.Tensor, torch.Tensor]:
        spec = self.spec
        keys = decode(self.key_indices, self.key_norms, spec.codebook, spec.rotation, dtype=self.dtype)
        values = decode(self.value_indices, self.value_norms, spec.codebook, spec.rotation, dtype=self.dtype)
        return keys, values

    # -- protocol surface --------------------------------------------------

    def get_seq_length(self) -> int:
        if not self.is_initialized or self.key_indices is None:
            return 0
        return self.key_indices.shape[-2]

    def get_max_length(self) -> int:
        return -1

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        if self.get_seq_length() == 0:
            return
        beam_idx = beam_idx.to(self.device)
        for name in ("key_indices", "key_norms", "value_indices", "value_norms"):
            setattr(self, name, getattr(self, name).index_select(0, beam_idx))

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        if self.get_seq_length() == 0:
            return
        for name in ("key_indices", "key_norms", "value_indices", "value_norms"):
            setattr(self, name, getattr(self, name)[indices, ...])

    def batch_repeat_interleave(self, repeats: int) -> None:
        if self.get_seq_length() == 0:
            return
        for name in ("key_indices", "key_norms", "value_indices", "value_norms"):
            setattr(self, name, getattr(self, name).repeat_interleave(repeats, dim=0))

    def crop(self, tokens_to_remove: int) -> None:
        if tokens_to_remove > 0:
            current_length = self.get_seq_length()
            if tokens_to_remove >= current_length:
                return
            tokens_to_remove = self.get_seq_length() - tokens_to_remove
        if tokens_to_remove == 0:
            return
        self.key_indices = self.key_indices[..., : -abs(tokens_to_remove), :]
        self.key_norms = self.key_norms[..., : -abs(tokens_to_remove)]
        self.value_indices = self.value_indices[..., : -abs(tokens_to_remove), :]
        self.value_norms = self.value_norms[..., : -abs(tokens_to_remove)]

    def reset(self) -> None:
        self.key_indices = None
        self.key_norms = None
        self.value_indices = None
        self.value_norms = None
        if self.is_initialized:
            self.keys.zero_()
            self.values.zero_()

    # -- memory accounting -------------------------------------------------

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
        return 2 * seq * self.spec.d * 2 * self.key_indices.shape[0] * self.key_indices.shape[1]


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


def enable_fibquant(model: torch.nn.Module | None, spec: FibQuantSpec) -> None:
    """Monkeypatch Qwen3_5TextModel.forward to build a FibQuantCache by default.

    Covers plain forward() calls and lm-eval (model may be None: the patch is
    class-level and applies to any Qwen3_5 text model instantiated after it).
    For model.generate(), pass cache=FibQuantCache(config, spec) explicitly
    (generate builds its own cache and skips the forward-side creation path).
    """
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

    if _FIBQUANT_PATCHED:
        return

    original_forward = Qwen3_5TextModel.forward

    def wrapped_forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        **kwargs,
    ):
        if use_cache and past_key_values is None:
            past_key_values = FibQuantCache(config=self.config, spec=spec)
        return original_forward(
            self,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

    Qwen3_5TextModel.forward = wrapped_forward
    globals()["_FIBQUANT_PATCHED"] = True
