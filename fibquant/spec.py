"""The FibQuant operating point: one validated (d, k, N, bits/coord) configuration.

An operating point names everything needed to run FibQuant at one compression
width:

  - d:    head dim, k:    coordinates per codebook block,
  - N:    codebook size,  bits = log2(N) / k bits per coordinate.

This module is the single home for the operating-point interface: the bits/N
arithmetic, the spec-checkpoint filename convention
(models/fibquant/fibquant_d{d}_k{k}_N{N}.pt), checkpoint validation, and
save/load. Codebook math lives in codebook.py; the runtime cache protocol in
cache.py; nothing else re-derives `1 << (bits * k)` or builds spec paths.

Validation happens at construction: an invalid operating point cannot be
instantiated, so scripts and the cache can trust the type. Specifics:

  - d % k == 0 (head vectors split into whole blocks)
  - n_levels <= 2^16 (index_dtype container limit)
  - when indices are pair-packed (12-bit, N = 4096), blocks must be even
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

_SPEC_NAME = "fibquant_d{d}_k{k}_N{n_levels}.pt"


def _bits(n_levels: int) -> int:
    """Container bits per block index: bit_length(n_levels - 1)."""
    return int(n_levels - 1).bit_length()


@dataclass
class FibQuantSpec:
    """Shared codebook + rotation metadata for one (d, k, N) operating point.

    All fields except seed/mse are validated at construction; seed/mse are
    checkpoint provenance and may be absent (None) for runtime-only specs.
    """

    codebook: torch.Tensor  # (N, k) fp32
    rotation: torch.Tensor  # (d, d) fp32 orthogonal
    d: int
    k: int
    n_levels: int
    seed: int | None = None
    mse: float | None = None

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError(f"k must be >= 1, got {self.k}")
        if self.d % self.k:
            raise ValueError(f"head dim d={self.d} not divisible by block k={self.k}")
        if self.n_levels < 2:
            raise ValueError(f"n_levels must be >= 2, got {self.n_levels}")
        if self.n_levels > 2**16:
            raise ValueError(
                f"n_levels={self.n_levels} exceeds uint16; not supported"
            )
        if self._packed and (self.d // self.k) % 2:
            raise ValueError(
                f"12-bit pair-packing requires an even number of blocks, got "
                f"d/k={self.d // self.k}"
            )
        if tuple(self.codebook.shape) != (self.n_levels, self.k):
            raise ValueError(
                f"codebook shape {tuple(self.codebook.shape)} != "
                f"(n_levels={self.n_levels}, k={self.k})"
            )
        if tuple(self.rotation.shape) != (self.d, self.d):
            raise ValueError(
                f"rotation shape {tuple(self.rotation.shape)} != ({self.d}, {self.d})"
            )

    @property
    def bits_per_coord(self) -> float:
        return _bits(self.n_levels) / self.k

    @property
    def blocks(self) -> int:
        return self.d // self.k

    @property
    def _packed(self) -> bool:
        """True when indices use the 12-bit pair-packed storage form."""
        return _bits(self.n_levels) == 12

    @classmethod
    def from_checkpoint(cls, ckpt: dict) -> "FibQuantSpec":
        """Build from a spec checkpoint dict (see save)."""
        return cls(
            codebook=ckpt["codebook"],
            rotation=ckpt["rotation"],
            d=ckpt["d"],
            k=ckpt["k"],
            n_levels=ckpt["n_levels"],
            seed=ckpt.get("seed"),
            mse=ckpt.get("mse"),
        )

    @classmethod
    def from_path(cls, path: str | Path) -> "FibQuantSpec":
        return cls.from_checkpoint(load_spec(path))

    @classmethod
    def from_bits(cls, d: int, k: int, bits: int, path: str | Path | None = None) -> "FibQuantSpec":
        """Load the operating point named by (d, k, bits); N = 1 << (bits * k)."""
        if bits < 1:
            raise ValueError(f"bits must be >= 1, got {bits}")
        n_levels = 1 << (bits * k)
        if n_levels > 2**16:
            raise ValueError(
                f"bits={bits}, k={k} implies n_levels={n_levels} > 2^16; not supported"
            )
        return cls.from_path(path if path is not None else spec_path(d, k, n_levels))

    def save(self, path: str | Path, *, seed: int | None = None, mse: float | None = None) -> None:
        """Persist codebook + rotation to disk as a single checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "codebook": self.codebook,
                "rotation": self.rotation,
                "d": self.d,
                "k": self.k,
                "n_levels": self.n_levels,
                "seed": seed if seed is not None else self.seed,
                "mse": mse if mse is not None else self.mse,
            },
            path,
        )


def spec_path(d: int, k: int, n_levels: int) -> Path:
    """models/fibquant/fibquant_d{d}_k{k}_N{n_levels}.pt"""
    return Path(__file__).resolve().parent.parent / "models" / "fibquant" / _SPEC_NAME.format(
        d=d, k=k, n_levels=n_levels
    )


def load_spec(path: str | Path) -> dict:
    """Load a spec checkpoint dict (see FibQuantSpec.save)."""
    return torch.load(path, map_location="cpu", weights_only=False)


# Backwards-compatible alias for code that imported default_spec_path.
default_spec_path = spec_path
