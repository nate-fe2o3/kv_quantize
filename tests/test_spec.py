"""Operating point: validation, bits arithmetic, path convention, save/load."""

import pytest
import torch

from fibquant.codebook import build_directions, build_radii, build_rotation
from fibquant.spec import FibQuantSpec, load_spec, spec_path


def _raw(d=8, k=2, n_levels=16):
    """Codebook of the right (n_levels, k) shape for validation/save-load tests.

    None of test_spec.py's assertions depend on quantization quality (only
    shape, bits arithmetic, and checkpoint round-tripping), so this uses the
    untrained radii*directions init directly instead of build_codebook: no
    Lloyd-Max, no (samples, n_levels) training score matrix -- keeps the
    n_levels=4096/65536 cases here cheap.
    """
    radii = build_radii(d, k, n_levels)
    directions = build_directions(k, n_levels)
    return radii.unsqueeze(-1) * directions


def test_validation_rejects_bad_operating_points():
    cb = _raw()
    rot = build_rotation(8, 0)
    with pytest.raises(ValueError, match="divisible"):
        FibQuantSpec(codebook=cb, rotation=rot, d=9, k=2, n_levels=16)
    with pytest.raises(ValueError, match="k must be"):
        FibQuantSpec(codebook=cb, rotation=rot, d=8, k=0, n_levels=16)
    with pytest.raises(ValueError, match=">= 2"):
        FibQuantSpec(codebook=cb[:1], rotation=rot, d=8, k=2, n_levels=1)
    with pytest.raises(ValueError, match="exceeds uint16"):
        FibQuantSpec(codebook=cb, rotation=rot, d=8, k=2, n_levels=1 << 17)
    with pytest.raises(ValueError, match="codebook shape"):
        FibQuantSpec(codebook=cb, rotation=rot, d=8, k=2, n_levels=32)
    with pytest.raises(ValueError, match="rotation shape"):
        FibQuantSpec(codebook=cb, rotation=build_rotation(6, 0), d=8, k=2, n_levels=16)


def test_validation_rejects_odd_blocks_at_12bit():
    cb = _raw(n_levels=4096)
    with pytest.raises(ValueError, match="even number of blocks"):
        FibQuantSpec(codebook=cb, rotation=build_rotation(6, 0), d=6, k=2, n_levels=4096)


def test_bits_per_coord_and_path_convention():
    rot = build_rotation(8, 0)
    for n_levels, bits in [(16, 2.0), (4096, 6.0), (65536, 8.0)]:
        assert FibQuantSpec(
            codebook=_raw(n_levels=n_levels), rotation=rot, d=8, k=2, n_levels=n_levels
        ).bits_per_coord == bits
    assert spec_path(256, 4, 256).name == "fibquant_d256_k4_N256.pt"


def test_from_bits_derives_n_levels_and_rejects_bad_bits(tmp_path):
    out = tmp_path / "fibquant_d8_k2_N16.pt"
    FibQuantSpec(
        codebook=_raw(), rotation=build_rotation(8, 0), d=8, k=2, n_levels=16
    ).save(out, seed=0, mse=0.25)
    spec = FibQuantSpec.from_bits(d=8, k=2, bits=2, path=out)
    assert spec.n_levels == 16
    assert spec.bits_per_coord == 2.0
    assert spec.seed == 0 and spec.mse == 0.25
    with pytest.raises(ValueError, match="implies n_levels=262144"):
        FibQuantSpec.from_bits(d=8, k=2, bits=9)  # 2^18 > 2^16
    with pytest.raises(ValueError, match="bits must be"):
        FibQuantSpec.from_bits(d=8, k=2, bits=0)


def test_from_bits_default_path_missing_is_actionable(tmp_path):
    # No path= and the repo-relative default checkpoint is absent (the
    # Databricks volume case): the error must say what to do, not just
    # "No such file or directory" from torch.load.
    with pytest.raises(FileNotFoundError, match="pass path= explicitly"):
        FibQuantSpec.from_bits(d=8, k=2, bits=2)  # models/fibquant/fibquant_d8_k2_N16.pt


def test_save_load_roundtrip_preserves_checkpoint_dict(tmp_path):
    spec = FibQuantSpec(
        codebook=_raw(n_levels=4096), rotation=build_rotation(8, 0), d=8, k=2, n_levels=4096
    )
    out = tmp_path / "s.pt"
    spec.save(out, seed=7, mse=0.42)
    ckpt = load_spec(out)
    assert ckpt["d"] == 8 and ckpt["k"] == 2 and ckpt["n_levels"] == 4096
    assert ckpt["seed"] == 7 and ckpt["mse"] == 0.42
    assert torch.equal(ckpt["codebook"], spec.codebook)
    spec2 = FibQuantSpec.from_checkpoint(ckpt)
    assert spec2.n_levels == 4096 and spec2.seed == 7 and spec2.mse == 0.42
