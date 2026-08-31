"""Unit tests for fibquant/probes.py, the shared probe-support module (CPU-only).

Covers the pieces every probe script (key_recall, multi_needle, logit_kl,
eval_longbench) now shares: filler generation, continuation normalization,
marker validation, the answer-token budget formula, and (name, spec) matrix
assembly. Script-level tests (tests/test_needles.py, tests/test_longbench.py)
still cover each script's own wrappers/call sites.
"""

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fibquant.probes import (  # noqa: E402
    SENTENCE_POOLS,
    SENTENCE_TEMPLATES,
    build_spec_matrix,
    normalize_continuation,
    required_answer_tokens,
    unique_filler_sentence,
    validate_markers,
)


# --- normalize_continuation ------------------------------------------------


def test_normalize_continuation_lowercases_and_collapses_whitespace():
    assert normalize_continuation("  Blue   Whale.\n") == "blue whale."
    assert normalize_continuation("") == ""
    assert normalize_continuation("A\tB\nC") == "a b c"


# --- validate_markers -------------------------------------------------------


def test_validate_markers_accepts_a_sane_set():
    validate_markers(["blue whale", "grand piano"])  # no error


def test_validate_markers_rejects_empty():
    with pytest.raises(ValueError, match="empty"):
        validate_markers(["  "])
    with pytest.raises(ValueError, match="empty"):
        validate_markers([""])


def test_validate_markers_rejects_exact_duplicate():
    with pytest.raises(ValueError, match="duplicate"):
        validate_markers(["rabbit", "rabbit"])
    with pytest.raises(ValueError, match="duplicate"):
        validate_markers(["Rabbit", "rabbit"])  # case-insensitive


def test_validate_markers_rejects_substring_overlap():
    with pytest.raises(ValueError, match="overlaps"):
        validate_markers(["rabbit", "rabbit hole"])
    with pytest.raises(ValueError, match="overlaps"):
        validate_markers(["whale", "humpback whale"])


def test_validate_markers_rejects_filler_pool_text():
    with pytest.raises(ValueError, match="filler pool"):
        validate_markers(["lantern"])  # a SENTENCE_POOLS['noun'] entry


def test_validate_markers_duplicate_check_precedes_overlap_check():
    # A duplicate pair must be reported as "duplicate", not misreported as an
    # "overlap" (every string is a substring of itself, so an overlap-only
    # implementation would still raise -- just with a confusing message).
    with pytest.raises(ValueError, match="duplicate"):
        validate_markers(["rabbit", "rabbit", "grand piano"])


# --- required_answer_tokens -------------------------------------------------


class _WordsTok:
    """Fake tokenizer: one token per whitespace-delimited word."""

    def encode(self, text, add_special_tokens=False):
        return list(range(len(text.split())))


def test_required_answer_tokens_formula():
    tok = _WordsTok()
    assert required_answer_tokens(tok, ["rabbit"], slack=3) == len(" Special token: rabbit.".split()) + 3
    two = len(" Special token: rabbit. Special token: blue whale.".split()) + 5
    assert required_answer_tokens(tok, ["rabbit", "blue whale"], slack=5) == two


def test_required_answer_tokens_scales_with_marker_count_and_length():
    tok = _WordsTok()
    one = required_answer_tokens(tok, ["a"], slack=0)
    five = required_answer_tokens(tok, ["a", "b", "c", "d", "e"], slack=0)
    assert five > one
    five_phrases = required_answer_tokens(tok, ["a b", "c d", "e f", "g h", "i j"], slack=0)
    assert five_phrases > five


def test_required_answer_tokens_custom_frame():
    tok = _WordsTok()
    default = required_answer_tokens(tok, ["rabbit"], slack=0)
    custom = required_answer_tokens(tok, ["rabbit"], slack=0, frame="Needle: {word}!!")
    assert custom != default  # a longer/shorter frame changes the budget


# --- unique_filler_sentence --------------------------------------------------


class _CountingTok:
    """Fake tokenizer good enough for filler generation: whitespace tokens."""

    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, ids):
        return " ".join(ids)


def test_unique_filler_sentence_avoids_repeats():
    tok = _CountingTok()
    rng = random.Random(0)
    used: set[str] = set()
    texts = set()
    for _ in range(50):
        text, ids = unique_filler_sentence(tok, rng, used)
        assert text not in texts  # unique_filler_sentence enforces this via `used`
        texts.add(text)
        assert ids  # non-empty token ids
    assert len(used) == 50


def test_unique_filler_sentence_rejects_avoid_terms():
    tok = _CountingTok()
    rng = random.Random(0)
    used: set[str] = set()
    # Every noun-slot word is individually forbidden; only templates/pools
    # that avoid all `noun` fills entirely could satisfy this, so as long as
    # a sentence is produced, it must not contain any forbidden noun.
    forbidden = tuple(w.lower() for w in SENTENCE_POOLS["noun"])
    for _ in range(20):
        text, _ids = unique_filler_sentence(tok, rng, used, avoid=forbidden)
        assert not any(f in text.lower() for f in forbidden)


def test_unique_filler_sentence_deterministic_given_seed():
    tok = _CountingTok()
    text1, _ = unique_filler_sentence(tok, random.Random(42), set())
    text2, _ = unique_filler_sentence(tok, random.Random(42), set())
    assert text1 == text2


def test_unique_filler_sentence_exhaustion_raises():
    tok = _CountingTok()
    rng = random.Random(0)
    # A single-slotless template/pool leaves only one possible sentence, so a
    # non-empty `used` set exhausts the generator immediately.
    templates = ["A fixed sentence with no slots."]
    used = {"A fixed sentence with no slots."}
    with pytest.raises(ValueError, match="could not generate"):
        unique_filler_sentence(tok, rng, used, templates=templates, pools=SENTENCE_POOLS)


def test_sentence_templates_and_pools_are_self_consistent():
    # Every {slot} referenced by a template must exist in SENTENCE_POOLS, or
    # unique_filler_sentence raises a KeyError deep in random generation
    # instead of failing fast/obviously.
    import re

    slot_re = re.compile(r"\{(\w+)\}")
    for template in SENTENCE_TEMPLATES:
        for slot in slot_re.findall(template):
            assert slot in SENTENCE_POOLS, f"template {template!r} references unknown slot {slot!r}"


# --- build_spec_matrix -------------------------------------------------------


def test_build_spec_matrix_baseline_only():
    specs = build_spec_matrix(True, [], [])
    assert specs == [("baseline", None)]


def test_build_spec_matrix_empty_when_nothing_requested():
    assert build_spec_matrix(False, [], []) == []


def test_build_spec_matrix_spec_paths(tmp_path, small_spec):
    ckpt = tmp_path / "fibquant_d8_k2_N16.pt"
    small_spec.save(ckpt)
    specs = build_spec_matrix(False, [], [ckpt])
    assert [name for name, _ in specs] == [f"fq-N{small_spec.n_levels}"]
    name, spec = specs[0]
    assert spec is not None
    assert spec.d == small_spec.d and spec.k == small_spec.k and spec.n_levels == small_spec.n_levels


def test_build_spec_matrix_bits_via_from_bits(monkeypatch, small_spec):
    # FibQuantSpec.from_bits() resolves a repo-relative default checkpoint
    # path with no path= override, so BITS assembly is exercised here by
    # stubbing from_bits itself rather than writing into the real repo tree.
    import fibquant.probes as probes_mod

    calls = []

    def fake_from_bits(d, k, bits):
        calls.append((d, k, bits))
        return small_spec

    monkeypatch.setattr(probes_mod.FibQuantSpec, "from_bits", staticmethod(fake_from_bits))
    specs = build_spec_matrix(False, [2, 4], [])
    assert [name for name, _ in specs] == ["fq-b2", "fq-b4"]
    assert calls == [(256, 4, 2), (256, 4, 4)]  # default d, k from build_spec_matrix


def test_build_spec_matrix_row_order_baseline_then_bits_then_paths(tmp_path, small_spec, monkeypatch):
    import fibquant.probes as probes_mod

    monkeypatch.setattr(probes_mod.FibQuantSpec, "from_bits", staticmethod(lambda d, k, bits: small_spec))
    ckpt = tmp_path / "spec.pt"
    small_spec.save(ckpt)
    specs = build_spec_matrix(True, [2], [ckpt])
    assert [name for name, _ in specs] == ["baseline", "fq-b2", f"fq-N{small_spec.n_levels}"]
    assert specs[0][1] is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
