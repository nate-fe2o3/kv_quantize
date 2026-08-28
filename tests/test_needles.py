"""Unit tests for the needle-probe marker machinery (CPU-only).

Both needle scripts (scripts/key_recall.py, scripts/multi_needle.py) now
accept multi-token markers; these tests pin the validation rules and the
whitespace-normalized continuation matching.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.key_recall import (  # noqa: E402
    MARKER as KR_MARKER,
    SENTENCE_POOLS as KR_POOLS,
    normalize_continuation as kr_norm,
    validate_marker as kr_validate,
)
from scripts.multi_needle import (  # noqa: E402
    MARKERS as MN_MARKERS,
    SENTENCE_POOLS as MN_POOLS,
    normalize_continuation as mn_norm,
    validate_markers as mn_validate,
)


def test_normalize_continuation_collapses_whitespace():
    assert kr_norm("  Blue   whale.\n") == "blue whale."
    assert mn_norm("Special token: GRAND  PIANO") == "special token: grand piano"
    assert kr_norm("") == ""
    assert mn_norm("a\tb\nc") == "a b c"
    # This is the case the old exact match missed: inserted spaces/newlines
    # between phrase-marker words must not count as a recall miss.
    assert "blue whale" in kr_norm("recalled it: blue \n  whale.")


def test_default_markers_validate():
    # Default set passes now even though 'whale' is two tokens bare in the
    # Qwen3.5 tokenizer (the old single-token rule rejected it).
    mn_validate(MN_MARKERS, MN_POOLS)
    kr_validate(KR_MARKER, KR_POOLS)
    # Multi-token phrases are accepted.
    mn_validate(["blue whale", "grand piano"], MN_POOLS)
    kr_validate("blue whale", KR_POOLS)


def test_validate_markers_overlap_rejected():
    with pytest.raises(ValueError, match="overlaps"):
        mn_validate(["rabbit", "rabbit hole"], MN_POOLS)
    with pytest.raises(ValueError, match="overlaps"):
        mn_validate(["whale", "humpback whale"], MN_POOLS)


def test_validate_markers_pool_text_rejected():
    with pytest.raises(ValueError, match="filler pool"):
        mn_validate(["lantern"], MN_POOLS)
    with pytest.raises(ValueError, match="filler pool"):
        kr_validate("lantern", KR_POOLS)


def test_validate_markers_empty_rejected():
    with pytest.raises(ValueError, match="empty"):
        mn_validate(["  "], MN_POOLS)
    with pytest.raises(ValueError, match="empty"):
        kr_validate("", KR_POOLS)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
