"""Unit tests for the needle-probe marker machinery (CPU-only).

Both needle scripts (scripts/key_recall.py, scripts/multi_needle.py) now
accept multi-token markers; these tests pin the validation rules and the
whitespace-normalized continuation matching.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.multi_needle as multi_needle  # noqa: E402
from scripts.key_recall import (  # noqa: E402
    ANSWER_BUDGET_SLACK as KR_SLACK,
    MARKER as KR_MARKER,
    SENTENCE_POOLS as KR_POOLS,
    normalize_continuation as kr_norm,
    required_answer_tokens as kr_required,
    validate_marker as kr_validate,
)
from scripts.multi_needle import (  # noqa: E402
    ANSWER_BUDGET_SLACK as MN_SLACK,
    MARKERS as MN_MARKERS,
    SENTENCE_POOLS as MN_POOLS,
    normalize_continuation as mn_norm,
    required_answer_tokens as mn_required,
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


def test_validate_markers_duplicate_rejected():
    # An exact repeat is neither an "overlap" of two distinct markers nor
    # pool text -- it needs its own check, or a repeated marker (e.g. a
    # copy-paste config mistake) silently slips through as if it were fine.
    with pytest.raises(ValueError, match="duplicate"):
        mn_validate(["rabbit", "rabbit"], MN_POOLS)
    with pytest.raises(ValueError, match="duplicate"):
        mn_validate(["Rabbit", "rabbit"], MN_POOLS)  # case-insensitive


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


class _WordsTok:
    """Fake tokenizer: one token per whitespace-delimited word.

    Enough to pin the answer-budget formula; real token counts differ but
    the margin (verbose listing + slack) is what the startup check enforces.
    """

    def encode(self, text, add_special_tokens=False):
        return list(range(len(text.split())))


class _CharTok:
    """Minimal reversible tokenizer with one special chat-template token."""

    im_end = 256

    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]

    def decode(self, ids):
        return "".join("<|im_end|>" if int(token) == self.im_end else chr(int(token)) for token in ids)

    def convert_tokens_to_ids(self, token):
        assert token == "<|im_end|>"
        return self.im_end

    def apply_chat_template(self, messages, **_kwargs):
        rendered = []
        for message in messages:
            rendered.extend(self.encode(f"<{message['role']}>{message['content']}"))
            rendered.append(self.im_end)
        rendered.extend(self.encode("<assistant>"))
        return {"input_ids": rendered}


def test_multi_needle_prompt_asks_for_markers(monkeypatch):
    monkeypatch.setattr(multi_needle, "NEEDLE_MIN_POS", 50)
    monkeypatch.setattr(multi_needle, "NEEDLE_MIN_TAIL", 100)
    monkeypatch.setattr(multi_needle, "NEEDLE_SPACING", 10)

    tokenizer = _CharTok()
    rows, _ = multi_needle.build_trials(tokenizer, ["rabbit", "whale"], depth=400, trials=1, seed=0)
    prompt = tokenizer.decode(rows[0])

    assert multi_needle.QUESTION.strip() in prompt


def test_required_answer_tokens_scales_with_marker_length():
    tok = _WordsTok()
    # Verbose per-line listing cost + slack (the format the model may echo).
    assert mn_required(tok, ["rabbit"]) == len(" Special token: rabbit.".split()) + MN_SLACK
    assert mn_required(tok, ["blue whale"]) == len(" Special token: blue whale.".split()) + MN_SLACK
    two = len(" Special token: rabbit. Special token: blue whale.".split()) + MN_SLACK
    assert mn_required(tok, ["rabbit", "blue whale"]) == two
    # Multi-token markers raise the required budget: 5 phrases cost more than
    # 5 single words.
    five_words = mn_required(tok, ["a", "b", "c", "d", "e"])
    five_phrases = mn_required(tok, ["a b", "c d", "e f", "g h", "i j"])
    assert five_phrases > five_words
    # key_recall: one marker, same formula with its own slack.
    assert kr_required(tok, ["blue whale"]) == len(" Special token: blue whale.".split()) + KR_SLACK


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
