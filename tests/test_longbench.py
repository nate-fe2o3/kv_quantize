"""Unit tests for the LongBench v2 eval script (CPU-only, no model/GPU).

Covers the official-protocol pieces that live in scripts/eval_longbench.py:
prompt rendering, middle truncation, answer extraction, and the summary
aggregation. Generation itself runs on Databricks CUDA.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.eval_longbench import (  # noqa: E402
    build_input_ids,
    extract_answer,
    middle_truncate_ids,
    normalize_item,
    render_prompt_text,
    summarize,
)

ITEM = {
    "_id": "abc123",
    "domain": "Single-Document QA",
    "sub_domain": "Financial",
    "difficulty": "easy",
    "length": "short",
    "question": "What is the correct answer to this question: q?",
    "choices": ["ch A", "ch B", "ch C", "ch D"],
    "answer": "D",
    "context": "somerandomdocument",
}


class FakeTokenizer:
    """Identity byte-level tokenizer: 1 char == 1 token, no template overhead.

    Makes build_input_ids fully deterministic in tests: the encoded prompt is
    exactly the rendered text, and decode(encode(x)) == x.
    """

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids)

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True, enable_thinking=False):
        assert enable_thinking is False  # non-thinking mode is the point
        return {"input_ids": [ord(c) for c in messages[0]["content"]]}


def test_render_prompt_verbatim_order():
    text = render_prompt_text("DOCBODY", "Which one?", ["a1", "b2", "c3", "d4"])
    # Official 0-shot shape: intro, <text> doc, question, four choices, formatter.
    assert text.startswith("Please read the following text and answer the question below.")
    assert "<text>\nDOCBODY\n</text>" in text
    assert text.index("Which one?") > text.index("DOCBODY")
    for i, (letter, choice) in enumerate(zip("ABCD", ["a1", "b2", "c3", "d4"])):
        assert f"({letter}) {choice}" in text
    assert text.endswith('(insert answer here)".')


def test_extract_answer_official_patterns():
    assert extract_answer('The correct answer is (C).') == "C"
    assert extract_answer("The correct answer is D") == "D"
    assert extract_answer("I think B.\nThe correct answer is (A)") == "A"
    assert extract_answer("The correct answer is *(A)*") == "A"  # star-stripped
    assert extract_answer("The correct answer is (E)") is None
    assert extract_answer("The correct answer is (a)") is None
    assert extract_answer("Probably C, not sure") is None
    assert extract_answer("") is None


def test_middle_truncate_ids():
    ids = list(range(100))
    assert middle_truncate_ids(ids, 1000) == ids
    out = middle_truncate_ids(ids, 40)
    assert len(out) == 40
    assert out == list(range(20)) + list(range(80, 100))  # head + tail halves
    assert middle_truncate_ids(ids, 1) == [99]  # degenerate budget: tail wins
    assert len(middle_truncate_ids(ids, 0)) == 0


def test_normalize_item_both_shapes():
    flat = normalize_item(ITEM)  # ITEM carries a bare `choices` list
    assert flat["choices"] == ["ch A", "ch B", "ch C", "ch D"]
    assert flat["answer"] == "D"
    assert flat["_id"] == "abc123"
    # Official THUDM field names win when all four are present.
    official = {k: v for k, v in ITEM.items() if k != "choices"}
    official.update({"choice_A": "ch A", "choice_B": "ch B",
                     "choice_C": "ch C", "choice_D": "ch D",
                     "choices": ["ignored"]})
    assert normalize_item(official)["choices"] == ["ch A", "ch B", "ch C", "ch D"]
    # Mirror variant: bare choices list, no choice_* keys (== ITEM, already
    # covered by `flat`; assert the else branch again for clarity).
    assert normalize_item(ITEM)["choices"] == ["ch A", "ch B", "ch C", "ch D"]


def test_build_input_ids_no_truncation():
    tok = FakeTokenizer()
    ids, n_doc, truncated = build_input_ids(tok, ITEM, max_context=100_000)
    assert not truncated
    assert n_doc == len(ITEM["context"])
    assert len(ids) == len(render_prompt_text(ITEM["context"], ITEM["question"], ITEM["choices"]))


def test_build_input_ids_middle_truncates_doc_only():
    tok = FakeTokenizer()
    big = ITEM | {"context": "X" * 10_000}
    max_context = 500
    over = len(render_prompt_text("", ITEM["question"], ITEM["choices"]))
    ids, n_doc, truncated = build_input_ids(tok, big, max_context)
    assert truncated
    assert n_doc == 10_000
    assert len(ids) <= max_context
    text = tok.decode(ids)
    # Doc head and tail halves survive; question/choices/formatter too.
    assert text.count("X") == len(ids) - over  # every doc token is an X
    assert "X" * 50 in text  # tail half non-empty
    assert "q?" in text
    assert "Format your response as follows" in text
    assert len(ids) == max_context - 8  # budget = max_context - overhead - slack


def test_summarize_official_buckets():
    rows = [
        {"pred": "A", "answer": "A", "difficulty": "easy", "length": "short",
         "n_doc_tokens": 100, "truncated": False},
        {"pred": "B", "answer": "A", "difficulty": "easy", "length": "medium",
         "n_doc_tokens": 200, "truncated": True},
        {"pred": None, "answer": "C", "difficulty": "hard", "length": "long",
         "n_doc_tokens": 300, "truncated": True},
        {"pred": "D", "answer": "D", "difficulty": "hard", "length": "long",
         "n_doc_tokens": 400, "truncated": False},
    ]
    s = summarize(rows)
    assert s["n"] == 4
    assert s["overall"] == 50.0  # 2/4 correct
    assert s["easy"] == 50.0
    assert s["hard"] == 50.0
    assert s["short"] == 100.0
    assert s["medium"] == 0.0
    assert s["long"] == 50.0
    assert s["n_parse_fail"] == 1  # None extract counts as wrong, not dropped
    assert s["mean_doc_tokens"] == 250.0
    assert s["n_truncated"] == 2
    assert summarize([])["overall"] == 0.0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
