"""Unit tests for the LongBench v2 eval script (CPU-only, no model/GPU).

Covers the official-protocol pieces that live in scripts/eval_longbench.py:
prompt rendering, middle truncation, answer extraction, the summary
aggregation, and the resume-safety manifest/fingerprint machinery.
Generation itself runs on Databricks CUDA.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.eval_longbench import (  # noqa: E402
    build_input_ids,
    build_manifest,
    ensure_manifest_compatible,
    extract_answer,
    items_fingerprint,
    load_and_repair_jsonl,
    middle_truncate_ids,
    model_identity,
    normalize_item,
    read_jsonl_rows,
    render_prompt_text,
    repair_truncated_jsonl,
    spec_fingerprint,
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


# --- resume-safety manifest -------------------------------------------------


class _FakeConfig:
    """Enough of a text_config for model_identity's cheap-field probe."""

    model_type = "qwen3_5"
    hidden_size = 1024
    num_hidden_layers = 24
    vocab_size = 151936
    max_position_embeddings = 262144


_FAKE_ITEMS = [
    {"_id": "1", "question": "q1?", "choices": ["a", "b", "c", "d"], "answer": "A", "context": "doc1"},
    {"_id": "2", "question": "q2?", "choices": ["a", "b", "c", "d"], "answer": "B", "context": "doc2"},
]


def _manifest(spec=None, **overrides):
    base = dict(
        model_identity=model_identity("/models/qwen", _FakeConfig()),
        spec=spec,
        max_context=16384,
        max_new_tokens=128,
        dataset_name="THUDM/LongBench-v2",
        limit=None,
        n_items=503,
        items_sha256=items_fingerprint(_FAKE_ITEMS),
    )
    base.update(overrides)
    return build_manifest(**base)


def test_model_identity_pulls_cheap_fields_only():
    ident = model_identity("/models/qwen", _FakeConfig())
    assert ident["model_dir"] == "/models/qwen"
    assert ident["model_type"] == "qwen3_5"
    assert ident["hidden_size"] == 1024
    assert ident["max_position_embeddings"] == 262144


def test_spec_fingerprint_none_is_baseline():
    assert spec_fingerprint(None) is None


def test_spec_fingerprint_identity_and_content(small_spec):
    fp = spec_fingerprint(small_spec)
    assert fp["d"] == small_spec.d
    assert fp["k"] == small_spec.k
    assert fp["n_levels"] == small_spec.n_levels
    assert "codebook_sha256" in fp and "rotation_sha256" in fp

    # A content-only change (same shape/identity, different trained values)
    # must change the fingerprint -- identity fields alone aren't enough to
    # tell two differently-trained checkpoints apart.
    import copy
    perturbed = copy.deepcopy(small_spec)
    perturbed.codebook = perturbed.codebook + 1.0
    fp2 = spec_fingerprint(perturbed)
    assert fp2["codebook_sha256"] != fp["codebook_sha256"]
    assert fp2["d"] == fp["d"] and fp2["n_levels"] == fp["n_levels"]


def test_items_fingerprint_content_sensitive():
    base = {"_id": "1", "question": "q?", "choices": ["a", "b", "c", "d"], "answer": "A", "context": "doc"}
    same = dict(base)
    a = items_fingerprint([base])
    assert a == items_fingerprint([same])  # identical content -> identical hash
    # A changed field under the *same* _id must change the fingerprint --
    # otherwise a dataset revision (edited question/choices/answer/context)
    # with stable ids would look like the same scope and silently mix runs.
    for field, new_value in (
        ("question", "different question?"),
        ("answer", "B"),
        ("context", "different doc"),
        ("choices", ["w", "x", "y", "z"]),
    ):
        changed = dict(base, **{field: new_value})
        assert items_fingerprint([changed]) != a, f"{field} change was not detected"


def test_items_fingerprint_is_order_sensitive():
    item1 = {"_id": "1", "question": "q1", "choices": ["a", "b", "c", "d"], "answer": "A", "context": "doc1"}
    item2 = {"_id": "2", "question": "q2", "choices": ["a", "b", "c", "d"], "answer": "B", "context": "doc2"}
    a = items_fingerprint([item1, item2])
    b = items_fingerprint([item2, item1])
    assert a != b
    assert a == items_fingerprint([item1, item2])


def test_ensure_manifest_compatible_writes_manifest_for_fresh_run(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    manifest = _manifest()
    ensure_manifest_compatible(run_dir, run_dir / "results.jsonl", manifest)
    assert json.loads((run_dir / "manifest.json").read_text()) == manifest


def test_ensure_manifest_compatible_allows_resume_with_matching_manifest(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    out_jsonl = run_dir / "results.jsonl"
    manifest = _manifest()
    ensure_manifest_compatible(run_dir, out_jsonl, manifest)  # writes manifest.json
    out_jsonl.write_text('{"_id": "1"}\n')
    ensure_manifest_compatible(run_dir, out_jsonl, manifest)  # same manifest: no raise


def test_ensure_manifest_compatible_rejects_mismatched_manifest(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    out_jsonl = run_dir / "results.jsonl"
    ensure_manifest_compatible(run_dir, out_jsonl, _manifest())
    out_jsonl.write_text('{"_id": "1"}\n')
    with pytest.raises(RuntimeError, match="manifest mismatch"):
        ensure_manifest_compatible(run_dir, out_jsonl, _manifest(max_context=8192))


def test_ensure_manifest_compatible_rejects_mismatched_spec(tmp_path, small_spec):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    out_jsonl = run_dir / "results.jsonl"
    ensure_manifest_compatible(run_dir, out_jsonl, _manifest(spec=None))
    out_jsonl.write_text('{"_id": "1"}\n')
    with pytest.raises(RuntimeError, match="manifest mismatch"):
        ensure_manifest_compatible(run_dir, out_jsonl, _manifest(spec=small_spec))


def test_ensure_manifest_compatible_rejects_existing_output_without_manifest(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    out_jsonl = run_dir / "results.jsonl"
    out_jsonl.write_text('{"_id": "1"}\n')  # legacy/pre-manifest output: no manifest.json
    with pytest.raises(RuntimeError, match="refusing to"):
        ensure_manifest_compatible(run_dir, out_jsonl, _manifest())


def test_read_jsonl_rows_no_truncation_for_clean_file(tmp_path):
    path = tmp_path / "results.jsonl"
    path.write_text('{"_id": "1"}\n{"_id": "2"}\n')
    rows, truncated = read_jsonl_rows(path)
    assert [r["_id"] for r in rows] == ["1", "2"]
    assert truncated is False


def test_read_jsonl_rows_tolerates_truncated_trailing_line(tmp_path):
    path = tmp_path / "results.jsonl"
    # A killed-mid-write process: two complete rows, one truncated last line.
    path.write_text('{"_id": "1"}\n{"_id": "2"}\n{"_id": "3", "trunc')
    rows, truncated = read_jsonl_rows(path)
    assert [r["_id"] for r in rows] == ["1", "2"]
    assert truncated is True
    # read_jsonl_rows only reports the truncation -- it must not itself
    # touch the file (repair is a separate, explicit step).
    assert path.read_text() == '{"_id": "1"}\n{"_id": "2"}\n{"_id": "3", "trunc'


def test_read_jsonl_rows_raises_on_non_trailing_corruption(tmp_path):
    path = tmp_path / "results.jsonl"
    path.write_text('{"_id": "1"\n{"_id": "2"}\n')  # corrupt *first* line, not the last
    with pytest.raises(json.JSONDecodeError):
        read_jsonl_rows(path)


def test_repair_truncated_jsonl_rewrites_exactly_the_valid_rows(tmp_path):
    path = tmp_path / "results.jsonl"
    path.write_text('{"_id": "1"}\n{"_id": "2"}\n{"_id": "3", "trunc')
    rows, truncated = read_jsonl_rows(path)
    assert truncated is True
    repair_truncated_jsonl(path, rows)
    # File on disk is now exactly the valid rows, newline-terminated, no
    # trace of the partial trailing bytes.
    assert path.read_text() == '{"_id": "1"}\n{"_id": "2"}\n'
    rows2, truncated2 = read_jsonl_rows(path)
    assert [r["_id"] for r in rows2] == ["1", "2"]
    assert truncated2 is False


def test_load_and_repair_jsonl_fixes_truncated_file_in_place(tmp_path):
    path = tmp_path / "results.jsonl"
    path.write_text('{"_id": "1"}\n{"_id": "2"}\n{"_id": "3", "trunc')
    rows = load_and_repair_jsonl(path)
    assert [r["_id"] for r in rows] == ["1", "2"]
    assert path.read_text() == '{"_id": "1"}\n{"_id": "2"}\n'


def test_load_and_repair_jsonl_is_a_noop_on_a_clean_file(tmp_path):
    path = tmp_path / "results.jsonl"
    original = '{"_id": "1"}\n{"_id": "2"}\n'
    path.write_text(original)
    rows = load_and_repair_jsonl(path)
    assert [r["_id"] for r in rows] == ["1", "2"]
    assert path.read_text() == original  # untouched: nothing to repair


def test_append_after_truncated_trailing_line_yields_valid_complete_jsonl(tmp_path):
    """Regression: appending after a truncated trailing line must not corrupt the file.

    Before the repair step existed, read_jsonl_rows only dropped the
    partial line *in memory*; the bytes stayed on disk, so a subsequent
    `open(path, "a")` (exactly what run_config does before generating the
    next pending row) concatenated a new JSON line right onto the partial
    bytes, permanently corrupting every line after it. load_and_repair_jsonl
    must truncate the file to the last complete row *before* anything else
    appends to it.
    """
    path = tmp_path / "results.jsonl"
    path.write_text('{"_id": "1"}\n{"_id": "2"}\n{"_id": "3", "trunc')

    # This is the resume-time read run_config performs before opening the
    # file in append mode.
    rows = load_and_repair_jsonl(path)
    assert [r["_id"] for r in rows] == ["1", "2"]

    # Simulate run_config appending the freshly (re-)generated row 3.
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"_id": "3"}) + "\n")

    # Every line must now parse as one complete JSON object -- the corrupt
    # tail must not have merged with the new row.
    text = path.read_text()
    lines = [line for line in text.splitlines() if line]
    parsed = [json.loads(line) for line in lines]  # raises if any line is malformed
    assert [r["_id"] for r in parsed] == ["1", "2", "3"]

    final_rows, truncated = read_jsonl_rows(path)
    assert [r["_id"] for r in final_rows] == ["1", "2", "3"]
    assert truncated is False


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
