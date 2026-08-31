"""Eval harness pure parts: penalty processor, config, output."""

import torch

from fibquant.eval_harness import (
    EvalConfig,
    PresencePenaltyLogitsProcessor,
    write_results,
)


def test_presence_penalty_processor():
    processor = PresencePenaltyLogitsProcessor(penalty=0.5)
    scores = torch.ones(1, 6)
    input_ids = torch.tensor([[1, 2, 3]])
    out = processor(input_ids, scores)
    assert torch.equal(out[0, 1:4], torch.full((3,), 0.5))
    assert torch.equal(out[0, [0, 4, 5]], torch.ones(3))


def test_presence_penalty_processor_is_row_isolated():
    """Regression: the old torch.unique(input_ids) flattened token ids across
    the whole batch, so row A's generated tokens penalized row B's scores
    too. Row 0 generated {1, 2, 3}; row 1 generated {4, 5} (5 repeated) -- the
    penalty for each row must be confined to that row's own tokens."""
    processor = PresencePenaltyLogitsProcessor(penalty=0.5)
    scores = torch.ones(2, 6)
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 5]])
    out = processor(input_ids, scores)
    assert torch.equal(out[0, [1, 2, 3]], torch.full((3,), 0.5))
    assert torch.equal(out[0, [0, 4, 5]], torch.ones(3))  # row 1's tokens untouched here
    assert torch.equal(out[1, [4, 5]], torch.full((2,), 0.5))  # repeat penalized once, not twice
    assert torch.equal(out[1, [0, 1, 2, 3]], torch.ones(4))  # row 0's tokens untouched here


def test_presence_penalty_processor_preserves_dtype_and_out_of_place():
    processor = PresencePenaltyLogitsProcessor(penalty=0.25)
    scores = torch.ones(2, 4, dtype=torch.float64)
    original = scores.clone()
    input_ids = torch.tensor([[0], [1]])
    out = processor(input_ids, scores)
    assert out.dtype == scores.dtype
    assert torch.equal(scores, original)  # input scores left untouched (out-of-place)


def test_eval_config_defaults_are_baseline():
    config = EvalConfig()
    assert config.spec is None and config.tag == "baseline"
    assert config.tasks == ["hellaswag", "wikitext"]
    assert config.max_length == 2048 and config.output_dir == "results/qwen3.5-0.8b"


def test_write_results_serializes_tensors(tmp_path):
    import json

    results = {"foo": torch.tensor(0.25), "bar": [1, 2]}
    path = write_results(results, str(tmp_path / "out"), "t")
    assert path.name == "results.json"
    parsed = json.loads(path.read_text())
    assert parsed == {"foo": 0.25, "bar": [1, 2]}
