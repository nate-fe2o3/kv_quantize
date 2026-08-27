"""Eval harness pure parts: gen-kwargs parsing, penalty processor, config, output."""

import torch

from fibquant.eval_harness import (
    EvalConfig,
    PresencePenaltyLogitsProcessor,
    parse_gen_kwargs,
    write_results,
)


def test_parse_gen_kwargs():
    assert parse_gen_kwargs(None) is None
    assert parse_gen_kwargs("") is None
    got = parse_gen_kwargs("do_sample=true,temperature=0.7,top_p=0.8,max_new_tokens=12,sys=abc")
    assert got == {
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.8,
        "max_new_tokens": 12,
        "sys": "abc",
    }
    got = parse_gen_kwargs("a=False")
    assert got == {"a": False}


def test_presence_penalty_processor():
    processor = PresencePenaltyLogitsProcessor(penalty=0.5)
    scores = torch.ones(1, 6)
    input_ids = torch.tensor([[1, 2, 3]])
    out = processor(input_ids, scores)
    assert torch.equal(out[0, 1:4], torch.full((3,), 0.5))
    assert torch.equal(out[0, [0, 4, 5]], torch.ones(3))


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
