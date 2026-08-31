"""run_eval / _generation_patch / FibQuantRuntime interaction regressions.

The old `_patch_generation` stacked an irreversible, process-global override
onto `lm_eval.models.huggingface.HFLM._model_generate`: two `run_eval()` calls
with different penalties or specs left the first override in place forever
(nothing to undo it), and a failure mid-run leaked whatever was half-applied.
These tests exercise the scoped replacement (contextlib.ExitStack +
unittest.mock.patch.object for the generate-path penalty patch,
FibQuantRuntime's context manager for the cache adapter): every patch must be
gone -- restored to exactly what it was before -- once run_eval() returns,
whether it returned normally, raised, or is about to run again with a
different config. lm_eval and a real (unloaded) Qwen3.5 model class are used
directly rather than re-mocked, so these tests also cover the class-level
default target `FibQuantRuntime` resolves in production (see
runtime._default_model_class).
"""

from unittest import mock

import lm_eval.models.huggingface as hf_mod
import pytest

from fibquant.eval_harness import EvalConfig, PresencePenaltyLogitsProcessor, _generation_patch, run_eval
from fibquant.runtime import active_specs


class _FakeHFLM:
    """Stand-in for an HFLM instance: _generation_patch's wrapper only reads
    `self` off to call through to the original, so no real model is needed to
    exercise the patch contract."""


def _install_fake_original():
    """Replace HFLM._model_generate with a spy that records generation_kwargs,
    returning (fake_original, captured) so tests can assert on what the
    scoped patch forwarded down."""
    captured = []

    def fake_original(self, context, max_length, stop, **generation_kwargs):
        captured.append(generation_kwargs)
        return "generated"

    return fake_original, captured


def test_generation_patch_scopes_and_restores_model_generate():
    fake_original, captured = _install_fake_original()
    with mock.patch.object(hf_mod.HFLM, "_model_generate", fake_original):
        before = hf_mod.HFLM._model_generate
        with _generation_patch(0.5):
            assert hf_mod.HFLM._model_generate is not before
            result = hf_mod.HFLM._model_generate(_FakeHFLM(), "ctx", 10, ["stop"])
            assert result == "generated"
            processors = captured[-1]["logits_processor"]
            assert len(processors) == 1
            assert isinstance(processors[0], PresencePenaltyLogitsProcessor)
            assert processors[0].penalty == 0.5
        # scoped: fully restored once the `with` block exits
        assert hf_mod.HFLM._model_generate is before


def test_generation_patch_pops_unsupported_kwarg_without_penalty():
    fake_original, captured = _install_fake_original()
    with mock.patch.object(hf_mod.HFLM, "_model_generate", fake_original):
        with _generation_patch(0.0):
            hf_mod.HFLM._model_generate(
                _FakeHFLM(), "ctx", 10, ["stop"], presence_penalty=0.0
            )
        # transformers 5.x has no presence_penalty on GenerationConfig: always
        # popped, even when the penalty itself is 0 (falsy) and no processor
        # is added.
        assert "presence_penalty" not in captured[-1]
        assert "logits_processor" not in captured[-1]


def test_generation_patch_appends_to_caller_supplied_logits_processor_list():
    """Regression: the patch used to *overwrite* logits_processor, silently
    dropping any processor a caller had already wired in via gen_kwargs.
    It must append instead, on both a plain list and a LogitsProcessorList,
    without mutating the caller's original object."""
    from transformers.generation.logits_process import LogitsProcessorList

    class _CallerProcessor:
        def __call__(self, input_ids, scores):
            return scores

    fake_original, captured = _install_fake_original()
    with mock.patch.object(hf_mod.HFLM, "_model_generate", fake_original):
        caller_processor = _CallerProcessor()
        caller_list = LogitsProcessorList([caller_processor])
        with _generation_patch(0.5):
            hf_mod.HFLM._model_generate(
                _FakeHFLM(), "ctx", 10, ["stop"], logits_processor=caller_list
            )
        result = captured[-1]["logits_processor"]
        assert caller_processor in result  # caller's processor kept, not discarded
        assert any(isinstance(p, PresencePenaltyLogitsProcessor) for p in result)
        assert len(result) == 2
        assert isinstance(result, LogitsProcessorList)  # type preserved
        assert list(caller_list) == [caller_processor]  # caller's own list untouched

        # a plain list (not LogitsProcessorList) is handled the same way
        plain_list = [caller_processor]
        with _generation_patch(0.5):
            hf_mod.HFLM._model_generate(
                _FakeHFLM(), "ctx", 10, ["stop"], logits_processor=plain_list
            )
        result2 = captured[-1]["logits_processor"]
        assert caller_processor in result2
        assert any(isinstance(p, PresencePenaltyLogitsProcessor) for p in result2)
        assert len(result2) == 2
        assert plain_list == [caller_processor]  # caller's own list untouched


def test_repeated_generation_patches_do_not_stack_and_use_latest_penalty():
    fake_original, captured = _install_fake_original()
    with mock.patch.object(hf_mod.HFLM, "_model_generate", fake_original):
        with _generation_patch(0.3):
            hf_mod.HFLM._model_generate(_FakeHFLM(), "ctx", 10, ["stop"])
        # first patch is gone: no lingering wrapper for the second to nest inside
        assert hf_mod.HFLM._model_generate is fake_original
        with _generation_patch(0.9):
            hf_mod.HFLM._model_generate(_FakeHFLM(), "ctx", 10, ["stop"])
        assert hf_mod.HFLM._model_generate is fake_original

        first_penalty = captured[0]["logits_processor"][0].penalty
        second_penalty = captured[1]["logits_processor"][0].penalty
        assert first_penalty == 0.3
        assert second_penalty == 0.9  # second run reflects its own config, not the first's


def _run_eval_with_mocked_simple_evaluate(config, side_effect=None):
    """Patch lm_eval.simple_evaluate so run_eval never loads a real model;
    side_effect lets a test observe runtime/patch state from inside the call
    or force a failure."""
    calls = []

    def fake_simple_evaluate(**kwargs):
        calls.append(kwargs)
        if side_effect is not None:
            side_effect()
        return {"results": {}}

    with mock.patch("lm_eval.simple_evaluate", side_effect=fake_simple_evaluate):
        results = run_eval(config)
    return results, calls


def test_run_eval_restores_generation_patch_and_runtime_on_exception(small_spec, tmp_path):
    before = hf_mod.HFLM._model_generate
    config = EvalConfig(
        spec=small_spec,
        gen_kwargs={"presence_penalty": 0.4},
        output_dir=str(tmp_path / "out"),
        tag="boom",
    )

    def boom():
        # while inside the scoped block: both patches must be live
        assert hf_mod.HFLM._model_generate is not before
        assert active_specs() != {}
        raise RuntimeError("simple_evaluate blew up")

    with pytest.raises(RuntimeError, match="blew up"):
        _run_eval_with_mocked_simple_evaluate(config, side_effect=boom)

    # ExitStack unwound despite the exception: nothing left installed
    assert hf_mod.HFLM._model_generate is before
    assert active_specs() == {}


def test_run_eval_repeat_runs_use_latest_spec_and_penalty(small_spec, packed_spec, tmp_path):
    seen = []

    def record():
        specs = active_specs()
        assert len(specs) == 1
        seen.append(next(iter(specs.values())).n_levels)

    config_a = EvalConfig(
        spec=small_spec,
        gen_kwargs={"presence_penalty": 0.2},
        output_dir=str(tmp_path / "a"),
        tag="a",
    )
    config_b = EvalConfig(
        spec=packed_spec,
        gen_kwargs={"presence_penalty": 0.8},
        output_dir=str(tmp_path / "b"),
        tag="b",
    )

    _run_eval_with_mocked_simple_evaluate(config_a, side_effect=record)
    assert active_specs() == {}  # first run fully torn down before the second starts
    _run_eval_with_mocked_simple_evaluate(config_b, side_effect=record)
    assert active_specs() == {}

    assert seen == [small_spec.n_levels, packed_spec.n_levels]  # second run sees its own spec, not the first's


def test_run_eval_without_spec_or_penalty_never_installs_anything(tmp_path):
    before = hf_mod.HFLM._model_generate
    config = EvalConfig(output_dir=str(tmp_path / "out"), tag="baseline")
    results, calls = _run_eval_with_mocked_simple_evaluate(config)
    assert results == {"results": {}}
    assert len(calls) == 1
    assert hf_mod.HFLM._model_generate is before
    assert active_specs() == {}
