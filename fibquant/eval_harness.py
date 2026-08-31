"""One deep adapter over lm-eval for FibQuant evaluations.

The harness owns every environment quirk a FibQuant eval has to work around:

  - transformers 5.x removed `presence_penalty` from GenerationConfig — an
    OpenAI-style additive penalty is emulated with a logits processor spliced
    into lm-eval's HFLM._model_generate for the duration of one run_eval()
    call (contextlib.ExitStack + unittest.mock.patch.object; never a
    process-global, irreversible patch that a later call can't undo).
  - FibQuant KV compression is entirely FibQuantRuntime's job (runtime.py is
    the single cache adapter): run_eval() installs it as a context manager on
    the same ExitStack, so a failed or repeated run never leaves a stale spec
    installed for the next call to inherit.
  - the MPS threaded weight-copy segfault workaround
    (`transformers.core_model_loading.GLOBAL_WORKERS = 1`) is applied by
    `load_model`, the single place models are loaded in this repo.

`run_eval(EvalConfig)` is the interface: eval_cuda.py (Databricks notebook
constants) is a thin config object; every other script loads models through
`load_model`.
"""

from __future__ import annotations

import contextlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from unittest import mock

import torch
from transformers.generation.logits_process import LogitsProcessor

from .runtime import FibQuantRuntime
from .spec import FibQuantSpec

logger = logging.getLogger(__name__)


@dataclass
class EvalConfig:
    """Everything an lm-eval run needs; scripts only build and pass this."""

    model_dir: str = "models/Qwen3.5-0.8B"
    device: str = "mps"
    dtype: str = "bfloat16"
    tasks: list[str] = field(default_factory=lambda: ["hellaswag", "wikitext"])
    batch_size: int | str = 8
    limit: int | None = None
    max_length: int = 2048
    apply_chat_template: bool = False
    system_instruction: str | None = None
    gen_kwargs: dict | None = None
    cache_requests: bool = False
    output_dir: str = "results/qwen3.5-0.8b"
    tag: str = "baseline"
    spec: FibQuantSpec | None = None  # None => fp16 baseline


class PresencePenaltyLogitsProcessor(LogitsProcessor):
    """OpenAI-style additive penalty for tokens already generated.

    transformers 5.x removed `presence_penalty` from GenerationConfig, so the
    harness injects this processor instead.
    """

    def __init__(self, penalty: float):
        self.penalty = penalty

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """Subtract `penalty` once per row for every token id present in that row.

        A per-row scatter into a (batch, vocab) mask -- not a
        (batch, seq, vocab) one-hot -- keeps memory at O(batch * vocab)
        instead of O(batch * seq * vocab), and keeps each row's penalty
        confined to that row's own tokens: the previous `torch.unique
        (input_ids)` flattened token ids across the whole batch, so row A's
        tokens penalized row B's scores too. `scatter_` (not `scatter_add_`)
        applies the penalty once per distinct id even when a row repeats a
        token, matching OpenAI's presence_penalty semantics (a presence, not
        a frequency, penalty).
        """
        penalty_mask = torch.zeros_like(scores)
        index = input_ids.to(device=scores.device, dtype=torch.long)
        penalty_mask.scatter_(1, index, self.penalty)
        return scores - penalty_mask


def _generation_patch(penalty: float) -> contextlib.AbstractContextManager:
    """Scope HFLM._model_generate for one run_eval call: presence-penalty only.

    FibQuantCache injection is deliberately not this function's concern:
    FibQuantRuntime is the single cache adapter (see runtime.py's module
    docstring), so duplicating that here would risk two independently
    constructed caches racing to land in the same `past_key_values` kwarg.
    Built with unittest.mock.patch.object so the original method is always
    restored -- including on exception -- rather than the hand-rolled global
    overwrite this replaces, which could never be undone and left whatever
    penalty (or lack of one) was last installed active for every later call.

    The presence-penalty processor is appended to any caller-supplied
    `logits_processor` (list or LogitsProcessorList), never substituted for
    it -- overwriting the kwarg would silently drop whatever processors the
    caller already wired in.
    """
    import lm_eval.models.huggingface as hf_mod

    original = hf_mod.HFLM._model_generate

    def patched(self, context, max_length, stop, **generation_kwargs):
        generation_kwargs.pop("presence_penalty", None)
        if penalty:
            processor = PresencePenaltyLogitsProcessor(penalty)
            existing = generation_kwargs.get("logits_processor")
            # Append rather than replace: a caller-supplied processor list (or
            # LogitsProcessorList) must keep running, just with this one added
            # -- overwriting it silently drops whatever the caller wired in.
            generation_kwargs["logits_processor"] = (
                type(existing)([*existing, processor]) if existing else [processor]
            )
        return original(self, context, max_length=max_length, stop=stop, **generation_kwargs)

    return mock.patch.object(hf_mod.HFLM, "_model_generate", patched)


def load_model(model_dir: str, device: str, dtype: str = "bfloat16") -> torch.nn.Module:
    """Load the eval model (Applies the MPS single-worker fix first).

    Called by run_eval and by sanity.py / key_recall.py — the only place
    models are loaded in this repo, so the threaded weight-copy workaround
    has a single home.
    """
    import transformers.core_model_loading as cml

    cml.GLOBAL_WORKERS = 1  # flaky MPS segfault/freeze in _materialize_copy
    from transformers import AutoModelForImageTextToText

    model = AutoModelForImageTextToText.from_pretrained(model_dir, dtype=dtype)
    model.to(device)
    model.eval()
    return model


def _json_default(o):
    if hasattr(o, "item"):
        return o.item()
    return str(o)


def write_results(results: dict, output_dir: str, tag: str) -> Path:
    out_dir = Path(output_dir) / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "results.json"
    path.write_text(json.dumps(results, indent=2, default=_json_default))
    return path


def run_eval(config: EvalConfig) -> dict:
    """Run one lm-eval experiment and persist results; returns the results dict.

    Every environment patch (presence-penalty emulation, FibQuant install) is
    scoped to this call through a contextlib.ExitStack, so they are always
    torn down when the function returns *or* raises -- a second run_eval()
    with a different spec/gen_kwargs never inherits a patch left over from a
    previous call, successful or not.
    """
    penalty = None
    if config.gen_kwargs and "presence_penalty" in config.gen_kwargs:
        penalty = config.gen_kwargs["presence_penalty"]
        logger.info(
            "presence_penalty=%.1f requested: transformers 5.x removed it from "
            "GenerationConfig, injecting a PresencePenaltyLogitsProcessor via "
            "HFLM._model_generate",
            penalty,
        )

    with contextlib.ExitStack() as stack:
        if penalty is not None:
            stack.enter_context(_generation_patch(penalty))

        if config.spec is not None:
            logger.info(
                "enabling FibQuant: d=%d k=%d N=%d b=%.1f bits/coord",
                config.spec.d,
                config.spec.k,
                config.spec.n_levels,
                config.spec.bits_per_coord,
            )
            stack.enter_context(FibQuantRuntime(config.spec))

        from lm_eval import simple_evaluate

        model_args = {
            "pretrained": config.model_dir,
            "dtype": config.dtype,
            "max_length": config.max_length,
            "device": config.device,
        }
        results = simple_evaluate(
            model="hf",
            model_args=model_args,
            tasks=config.tasks,
            num_fewshot=None,
            batch_size=config.batch_size,
            limit=config.limit,
            apply_chat_template=config.apply_chat_template,
            system_instruction=config.system_instruction,
            gen_kwargs=config.gen_kwargs,
            cache_requests=config.cache_requests,
        )

    path = write_results(results, config.output_dir, config.tag)
    logger.info("results written to %s", path)
    return results
