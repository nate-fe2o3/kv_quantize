"""One deep adapter over lm-eval for FibQuant evaluations.

The harness owns every environment quirk a FibQuant eval has to work around:

  - transformers 5.x removed `presence_penalty` from GenerationConfig — an
    OpenAI-style additive penalty is emulated with a logits processor.
  - lm-eval passes no cache on the generate path, so the FibQuant spec is
    injected as an explicit `FibQuantCache` past_key_values, otherwise
    generate() builds a plain DynamicCache and skips compression entirely.
  - the MPS threaded weight-copy segfault workaround
    (`transformers.core_model_loading.GLOBAL_WORKERS = 1`) is applied by
    `load_model`, the single place models are loaded in this repo.

`run_eval(EvalConfig)` is the interface: eval.py (CLI) and eval_cuda.py
(Databricks notebook constants) are thin config objects; every other script
loads models through `load_model`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch
from transformers.generation.logits_process import LogitsProcessor

from .cache import FibQuantCache
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


def parse_gen_kwargs(kv_str: str | None) -> dict | None:
    """Parse "k=v,k2=v2" CLI strings: true/false -> bool, int/float -> numbers."""
    if not kv_str:
        return None
    gen_kwargs: dict = {}
    for kv in kv_str.split(","):
        key, value = kv.split("=", 1)
        lowered = value.lower()
        if lowered in ("true", "false"):
            gen_kwargs[key] = lowered == "true"
        else:
            try:
                gen_kwargs[key] = int(value) if "." not in value else float(value)
            except ValueError:
                gen_kwargs[key] = value
    return gen_kwargs


class PresencePenaltyLogitsProcessor(LogitsProcessor):
    """OpenAI-style additive penalty for tokens already generated.

    transformers 5.x removed `presence_penalty` from GenerationConfig, so the
    harness injects this processor instead.
    """

    def __init__(self, penalty: float):
        self.penalty = penalty

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        generated = torch.unique(input_ids)
        scores[:, generated] = scores[:, generated] - self.penalty
        return scores


def _patch_generation(penalty: float, spec: FibQuantSpec | None) -> None:
    """Patch lm-eval's generate path once: penalty emulation + cache injection."""
    import lm_eval.models.huggingface as hf_mod

    original = hf_mod.HFLM._model_generate

    def patched(self, context, max_length, stop, **generation_kwargs):
        generation_kwargs.pop("presence_penalty", None)
        if penalty:
            generation_kwargs["logits_processor"] = [
                PresencePenaltyLogitsProcessor(penalty)
            ]
        if spec is not None:
            generation_kwargs["past_key_values"] = FibQuantCache(
                config=self.model.config, spec=spec
            )
        return original(self, context, max_length=max_length, stop=stop, **generation_kwargs)

    hf_mod.HFLM._model_generate = patched


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
    """Run one lm-eval experiment and persist results; returns the results dict."""
    if config.gen_kwargs and "presence_penalty" in config.gen_kwargs:
        logger.info(
            "presence_penalty=%.1f requested: transformers 5.x removed it from "
            "GenerationConfig, injecting a PresencePenaltyLogitsProcessor via "
            "HFLM._model_generate",
            config.gen_kwargs["presence_penalty"],
        )

    patch = config.gen_kwargs is not None and "presence_penalty" in config.gen_kwargs
    if patch or config.spec is not None:
        _patch_generation(
            penalty=config.gen_kwargs.get("presence_penalty", 0.0) if config.gen_kwargs else 0.0,
            spec=config.spec,
        )

    if config.spec is not None:
        logger.info(
            "enabling FibQuant: d=%d k=%d N=%d b=%.1f bits/coord",
            config.spec.d,
            config.spec.k,
            config.spec.n_levels,
            config.spec.bits_per_coord,
        )
        FibQuantRuntime(config.spec).install()

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
