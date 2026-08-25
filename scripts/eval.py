"""Run lm-eval on Qwen3.5-0.8B, optionally with FibQuant KV compression.

Usage:
    # baseline
    .venv/bin/python scripts/eval.py --tag baseline --tasks hellaswag,wikitext
    # fibquant (b=2)
    .venv/bin/python scripts/eval.py --tag fibquant-b2 --fibquant --tasks hellaswag,wikitext
    # quick smoke
    .venv/bin/python scripts/eval.py --tag smoke --fibquant --tasks hellaswag --limit 20
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")

MODEL_DIR = "models/Qwen3.5-0.8B"


def _patch_generation(penalty: float, spec) -> None:
    """Patch lm-eval's generate path:
    - transformers 5.x dropped `presence_penalty`; emulate the OpenAI-style
      additive penalty with a logits processor.
    - when a FibQuant spec is given, inject a FibQuantCache so generate()
      actually compresses (lm-eval passes no cache, so generate would
      otherwise build a plain DynamicCache and skip compression entirely).
    """
    import lm_eval.models.huggingface as hf_mod
    import torch
    from transformers.generation.logits_process import LogitsProcessor

    from fibquant import FibQuantCache

    if penalty:

        class PresencePenaltyLogitsProcessor(LogitsProcessor):
            def __init__(self, penalty: float):
                self.penalty = penalty

            def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
                generated = torch.unique(input_ids)
                scores[:, generated] = scores[:, generated] - self.penalty
                return scores

    original = hf_mod.HFLM._model_generate

    def patched(self, context, max_length, stop, **generation_kwargs):
        generation_kwargs.pop("presence_penalty", None)
        if penalty:
            generation_kwargs["logits_processor"] = [PresencePenaltyLogitsProcessor(penalty)]
        if spec is not None:
            generation_kwargs["past_key_values"] = FibQuantCache(config=self.model.config, spec=spec)
        return original(self, context, max_length=max_length, stop=stop, **generation_kwargs)

    hf_mod.HFLM._model_generate = patched


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, required=True, help="output subdirectory under results/qwen3.5-0.8b/")
    parser.add_argument("--tasks", type=str, default="hellaswag,wikitext")
    parser.add_argument("--limit", type=int, default=None, help="eval limit (testing only)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--fibquant", action="store_true", help="enable FibQuant b=2 compression")
    parser.add_argument("--spec", type=str, default="models/fibquant/fibquant_d256_k4_N256.pt")
    parser.add_argument("--output-dir", type=str, default="results/qwen3.5-0.8b")
    parser.add_argument("--apply-chat-template", action="store_true", help="wrap docs with the model chat template")
    parser.add_argument("--cache-requests", action="store_true", help="cache lm-eval requests to disk (retry-safe)")
    parser.add_argument("--system-instruction", type=str, default=None, help="system message for chat template")
    parser.add_argument(
        "--gen-kwargs",
        type=str,
        default=None,
        help="comma-separated k=v overrides for generation kwargs, e.g. do_sample=true,temperature=0.7,top_p=0.8",
    )
    args = parser.parse_args()

    if args.fibquant:
        from fibquant import FibQuantSpec, enable_fibquant, load_spec

        spec = FibQuantSpec.from_checkpoint(load_spec(args.spec))
        logging.info("enabling FibQuant: d=%d k=%d N=%d b=%.1f bits/coord", spec.d, spec.k, spec.n_levels, spec.bits_per_coord)
        enable_fibquant(None, spec)

    gen_kwargs = None
    if args.gen_kwargs:
        gen_kwargs = {}
        for kv in args.gen_kwargs.split(","):
            key, value = kv.split("=", 1)
            lowered = value.lower()
            if lowered in ("true", "false"):
                gen_kwargs[key] = lowered == "true"
            else:
                try:
                    gen_kwargs[key] = int(value) if "." not in value else float(value)
                except ValueError:
                    gen_kwargs[key] = value

    if gen_kwargs and "presence_penalty" in gen_kwargs:
        logging.info(
            "presence_penalty=%.1f requested: transformers 5.x removed it from GenerationConfig, "
            "injecting a PresencePenaltyLogitsProcessor via HFLM._model_generate",
            gen_kwargs["presence_penalty"],
        )
    if gen_kwargs and "presence_penalty" in gen_kwargs or args.fibquant:
        _patch_generation(
            penalty=gen_kwargs.get("presence_penalty", 0.0) if gen_kwargs else 0.0,
            spec=spec if args.fibquant else None,
        )

    from lm_eval import simple_evaluate

    # Flaky MPS segfault in transformers' threaded weight copy (_materialize_copy);
    # serialize loading to a single worker.
    import transformers.core_model_loading as cml

    cml.GLOBAL_WORKERS = 1

    model_args = {
        "pretrained": MODEL_DIR,
        "dtype": "bfloat16",
        "max_length": args.max_length,
        "device": "mps",
    }
    results = simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=args.tasks.split(","),
        num_fewshot=None,
        batch_size=args.batch_size,
        limit=args.limit,
        apply_chat_template=args.apply_chat_template,
        system_instruction=args.system_instruction,
        gen_kwargs=gen_kwargs,
        cache_requests=args.cache_requests,
    )

    out_dir = Path(args.output_dir) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "results.json"

    def _json_default(o):
        if hasattr(o, "item"):
            return o.item()
        if isinstance(o, (Path,)):
            return str(o)
        return str(o)

    path.write_text(json.dumps(results, indent=2, default=_json_default))
    logging.info("results written to %s", path)


if __name__ == "__main__":
    main()
