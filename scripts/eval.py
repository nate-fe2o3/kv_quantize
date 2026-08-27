"""Run lm-eval on Qwen3.5-0.8B, optionally with FibQuant KV compression.

Configure via the constants below and run the file directly — no CLI
arguments, so the same file works locally, in a notebook, and in Databricks:

    .venv/bin/python scripts/eval.py

The script is thin: constants -> EvalConfig -> fibquant.eval_harness.run_eval,
which owns the lm-eval integration quirks. Edit the constants for each run
(see scripts/eval_cuda.py for the CUDA/Databricks twin).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Make the repo root importable even when run as "python scripts/foo.py".
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")

from fibquant import FibQuantSpec
from fibquant.eval_harness import EvalConfig, run_eval

MODEL_DIR = "models/Qwen3.5-0.8B"
DEVICE = "mps"  # "cuda" on Databricks

# --- quantization width --------------------------------------------------
BITS = 2  # bits/coordinate: 2, 3, or 4 (the codebook checkpoint must exist)
FIBQUANT_K = 4
FIBQUANT_D = 256

# Explicit spec checkpoint overrides the derived default (e.g. for a nonstandard
# k); None = models/fibquant/fibquant_d{D}_k{K}_N{1 << (BITS * K)}.pt
SPEC_PATH = None
ENABLE_FIBQUANT = True

# --- run configuration ---------------------------------------------------
TAG = f"fibquant-b{BITS}" if ENABLE_FIBQUANT else "baseline"  # output subdirectory under OUTPUT_DIR
OUTPUT_DIR = "results/qwen3.5-0.8b"
TASKS = ["hellaswag", "wikitext"]
LIMIT = None  # eval limit (testing only)
BATCH_SIZE = 8
MAX_LENGTH = 2048
APPLY_CHAT_TEMPLATE = False  # wrap docs with the model chat template
CACHE_REQUESTS = False  # cache lm-eval requests to disk (retry-safe)
SYSTEM_INSTRUCTION = None  # system message for chat template
GEN_KWARGS = None  # e.g. {"do_sample": True, "temperature": 0.7, "top_p": 0.8, "presence_penalty": 0.5}


def main() -> None:
    if ENABLE_FIBQUANT:
        if SPEC_PATH is not None:
            spec = FibQuantSpec.from_path(SPEC_PATH)
        else:
            spec = FibQuantSpec.from_bits(d=FIBQUANT_D, k=FIBQUANT_K, bits=BITS)
    else:
        spec = None

    run_eval(
        EvalConfig(
            model_dir=MODEL_DIR,
            device=DEVICE,
            tasks=TASKS,
            batch_size=BATCH_SIZE,
            limit=LIMIT,
            max_length=MAX_LENGTH,
            apply_chat_template=APPLY_CHAT_TEMPLATE,
            system_instruction=SYSTEM_INSTRUCTION,
            gen_kwargs=GEN_KWARGS,
            cache_requests=CACHE_REQUESTS,
            output_dir=OUTPUT_DIR,
            tag=TAG,
            spec=spec,
        )
    )


if __name__ == "__main__":
    main()
