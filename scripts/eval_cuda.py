"""Run lm-eval on Qwen3.5-0.8B, optionally with FibQuant KV compression.

NVIDIA GPU variant (Databricks notebook): runs on CUDA. Configure via the
constants below and execute the file/cell directly -- no CLI arguments.

Databricks (serverless) setup:
1. Attach the custom environment: notebook Environment side pane >
   Base environment > Custom > select env.yaml from this repo (pins
   transformers 5.15.1 / lm-eval 0.4.12 / accelerate 1.14.0 on Python 3.12).
2. Run on Serverless GPU compute (AI Runtime).
3. Set REPO_ROOT below to this repo's workspace path and MODELS_DIR to the
   volume holding the checkpoints, then execute.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")

import transformers

try:
    import transformers.models.qwen3_5  # noqa: F401
except ModuleNotFoundError:
    raise RuntimeError(
        f"transformers {transformers.__version__} lacks Qwen3.5 support (need 5.15.1). "
        "Attach the custom environment from env.yaml via the notebook Environment "
        "side pane (Base environment > Custom), then re-run."
    ) from None

import typing_extensions

try:
    typing_extensions.TypedDict("_Probe", {"x": int}, extra_items=int)
except TypeError:
    raise RuntimeError(
        f"typing_extensions at {typing_extensions.__file__} is too old for lm-eval "
        "0.4.12 (PEP 728 extra_items). Attach the custom environment from env.yaml, "
        "or on a classic cluster run in a cell above: %pip install -U typing_extensions"
    ) from None

# --- quantization width --------------------------------------------------
BITS = 2  # FibQuant bits/coordinate (codebook checkpoint must exist for this)
FIBQUANT_K = 4  # coordinates per codebook block
FIBQUANT_D = 256  # per-head key/value dim
N_LEVELS = 1 << (BITS * FIBQUANT_K)  # codebook size implied by (BITS, K)

# --- environment layout --------------------------------------------------
# A Databricks notebook's CWD is not the repo root, so relative paths like
# models/... don't resolve. Set REPO_ROOT to wherever the repo code lives,
# e.g. "/Workspace/Users/<user>/kv_quantize". Required on serverless (fibquant
# is never on sys.path there). None = auto-detect: parent directory of the
# importable fibquant package (works locally and on classic clusters).
REPO_ROOT = None

# Checkpoints (models/) are gitignored, so a Databricks git folder never
# contains them. Upload models/ to DBFS or a UC volume and point MODELS_DIR
# at it, e.g. "/dbfs/FileStore/kv_quantize/models" or
# "/Volumes/<catalog>/<schema>/<volume>/models". None = <repo>/models.
MODELS_DIR = None


def _resolve_repo_root() -> Path:
    if REPO_ROOT is not None:
        return Path(REPO_ROOT).expanduser().resolve()
    try:
        import fibquant as _fq
    except ImportError:
        return Path.cwd()
    return Path(_fq.__file__).resolve().parent.parent


_REPO_ROOT = _resolve_repo_root()
sys.path.insert(0, str(_REPO_ROOT))  # keep `import fibquant` working in notebooks
logging.info("repo root: %s", _REPO_ROOT)

_MODELS_DIR = Path(MODELS_DIR).expanduser() if MODELS_DIR else _REPO_ROOT / "models"

# --- run configuration ---------------------------------------------------
MODEL_DIR = str(_MODELS_DIR / "Qwen3.5-0.8B")
SPEC_PATH = str(_MODELS_DIR / "fibquant" / f"fibquant_d{FIBQUANT_D}_k{FIBQUANT_K}_N{N_LEVELS}.pt")
ENABLE_FIBQUANT = True
TAG = f"fibquant-b{BITS}" if ENABLE_FIBQUANT else "baseline"
OUTPUT_DIR = "results/qwen3.5-0.8b"
TASKS = ["hellaswag", "wikitext"]
BATCH_SIZE = 8
MAX_LENGTH = 2048
LIMIT = None  # eval limit (testing only)
APPLY_CHAT_TEMPLATE = False
CACHE_REQUESTS = False  # cache lm-eval requests to disk (retry-safe)
SYSTEM_INSTRUCTION = None  # system message for chat template
GEN_KWARGS = None  # e.g. {"do_sample": True, "temperature": 0.7, "top_p": 0.8, "presence_penalty": 0.5}

# Cache HF datasets under a UC volume so downloads survive serverless restarts
# (the default /tmp cache is ephemeral), e.g.
#   "/Volumes/<catalog>/<schema>/<volume>/hf_cache"
DATASETS_CACHE_DIR = None

if DATASETS_CACHE_DIR:
    os.environ.setdefault("HF_DATASETS_CACHE", DATASETS_CACHE_DIR)


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
    if not Path(MODEL_DIR).exists():
        logging.warning(
            "model directory not found: %s -- upload models/ to DBFS/UC volume and set "
            "MODELS_DIR at the top of this script (or set MODEL_DIR to a HF hub repo id)",
            MODEL_DIR,
        )

    spec = None
    if ENABLE_FIBQUANT:
        if not Path(SPEC_PATH).exists():
            raise FileNotFoundError(
                f"FibQuant codebook not found: {SPEC_PATH} -- models/ is gitignored, so "
                f"upload it alongside the model checkpoints (e.g. DBFS/UC volume) and set "
                f"MODELS_DIR at the top of this script"
            )
        from fibquant import FibQuantSpec, enable_fibquant, load_spec

        spec = FibQuantSpec.from_checkpoint(load_spec(SPEC_PATH))
        logging.info("enabling FibQuant: d=%d k=%d N=%d b=%.1f bits/coord", spec.d, spec.k, spec.n_levels, spec.bits_per_coord)
        enable_fibquant(None, spec)

    gen_kwargs = GEN_KWARGS
    if gen_kwargs and "presence_penalty" in gen_kwargs:
        logging.info(
            "presence_penalty=%.1f requested: transformers 5.x removed it from GenerationConfig, "
            "injecting a PresencePenaltyLogitsProcessor via HFLM._model_generate",
            gen_kwargs["presence_penalty"],
        )
    if (gen_kwargs and "presence_penalty" in gen_kwargs) or ENABLE_FIBQUANT:
        _patch_generation(
            penalty=gen_kwargs.get("presence_penalty", 0.0) if gen_kwargs else 0.0,
            spec=spec if ENABLE_FIBQUANT else None,
        )

    from lm_eval import simple_evaluate

    model_args = {
        "pretrained": MODEL_DIR,
        "dtype": "bfloat16",
        "max_length": MAX_LENGTH,
        "device": "cuda",
    }
    results = simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=TASKS,
        num_fewshot=None,
        batch_size=BATCH_SIZE,
        limit=LIMIT,
        apply_chat_template=APPLY_CHAT_TEMPLATE,
        system_instruction=SYSTEM_INSTRUCTION,
        gen_kwargs=gen_kwargs,
        cache_requests=CACHE_REQUESTS,
    )

    out_dir = Path(OUTPUT_DIR) / TAG
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


main()
