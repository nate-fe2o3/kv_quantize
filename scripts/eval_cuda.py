"""Run lm-eval on Qwen3.5-0.8B, optionally with FibQuant KV compression.

Databricks notebook variant (CUDA). Configure via the constants below and
execute the file/cell directly — no CLI arguments. The script is thin:
constants -> EvalConfig -> fibquant.eval_harness.run_eval.

Databricks (serverless) setup:
1. Attach the custom environment: notebook Environment side pane >
   Base environment > Custom > select env.yaml from this repo (pins
   transformers 5.15.1 / lm-eval 0.4.12 / accelerate 1.14.0 on Python 3.12).
2. Run on Serverless GPU compute (AI Runtime).
3. Execute; REPO_ROOT auto-detects from the imported fibquant package
   (set it explicitly only if that fails).
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s"
)

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
        "or run in a cell above: %pip install -U typing_extensions"
    ) from None

# --- quantization width --------------------------------------------------
BITS = 2  # FibQuant bits/coordinate: 2, 3, or 4 (N_LEVELS = 1 << (BITS * K); the codebook checkpoint must exist)
FIBQUANT_K = 4  # coordinates per codebook block
FIBQUANT_D = 256  # per-head key/value dim

# --- environment layout --------------------------------------------------
# A Databricks notebook's CWD is not the repo root, so relative paths don't
# resolve. Auto-detect via the importable fibquant package; set REPO_ROOT
# explicitly if a different layout requires it.
REPO_ROOT = None

MODELS_DIR = "/Volumes/security_engineering/nbutton/q34b/models"


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

_MODELS_DIR = Path(MODELS_DIR).expanduser()

# --- run configuration ---------------------------------------------------
MODEL_DIR = str(_MODELS_DIR / "Qwen3.5-0.8B")
SPEC_PATH = str(_MODELS_DIR / "fibquant" / f"fibquant_d{FIBQUANT_D}_k{FIBQUANT_K}_N{1 << (BITS * FIBQUANT_K)}.pt")
ENABLE_FIBQUANT = True
TAG = f"fibquant-b{BITS}" if ENABLE_FIBQUANT else "baseline"
OUTPUT_DIR = "/Volumes/security_engineering/nbutton/q34b/results/qwen3.5-0.8b"
TASKS = ["hellaswag", "wikitext"]
BATCH_SIZE = "auto"
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


def main() -> None:
    from fibquant import FibQuantSpec
    from fibquant.eval_harness import EvalConfig, run_eval

    if not Path(MODEL_DIR).exists():
        logging.warning(
            "model directory not found: %s -- upload the model to the UC volume and set "
            "MODELS_DIR at the top of this script (or set MODEL_DIR to a HF hub repo id)",
            MODEL_DIR,
        )

    spec = None
    if ENABLE_FIBQUANT:
        if not Path(SPEC_PATH).exists():
            raise FileNotFoundError(
                f"FibQuant codebook not found: {SPEC_PATH} -- upload the codebook "
                f"checkpoint next to the model on the UC volume (models/ is gitignored) "
                f"and set MODELS_DIR at the top of this script"
            )
        spec = FibQuantSpec.from_path(SPEC_PATH)

    run_eval(
        EvalConfig(
            model_dir=MODEL_DIR,
            device="cuda",
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
