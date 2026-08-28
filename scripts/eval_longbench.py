"""LongBench v2 (THUDM) evaluation for Qwen3.5-0.8B with/without FibQuant.

Replicates the official *non-thinking* protocol (THUDM/LongBench repo,
``pred.py`` with ``--cot`` unset == the leaderboard "w/o CoT" column):

  * 503 real-world MCQs, 0-shot, contexts 8k-2M words (majority <128k)
  * prompt = ``prompts/0shot.txt`` verbatim, rendered by the model chat
    template as a single user turn with **thinking disabled**
    (``enable_thinking=False``; output mode is direct answer, not CoT)
  * generation: greedy, ``max_new_tokens=128`` (the official non-thinking
    cap; the official runtime uses temperature=0.1 -- greedy is the repo
    convention for deterministic paired comparisons and matches how every
    baseline in this repo runs)
  * scoring: official regex extraction of ``The correct answer is (X)``,
    exact letter match, no LLM judge; extraction misses are wrong
  * overlong contexts: official-style *middle truncation* (keep first half +
    last half) at the model's context window (Qwen3.5-0.8B = 262,144)

Config identical to baseline (scripts/eval_cuda.py, scripts/multi_needle.py):

  * same model and load path: ``load_model(MODEL_DIR, DEVICE)`` (bf16,
    single-worker fix), same ``MODELS_DIR`` volume layout
  * same greedy generate call shape and cache injection:
    ``generate(..., do_sample=False, past_key_values=FibQuantCache(...))``;
    ``spec=None`` is the fp16 baseline, exactly like multi_needle.py
  * no presence penalty, no RAG, no few-shot, no system prompt

Intentional, benchmark-mandated differences from the other probes:
  * ``MAX_NEW_TOKENS=128`` (official non-thinking cap; needles use 24)
  * context window 262,144 (model max; needles cap at 16,384)
  * one config runs all 503 questions (paired baseline/FibQuant configs run
    in one invocation over the *same* cached prompts, as multi_needle does)

Run on Databricks CUDA as a notebook cell or ``python scripts/eval_longbench.py``;
constants at the top, no CLI args. A fresh FibQuantCache per question (reusing
one would append the continuation across questions).
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from fibquant import FibQuantCache, FibQuantSpec
from fibquant.eval_harness import load_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")

# --- paths (Databricks volume layout) ------------------------------------
MODEL_DIR = "/Volumes/security_engineering/nbutton/q34b/models/Qwen3.5-0.8B/"
DEVICE = "cuda"
DATASET_NAME = "THUDM/LongBench-v2"  # official 503-question LongBench v2
OUTPUT_DIR = "/Volumes/security_engineering/nbutton/q34b/results/qwen3.5-0.8b"
# Cache HF datasets under a UC volume so downloads survive serverless restarts
# (the default /tmp cache is ephemeral); None keeps the default cache.
DATASETS_CACHE_DIR = None
if DATASETS_CACHE_DIR:
    os.environ.setdefault("HF_DATASETS_CACHE", DATASETS_CACHE_DIR)

# --- run configuration ---------------------------------------------------
LIMIT = None  # number of questions to run (smoke test only; None = all 503)
MAX_CONTEXT = 262144  # Qwen3.5-0.8B max_position_embeddings; None = read from model config
MAX_NEW_TOKENS = 128  # official non-thinking output cap
INCLUDE_BASELINE = True  # fp16 (no FibQuant) as the reference config
BITS = []  # e.g. [2] -- FibQuant bits/coord pairs; [] when SPEC_PATHS is set
SPEC_PATHS = [
    "/Volumes/security_engineering/nbutton/q34b/models/fibquant/fibquant_d256_k4_N256.pt",
    "/Volumes/security_engineering/nbutton/q34b/models/fibquant/fibquant_d256_k4_N4096.pt",
]

# --- official THUDM non-thinking template (prompts/0shot.txt, verbatim) ---
PROMPT_0SHOT = (
    "Please read the following text and answer the question below.\n\n"
    "<text>\n$DOC$\n</text>\n\n"
    "What is the correct answer to this question: $Q$\n"
    "Choices:\n"
    "(A) $C_A$\n"
    "(B) $C_B$\n"
    "(C) $C_C$\n"
    "(D) $C_D$\n\n"
    'Format your response as follows: "The correct answer is '
    '(insert answer here)".'
)

# Official extraction (pred.py): strip '*', then the parenthesized pattern,
# then the bare-letter fallback. No match -> None -> counts as wrong.
_ANS_PAREN_RE = re.compile(r"The correct answer is \(([A-D])\)")
_ANS_BARE_RE = re.compile(r"The correct answer is ([A-D])")


def normalize_item(item: dict) -> dict:
    """Normalize one dataset row to the {context, question, choices[4], answer} shape.

    Handles the official THUDM field names (choice_A..choice_D) and the
    common mirror variant that flattens choices to a list.
    """
    if all(k in item for k in ("choice_A", "choice_B", "choice_C", "choice_D")):
        choices = [item["choice_A"], item["choice_B"], item["choice_C"], item["choice_D"]]
    else:
        choices = list(item["choices"])
    return {
        "_id": item.get("_id"),
        "domain": item.get("domain"),
        "sub_domain": item.get("sub_domain"),
        "difficulty": item.get("difficulty"),
        "length": item.get("length"),
        "question": item["question"],
        "choices": choices,
        "answer": item["answer"],
        "context": item["context"],
    }


def render_prompt_text(doc: str, question: str, choices: list[str]) -> str:
    """Render the official 0-shot prompt for one question."""
    if len(choices) != 4:
        raise ValueError(f"need 4 choices, got {len(choices)}")
    return (
        PROMPT_0SHOT.replace("$DOC$", doc)
        .replace("$Q$", question.strip())
        .replace("$C_A$", choices[0].strip())
        .replace("$C_B$", choices[1].strip())
        .replace("$C_C$", choices[2].strip())
        .replace("$C_D$", choices[3].strip())
    )


def middle_truncate_ids(ids: list[int], budget: int) -> list[int]:
    """Official truncation: keep the first half and the last half.

    The question/choices/formatter sit at the end of the rendered prompt, so
    the tail half always preserves the answerable part of the request.
    """
    if len(ids) <= budget:
        return ids
    half = budget // 2
    return ids[:half] + ids[len(ids) - (budget - half):]


def _chat_encode(tokenizer, text: str) -> list[int]:
    """Encode one user turn with the model's chat template, thinking OFF."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )["input_ids"]


def build_input_ids(
    tokenizer: AutoTokenizer,
    item: dict,
    max_context: int,
) -> tuple[list[int], int, bool]:
    """Prompt ids + (original doc token count, truncated) for one question.

    Truncation order: first trim the *document* to the budget (doc-level
    middle cut, question/choices always intact); if template re-encoding
    drift still exceeds the budget, fall back to the official whole-prompt
    middle cut (which keeps the question/choices in its tail half).
    """
    doc = item["context"]
    doc_ids = tokenizer.encode(doc, add_special_tokens=False)
    ids = _chat_encode(tokenizer, render_prompt_text(doc, item["question"], item["choices"]))
    if len(ids) <= max_context:
        return ids, len(doc_ids), False

    # Fixed overhead: everything outside the document (template + chat tokens).
    over = len(ids) - len(doc_ids)
    budget = max_context - over - 8  # slack for cross-boundary re-merge drift
    if budget <= 0:
        raise ValueError(f"max_context {max_context} too small for the fixed prompt overhead {over}")
    truncated = middle_truncate_ids(doc_ids, budget)
    doc_cut = tokenizer.decode(truncated)
    ids = _chat_encode(tokenizer, render_prompt_text(doc_cut, item["question"], item["choices"]))
    if len(ids) > max_context:  # re-encoding drift: official whole-prompt cut
        ids = middle_truncate_ids(ids, max_context)
    assert len(ids) <= max_context
    return ids, len(doc_ids), True


def extract_answer(response: str) -> str | None:
    """Official answer extraction; None means the regex missed."""
    response = response.replace("*", "")
    match = _ANS_PAREN_RE.search(response)
    if match:
        return match.group(1)
    match = _ANS_BARE_RE.search(response)
    return match.group(1) if match else None


def summarize(rows: list[dict]) -> dict:
    """Official headline numbers: overall + easy/hard + short/medium/long.

    A missing extraction counts as wrong (official result.py: judge = pred
    == answer, and None never equals a letter). Extraction-miss rate and the
    rows' token stats are reported alongside, since they explain a score.
    """
    accs: dict[str, list[int]] = {
        "overall": [], "easy": [], "hard": [], "short": [], "medium": [], "long": [],
    }
    for row in rows:
        hit = 1 if row["pred"] == row["answer"] else 0
        accs["overall"].append(hit)
        for key in ("difficulty", "length"):
            if row[key]:
                accs.setdefault(row[key], []).append(hit)
    summary = {
        "n": len(rows),
        "n_parse_fail": sum(1 for r in rows if r["pred"] is None),
        "mean_doc_tokens": round(sum(r["n_doc_tokens"] for r in rows) / max(len(rows), 1), 1),
        "n_truncated": sum(1 for r in rows if r["truncated"]),
    }
    for key, hits in accs.items():
        if key == "overall":
            summary["overall"] = round(100 * sum(hits) / len(hits), 1) if hits else 0.0
        else:
            summary[key] = round(100 * sum(hits) / len(hits), 1) if hits else 0.0
    return summary


def run_config(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompts: list[dict],
    spec: FibQuantSpec | None,
    out_jsonl: Path,
) -> tuple[list[dict], float]:
    """Generate every prompt once with one cache configuration; resumes.

    Generates sequentially (one 262k-token prompt at a time; batching long
    prompts multiplies peak memory) and appends one JSON line per question,
    so killed runs resume by _id. Returns (rows, wall seconds).
    """
    config_text = model.config.text_config
    pending: list[dict] = []
    done_ids: set[str] = set()
    if out_jsonl.exists():
        with open(out_jsonl, encoding="utf-8") as f:
            done_ids = {json.loads(line)["_id"] for line in f}
    for p in prompts:
        if p["_id"] not in done_ids:
            pending.append(p)
    if done_ids:
        logging.info("resuming: %d/%d already in %s", len(done_ids), len(prompts), out_jsonl)

    rows: list[dict] = []
    t0 = time.time()
    with torch.no_grad(), open(out_jsonl, "a", encoding="utf-8") as fout:
        for i, p in enumerate(pending, 1):
            ids = torch.tensor([p["input_ids"]], dtype=torch.long, device=model.device)
            # A fresh cache per question: generate() appends the continuation
            # into the passed cache, so reusing one would mix questions.
            cache = FibQuantCache(config=config_text, spec=spec)
            out = model.generate(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                past_key_values=cache,
            )
            response = tokenizer.decode(out[0, ids.shape[-1]:], skip_special_tokens=True)
            pred = extract_answer(response)
            row = {
                "_id": p["_id"],
                "domain": p["domain"],
                "sub_domain": p["sub_domain"],
                "difficulty": p["difficulty"],
                "length": p["length"],
                "answer": p["answer"],
                "pred": pred,
                "judge": pred == p["answer"],
                "response": response,
                "n_input_tokens": len(p["input_ids"]),
                "n_doc_tokens": p["n_doc_tokens"],
                "truncated": p["truncated"],
            }
            rows.append(row)
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()
            if i % 25 == 0 or i == len(pending):
                logging.info("  %d/%d generated (%.0fs)", i, len(pending), time.time() - t0)
    # Return every row on disk (resumed + fresh) so summaries never go partial.
    with open(out_jsonl, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    return rows, time.time() - t0


def main() -> None:
    specs: list[tuple[str, FibQuantSpec | None]] = []
    if INCLUDE_BASELINE:
        specs.append(("baseline", None))
    for bits in BITS:
        specs.append((f"fq-b{bits}", FibQuantSpec.from_bits(d=256, k=4, bits=bits)))
    for path in SPEC_PATHS:
        spec = FibQuantSpec.from_path(path)
        specs.append((f"fq-N{spec.n_levels}", spec))
    if not specs:
        raise ValueError("set BITS/SPEC_PATHS or INCLUDE_BASELINE so at least one configuration runs")

    model_dir = Path(MODEL_DIR)
    if not model_dir.exists():
        raise FileNotFoundError(
            f"model directory not found: {model_dir} -- upload the model to the UC volume"
        )
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = load_model(str(model_dir), DEVICE)
    max_context = MAX_CONTEXT or model.config.text_config.max_position_embeddings
    logging.info("model: %s | context window: %d", model_dir.name, max_context)
    logging.info("configs: %s", ", ".join(name for name, _ in specs))

    ds = load_dataset(DATASET_NAME, split="train")
    if LIMIT is not None:
        ds = ds.select(range(min(LIMIT, len(ds))))
    items = [normalize_item(d) for d in ds]
    if len(items) != 503:
        logging.warning("expected 503 LongBench v2 questions, got %d", len(items))

    # Prompts are configuration-independent: build once, reuse across configs
    # (paired comparison on identical inputs, like multi_needle.py).
    prompts: list[dict] = []
    for it in items:
        ids, n_doc, truncated = build_input_ids(tokenizer, it, max_context)
        prompts.append({**it, "input_ids": ids, "n_doc_tokens": n_doc, "truncated": truncated})
    logging.info(
        "prompts built: %d (%.1f%% truncated, mean %d doc tokens)",
        len(prompts),
        100 * sum(p["truncated"] for p in prompts) / max(len(prompts), 1),
        sum(p["n_doc_tokens"] for p in prompts) / max(len(prompts), 1),
    )

    out_root = Path(OUTPUT_DIR) / "longbench_v2"
    suffix = "smoke" if LIMIT else ""
    results: dict[str, dict] = {}
    for name, spec in specs:
        run_dir = out_root / (f"{name}-{suffix}" if suffix else name)
        run_dir.mkdir(parents=True, exist_ok=True)
        rows, secs = run_config(model, tokenizer, prompts, spec, run_dir / "results.jsonl")
        summary = summarize(rows)
        summary["wall_seconds"] = round(secs, 1)
        results[name] = summary
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        logging.info("[%s] done in %.0fs", name, secs)

    header = "subset  " + "".join(f"{name:>12}" for name, _ in specs)
    print("\nLongBench v2 (non-thinking) accuracy\n" + header)
    for key in ("overall", "easy", "hard", "short", "medium", "long"):
        row = "".join(f"{results[name][key]:10.1f}%" for name, _ in specs)
        print(f"{key:<8}{row}")
    print("\nparse-fail " + "".join(f"{results[name]['n_parse_fail']:>12}" for name, _ in specs))
    print("trunc " + "".join(f"{results[name]['n_truncated']:>12}" for name, _ in specs))
    print("mean-doc-tokens " + "".join(f"{results[name]['mean_doc_tokens']:>12}" for name, _ in specs))
    print(f"\nresults: {out_root}")


if __name__ == "__main__":
    main()
