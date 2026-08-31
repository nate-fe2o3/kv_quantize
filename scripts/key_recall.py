"""Long-range KV-cache recall probe for FibQuant on Qwen3.5-0.8B.

Needle-in-haystack for KV fidelity: for each marker-to-query *depth* and each
cache configuration (bf16 baseline + one row per FibQuant spec), the model
must recall a marker token placed `depth` tokens earlier. The same
deterministic filler windows are reused across configurations, so the cache
is the only variable; greedy decode, success = the marker appears in the
continuation. Filler is template-generated unique sentences (seeded per depth
and trial), so the background is diverse -- never a repeating passage. Batching
by depth keeps it cheap: no thinking traces
(enable_thinking=False; the default thinking mode is what makes IFEval take
~5h) and short generations.

Databricks-only (model and codebook checkpoints live on the UC volume); run as
a notebook or `python scripts/key_recall.py` on the cluster. Configure via the
constants below — no CLI arguments.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from fibquant import FibQuantCache, FibQuantSpec
from fibquant.eval_harness import load_model
from fibquant.probes import (
    SENTENCE_POOLS,
    build_spec_matrix,
    normalize_continuation,
    required_answer_tokens as _required_answer_tokens,
    unique_filler_sentence,
    validate_markers,
)

# --- paths (Databricks volume layout) ------------------------------------
MODEL_DIR = "/Volumes/security_engineering/nbutton/q34b/models/Qwen3.5-0.8B/"
DEVICE = "cuda"

# --- run configuration ---------------------------------------------------
# Cache configurations to compare, one row each in the output table.
# INCLUDE_BASELINE adds the fp16 (uncompressed) row; SPEC_PATHS are explicit
# spec checkpoints; BITS entries resolve via FibQuantSpec.from_bits (d=256,
# k=4), which uses a *repo-relative* default path that does not exist here --
# keep BITS empty and list the volume checkpoints in SPEC_PATHS.
INCLUDE_BASELINE = True
BITS = []  # e.g. [2, 3, 4] -- keep [] and use SPEC_PATHS
SPEC_PATHS = [
    "/Volumes/security_engineering/nbutton/q34b/models/fibquant/fibquant_d256_k4_N256.pt",
    "/Volumes/security_engineering/nbutton/q34b/models/fibquant/fibquant_d256_k4_N4096.pt",
]

TRIALS = 50  # trials per depth
# Trials are batched into generate() calls of this size; peak activation
# memory scales with batch * seq^2 (attention scores), so 30 x 4096 in one
# call OOMs. Lower to 5 if a long depth still OOMs, raise if the GPU has
# headroom.
TRIAL_BATCH = 10
DEPTHS = [1024, 2048, 4096]  # marker-to-query token distances
MAX_LENGTH = 4096  # cap on total prompt length (filler is truncated)
MAX_NEW_TOKENS = None  # generated tokens per trial; None = auto (verbose marker + ANSWER_BUDGET_SLACK)
ANSWER_BUDGET_SLACK = 12  # framing/prefix allowance beyond the marker itself
MARKER = "rabbit"  # recall marker (a word or a short phrase)
SEED = 0
OUT = None  # write JSON results here, or None to only print the table

SYSTEM_PROMPT = "You answer in one short phrase, nothing else."

# --- filler: template-generated unique sentences -------------------------
# Templates, word pools, and the generator live in fibquant.probes (shared
# with scripts/multi_needle.py and scripts/logit_kl.py); the marker word and
# anything containing it are absent from the pools by construction, and the
# generated sentence is decode-verified before use.


def validate_marker(marker: str, pools: dict[str, list[str]] = SENTENCE_POOLS) -> None:
    """Single-marker convenience wrapper over fibquant.probes.validate_markers.

    key_recall configures exactly one MARKER (a word or short phrase, not a
    list), so this keeps the script's public surface string-typed while
    sharing the actual invariant checks (empty/pool-reachability; duplicate
    and self-overlap checks are no-ops for a single marker).
    """
    validate_markers([marker], pools)


def required_answer_tokens(tokenizer: AutoTokenizer, markers: list[str], slack: int = ANSWER_BUDGET_SLACK) -> int:
    """Thin wrapper over fibquant.probes.required_answer_tokens binding this
    script's own ANSWER_BUDGET_SLACK as the default slack."""
    return _required_answer_tokens(tokenizer, markers, slack)


def _sentence(tokenizer: AutoTokenizer, rng: random.Random, marker: str, used: set[str]) -> list[int]:
    """One fresh, unique, marker-free filler sentence as token ids."""
    _, ids = unique_filler_sentence(tokenizer, rng, used, avoid=(marker.lower(),))
    return ids


def build_trials(
    tokenizer: AutoTokenizer,
    marker: str,
    depth: int,
    trials: int,
    seed: int,
) -> torch.Tensor:
    """Build (trials, seq) input_ids for one depth.

    The template is evaluated twice: once for the user text *with* the query
    suffix and once *without* it. The second is a strict prefix of the first,
    so filler tokens can be spliced exactly between marker and query without
    re-tokenizing.
    """
    user_prefix = f"Special token: {marker}."
    user_suffix = " Question: What is the special token? Answer:"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prefix + user_suffix},
    ]
    ids_full = list(
        tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)["input_ids"]
    )
    messages[1]["content"] = user_prefix
    ids_short = list(
        tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)["input_ids"]
    )
    # The two encodings are identical up to the character right after
    # "rabbit." (one continues with the query, the other with <|im_end|>),
    # so the longest common prefix is the exact splice point. Splicing raw
    # token ids here avoids re-tokenizing the filler.
    lcp = 0
    while lcp < min(len(ids_full), len(ids_short)) and ids_full[lcp] == ids_short[lcp]:
        lcp += 1
    assert lcp > 0, "could not locate template splice point"
    assert f" {marker}" in tokenizer.decode(ids_short[:lcp]), "marker text not in shared template prefix"

    used: set[str] = set()
    rng = random.Random(seed + depth)  # same windows for every config
    rows = []
    for _ in range(trials):
        filler = []
        while len(filler) < depth:
            filler.extend(_sentence(tokenizer, rng, marker, used))
        filler = filler[:depth]  # overshoot then trim to exactly `depth`
        rows.append(ids_short[:lcp] + filler + ids_full[lcp:])
    return torch.tensor(rows, dtype=torch.long)


def run_config(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    inputs_by_depth: dict[int, torch.Tensor],
    marker: str,
    max_new_tokens: int,
    device: str,
    config_text,
    spec: FibQuantSpec | None,
) -> dict[int, float]:
    """Run every depth for one cache configuration; return recall per depth."""
    results: dict[int, float] = {}
    marker_l = marker.lower()  # conts are normalized (lowercased); match on the same footing
    for depth in sorted(inputs_by_depth):
        inputs = inputs_by_depth[depth].to(device)
        hits = 0
        # A fresh cache per chunk: generate() appends the continuation into the
        # passed cache, so reusing one cache would feed chunk 2 into chunk 1's
        # already-built sequence.
        with torch.no_grad():
            for chunk in inputs.split(TRIAL_BATCH, dim=0):
                cache = FibQuantCache(config=config_text, spec=spec)
                out = model.generate(
                    input_ids=chunk,
                    attention_mask=torch.ones_like(chunk),  # no padding in batch, but silence pad-token warnings
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    past_key_values=cache,
                )
                conts = [normalize_continuation(tokenizer.decode(row[chunk.shape[-1]:])) for row in out]
                hits += sum(marker_l in c for c in conts)
        results[depth] = hits / inputs.shape[0]
    return results


def main() -> None:
    specs = build_spec_matrix(INCLUDE_BASELINE, BITS, SPEC_PATHS)
    if not specs:
        raise ValueError("set BITS/SPEC_PATHS or INCLUDE_BASELINE so at least one configuration runs")

    if not DEPTHS:
        raise ValueError("DEPTHS must list at least one depth")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = load_model(MODEL_DIR, DEVICE)

    # Filler invariant: the marker must not be reachable from the pools --
    # _sentence re-verifies each generated sentence, but fail at startup if a
    # pool edit introduces it. Single-token-ness is NOT required: multi-token
    # markers are supported (see normalize_continuation).
    validate_marker(MARKER, SENTENCE_POOLS)
    required = required_answer_tokens(tokenizer, [MARKER], slack=ANSWER_BUDGET_SLACK)
    if MAX_NEW_TOKENS is not None and MAX_NEW_TOKENS < required:
        raise ValueError(
            f"MAX_NEW_TOKENS={MAX_NEW_TOKENS} below required {required} for MARKER={MARKER!r}; "
            "raise it or set MAX_NEW_TOKENS=None (auto)"
        )
    max_new = required if MAX_NEW_TOKENS is None else MAX_NEW_TOKENS

    config_text = model.config.text_config
    max_depth = MAX_LENGTH - 16
    print(f"configs: {', '.join(name for name, _ in specs)}")
    print(f"trials={TRIALS} (batched {TRIAL_BATCH}) depths={DEPTHS} (filler capped at {max_depth} tokens) marker={MARKER!r} max_new={max_new}")

    # Build trial inputs once, reused for every config (same contexts, only cache differs).
    inputs_by_depth: dict[int, torch.Tensor] = {}
    for depth in DEPTHS:
        d = min(depth, max_depth)
        inputs_by_depth[d] = build_trials(tokenizer, MARKER, d, TRIALS, SEED)

    table: dict[str, dict[int, float]] = {}
    for name, spec in specs:
        t0 = time.time()
        table[name] = run_config(
            model, tokenizer, inputs_by_depth, MARKER, max_new, DEVICE, config_text, spec
        )
        print(f"[{name}] done in {time.time() - t0:.0f}s")

    print()
    print("depth   " + "".join(f"{name:>10}" for name, _ in specs))
    for depth in sorted(inputs_by_depth):
        row = "".join(f"{100 * table[name][depth]:9.0f}%" for name, _ in specs)
        print(f"{depth:<8}{row}")

    payload = {
        "marker": MARKER,
        "trials": TRIALS,
        "max_new_tokens": max_new,
        "seed": SEED,
        "recall": {name: {str(d): table[name][d] for d in table[name]} for name, _ in specs},
    }
    if OUT:
        out = Path(OUT)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
