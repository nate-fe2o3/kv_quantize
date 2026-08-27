"""Long-range KV-cache recall probe for FibQuant on Qwen3.5-0.8B.

Needle-in-haystack for KV fidelity: for each marker-to-query *depth* and each
cache configuration (bf16 baseline + one row per FibQuant spec), the model
must recall a marker token placed `depth` tokens earlier. The same
deterministic filler windows are reused across configurations, so the cache
is the only variable; greedy decode, success = the marker appears in the
continuation. Batching by depth keeps it cheap: no thinking traces
(enable_thinking=False; the default thinking mode is what makes IFEval take
~5h on MPS) and short generations.

Configure via the constants below and run the file directly — no CLI
arguments:

    .venv/bin/python scripts/key_recall.py
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

# Make the repo root importable when run as "python scripts/foo.py". In a
# Databricks notebook __file__ is undefined (NameError); the notebook's
# directory is already on sys.path there, so skip the insert.
if "__file__" in globals():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fibquant import FibQuantCache, FibQuantSpec  # noqa: E402
from fibquant.eval_harness import load_model  # noqa: E402

# --- run configuration ---------------------------------------------------
MODEL_DIR = "models/Qwen3.5-0.8B"  # Databricks: "/Volumes/security_engineering/nbutton/q34b/models/Qwen3.5-0.8B/"
DEVICE = "mps"  # Databricks: "cuda"

# Cache configurations to compare, one row each in the output table.
# INCLUDE_BASELINE adds the fp16 (uncompressed) row; SPEC_PATHS are explicit
# spec checkpoints; BITS entries resolve via FibQuantSpec.from_bits (d=256,
# k=4), which uses a *repo-relative* default path -- so on Databricks leave
# BITS empty and list the volume checkpoints in SPEC_PATHS.
INCLUDE_BASELINE = True
BITS = []  # e.g. [2], [2, 3, 4] -- use [] when SPEC_PATHS is set
SPEC_PATHS = [
    "models/fibquant/fibquant_d256_k4_N256.pt",
    "models/fibquant/fibquant_d256_k4_N4096.pt",
]  # Databricks: ["/Volumes/security_engineering/nbutton/q34b/models/fibquant/fibquant_d256_k4_N256.pt", "..."]

TRIALS = 10  # trials per depth
# Trials are batched into generate() calls of this size; peak activation
# memory scales with batch * seq^2 (attention scores), so 30 x 4096 in one
# call OOMs. 10 matches the memory of the earlier 10-trial runs; lower to 5
# if a long depth still OOMs, raise if the GPU has headroom.
TRIAL_BATCH = 10
DEPTHS = [256, 512, 1024, 2048, 4096]  # marker-to-query token distances
MAX_LENGTH = 4096  # cap on total prompt length (filler is truncated)
MAX_NEW_TOKENS = 12  # generated tokens per trial
MARKER = "rabbit"  # recall marker word
SEED = 0
OUT = None  # write JSON results here, or None to only print the table

SYSTEM_PROMPT = "You answer in one short phrase, nothing else."

# Varied prose, tokenized once and sliced as filler. Marker word must not appear.
CORPUS = """The cat sat on the mat and watched the birds beyond the window. A dog
barked at the moon while seven stars shone over the quiet hills. Rain fell
softly on the rooftops of the old town, and the streets glistened under the
lamps. Farmers loaded their carts at dawn, heading toward the distant market.
The river carried leaves downstream past the mill and the weathered bridge.
Children played in the orchard, chasing butterflies through the tall grass.
Cooks stirred a large pot of stew in the kitchen, where the fire crackled
warmly. Trains crossed the plains under a pale blue sky, carrying letters and
parcels between the cities. Sailors mended their nets on the wooden dock while
gulls circled above the harbor. The librarian shelved books in the dim aisle,
pausing to dust the spines of the old volumes. Winter came early that year,
frosting the windows and silencing the gardens. The baker opened his shop
before sunrise, filling the street with the smell of fresh bread. A musician
played a slow tune in the square, and passersby paused to listen. The
astronomer recorded the brightness of the comet from the hilltop observatory.
The carpenter built a sturdy chest, fitting each joint with care. Green fields
stretched to the horizon, dotted with sheep and lone oak trees. The ferry
crossed the lake every hour, carrying passengers and bicycles. Smoke rose from
the chimneys of the village, curling into the cold morning air."""


def _repeat(seq, n):
    return [seq[i % len(seq)] for i in range(n)]


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
    assert f" {marker}" in tokenizer.decode(ids_short[:lcp]), "marker token not in shared template prefix"

    corpus_ids = tokenizer.encode(CORPUS, add_special_tokens=False)
    n = len(corpus_ids)
    rng = random.Random(seed + depth)  # same windows for every config
    rows = []
    for _ in range(trials):
        start = rng.randrange(n)
        filler = _repeat(corpus_ids[start:start + depth], depth) if depth else []
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
                conts = [tokenizer.decode(row[chunk.shape[-1]:]) for row in out]
                hits += sum(marker.lower() in c.lower() for c in conts)
        results[depth] = hits / inputs.shape[0]
    return results


def main() -> None:
    specs: list[tuple[str, FibQuantSpec | None]] = []
    if INCLUDE_BASELINE:
        specs.append(("baseline", None))
    for bits in BITS:
        specs.append((f"fq-b{bits}", FibQuantSpec.from_bits(d=256, k=4, bits=bits)))
    for p in SPEC_PATHS:
        spec = FibQuantSpec.from_path(p)
        specs.append((f"fq-N{spec.n_levels}", spec))
    if not specs:
        raise ValueError("set BITS/SPEC_PATHS or INCLUDE_BASELINE so at least one configuration runs")

    if not DEPTHS:
        raise ValueError("DEPTHS must list at least one depth")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = load_model(MODEL_DIR, DEVICE)

    if len(tokenizer.encode(MARKER)) != 1 or len(tokenizer.encode(f" {MARKER}")) != 1:
        raise ValueError(f"MARKER {MARKER!r} is not a single token in context; pick another word")
    if MARKER.lower() in CORPUS.lower():
        raise ValueError("MARKER must not appear in the filler corpus")

    config_text = model.config.text_config
    max_depth = MAX_LENGTH - 16
    print(f"configs: {', '.join(name for name, _ in specs)}")
    print(f"trials={TRIALS} (batched {TRIAL_BATCH}) depths={DEPTHS} (filler capped at {max_depth} tokens) marker={MARKER!r}")

    # Build trial inputs once, reused for every config (same contexts, only cache differs).
    inputs_by_depth: dict[int, torch.Tensor] = {}
    for depth in DEPTHS:
        d = min(depth, max_depth)
        inputs_by_depth[d] = build_trials(tokenizer, MARKER, d, TRIALS, SEED)

    table: dict[str, dict[int, float]] = {}
    for name, spec in specs:
        t0 = time.time()
        table[name] = run_config(
            model, tokenizer, inputs_by_depth, MARKER, MAX_NEW_TOKENS, DEVICE, config_text, spec
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
        "max_new_tokens": MAX_NEW_TOKENS,
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
