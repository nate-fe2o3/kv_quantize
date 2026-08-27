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
~5h on MPS) and short generations.

Configure via the constants below and run the file directly — no CLI
arguments:

    .venv/bin/python scripts/key_recall.py
"""

from __future__ import annotations

import json
import random
import re
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

# --- environment layout --------------------------------------------------
# One switch: False = local repo layout (models/ in the repo, MPS). True =
# the Databricks volume layout (GPU, model + codebook checkpoints on a volume).
DATABRICKS = False

# --- run configuration ---------------------------------------------------
MODEL_DIR = (
    "/Volumes/security_engineering/nbutton/q34b/models/Qwen3.5-0.8B/"
    if DATABRICKS
    else "models/Qwen3.5-0.8B"
)
DEVICE = "cuda" if DATABRICKS else "mps"

# Cache configurations to compare, one row each in the output table.
# INCLUDE_BASELINE adds the fp16 (uncompressed) row; SPEC_PATHS are explicit
# spec checkpoints; BITS entries resolve via FibQuantSpec.from_bits (d=256,
# k=4), which uses a *repo-relative* default path -- so leave BITS empty on
# Databricks (checkpoints are on the volume) and list them in SPEC_PATHS.
INCLUDE_BASELINE = True
BITS = []  # e.g. [2], [2, 3, 4] -- use [] when SPEC_PATHS is set
SPEC_PATHS = (
    [
        "/Volumes/security_engineering/nbutton/q34b/models/fibquant/fibquant_d256_k4_N256.pt",
        "/Volumes/security_engineering/nbutton/q34b/models/fibquant/fibquant_d256_k4_N4096.pt",
    ]
    if DATABRICKS
    else [
        "models/fibquant/fibquant_d256_k4_N256.pt",
        "models/fibquant/fibquant_d256_k4_N4096.pt",
    ]
)

TRIALS = 50  # trials per depth
# Trials are batched into generate() calls of this size; peak activation
# memory scales with batch * seq^2 (attention scores), so 30 x 4096 in one
# call OOMs. 10 matches the memory of the earlier 10-trial runs; lower to 5
# if a long depth still OOMs, raise if the GPU has headroom.
TRIAL_BATCH = 10
DEPTHS = [1024, 2048, 4096]  # marker-to-query token distances
MAX_LENGTH = 4096  # cap on total prompt length (filler is truncated)
MAX_NEW_TOKENS = 12  # generated tokens per trial
MARKER = "rabbit"  # recall marker word
SEED = 0
OUT = None  # write JSON results here, or None to only print the table

SYSTEM_PROMPT = "You answer in one short phrase, nothing else."

# --- filler: template-generated unique sentences -------------------------
# Every filler sentence is one template + iid slot words (no config, no files,
# no repeating passages). Pool products make exact collisions astronomically
# unlikely; a per-build `used` set makes it impossible regardless. The marker
# word and anything containing it are absent from the pools by construction,
# and the generated sentence is decode-verified before use.
SENTENCE_TEMPLATES = [
    "The {animal} {verb} through the {place} {when}.",
    "A {animal} {verb} {adv} near the {place}.",
    "The {adj} {animal} {verb} {adv} {when}.",
    "The {occupation} {verb} {adv} beside a {noun}.",
    "A {occupation} {verb} through the {place} {when}.",
    "The {adj} {occupation} examined a {noun} {when}.",
    "The {occupation} carried a {noun} across the {place}.",
    "A {noun} {verb} {adv} on the {place} {when}.",
    "The {adj} {noun} was {adv} visible {when}.",
    "The {animal} watched the {adj} {noun} {when}.",
    "A {noun} hung {adv} over the {place}.",
    "The {adj} {occupation} saw a {noun} at the {place}.",
    "Every {occupation} at the {place} owns a {noun}.",
    "The {animal} hid {adv} behind the {noun} {when}.",
]

SENTENCE_POOLS = {
    "animal": ["dog", "cat", "fox", "hawk", "heron", "otter", "badger", "lynx",
               "eel", "newt", "crow", "seal", "wolf", "deer", "moose", "bison",
               "gecko", "crane", "swan", "mole"],
    "verb": ["wandered", "crept", "peered", "dashed", "drifted", "clambered",
             "lingered", "scurried", "ambled", "soared", "trudged", "bounded",
             "veered", "nestled", "vanished", "circled", "rested", "paced"],
    "adv": ["quietly", "slowly", "steadily", "briefly", "softly", "gradually",
            "warily", "eagerly", "haphazardly", "carefully", "awkwardly"],
    "place": ["meadow", "harbor", "orchard", "courtyard", "thicket", "valley",
              "station", "street", "rooftop", "corridor", "garden", "plateau",
              "market", "tunnel", "attic", "promenade", "forest", "bridge"],
    "noun": ["lantern", "ledger", "crate", "saddle", "hatbox", "map", "basket",
             "kettle", "telescope", "compass", "bundle", "barrel", "mirror",
             "whistle"],
    "occupation": ["librarian", "baker", "blacksmith", "cartographer",
                   "apothecary", "cartwright", "beekeeper", "clockmaker",
                   "ferryman", "mason", "weaver", "oarsman", "lamplighter"],
    "adj": ["weathered", "mottled", "sturdy", "curious", "weary", "faded",
            "coiled", "glazed", "hollow", "bronze", "mossy", "threadbare"],
    "when": ["at dusk", "before dawn", "in the rain", "under a thin moon",
             "amid fog", "past midnight", "at first light", "in a stiff wind"],
}

_SLOT_RE = re.compile(r"\{(\w+)\}")


def _sentence(tokenizer: AutoTokenizer, rng: random.Random, marker: str, used: set[str]) -> list[int]:
    """One fresh, unique, marker-free filler sentence as token ids.

    Encoded with a leading space so the sentence-initial word is the *mid-text*
    token variant (" A" != "A" in this BPE), keeping the concatenated filler
    properly spaced at the token level.
    """
    for _ in range(100):
        template = rng.choice(SENTENCE_TEMPLATES)
        text = template.format(**{s: rng.choice(SENTENCE_POOLS[s]) for s in _SLOT_RE.findall(template)})
        if text in used:
            continue
        ids = tokenizer.encode(" " + text, add_special_tokens=False)
        if marker.lower() in tokenizer.decode(ids).lower():
            continue
        used.add(text)
        return ids
    raise ValueError("could not generate a unique marker-free filler sentence")


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
    # Filler invariant (replaces the old CORPUS check): the marker word must not
    # be reachable from the pools -- _sentence re-verifies each generated
    # sentence, but fail at startup if a pool edit introduces it.
    marker_l = MARKER.lower()
    for _slot, _words in SENTENCE_POOLS.items():
        for _w in _words:
            if marker_l in _w.lower():
                raise ValueError(f"MARKER {MARKER!r} appears in filler pool '{_slot}' as {_w!r}")

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
