"""Long-range KV-cache recall probe for FibQuant on Qwen3.5-0.8B.

Needle-in-haystack for KV fidelity: for each marker-to-query *depth* and each
cache configuration (bf16 baseline + one row per FibQuant spec), the model
must recall a marker token placed `depth` tokens earlier. The same
deterministic filler windows are reused across configurations, so the cache
is the only variable; greedy decode, success = the marker appears in the
continuation. Batching by depth keeps it cheap: no thinking traces
(enable_thinking=False; the default thinking mode is what makes IFEval take
~5h on MPS) and short generations.

Usage:
    # baseline + b=2 codebook
    .venv/bin/python scripts/key_recall.py --bits 2
    # baseline + b=2/3/4, custom spec paths and run settings
    .venv/bin/python scripts/key_recall.py \
        --spec models/fibquant/fibquant_d256_k4_N4096.pt --bits 4 \
        --trials 20 --depths 256,512,1024,2048,4096 \
        --out results/qwen3.5-0.8b/key-recall-b2b3b4.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

# Make the repo root importable even when run as "python scripts/foo.py".
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fibquant import FibQuantCache, FibQuantSpec, default_spec_path, load_spec  # noqa: E402

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


def _int_list(value: str) -> list[int]:
    return [int(v) for v in value.split(",") if v.strip()]


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
        cache = FibQuantCache(config=config_text, spec=spec)
        with torch.no_grad():
            out = model.generate(
                input_ids=inputs,
                attention_mask=torch.ones_like(inputs),  # no padding in batch, but silence pad-token warnings
                max_new_tokens=max_new_tokens,
                do_sample=False,
                past_key_values=cache,
            )
        conts = [tokenizer.decode(row[inputs.shape[-1]:]) for row in out]
        hits = sum(marker.lower() in c.lower() for c in conts)
        results[depth] = hits / len(conts)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=str, default=None, action="append", help="FibQuant spec checkpoint (repeatable)")
    parser.add_argument("--bits", type=int, default=None, action="append", choices=(2, 3, 4), help="bits/coord; resolves spec via default_spec_path (repeatable)")
    parser.add_argument("--trials", type=int, default=10, help="trials per depth")
    parser.add_argument("--depths", type=str, default="256,512,1024,2048,4096", help="comma-separated marker-to-query token distances")
    parser.add_argument("--max-length", type=int, default=4096, help="cap on total prompt length (filler is truncated)")
    parser.add_argument("--max-new-tokens", type=int, default=12, help="generated tokens per trial")
    parser.add_argument("--marker", type=str, default="rabbit", help="recall marker word")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--model", type=str, default="models/Qwen3.5-0.8B")
    parser.add_argument("--out", type=str, default=None, help="write JSON results here")
    parser.add_argument("--no-baseline", action="store_true", help="skip the fp16 cache row")
    args = parser.parse_args()

    specs: list[tuple[str, FibQuantSpec | None]] = []
    if not args.no_baseline:
        specs.append(("baseline", None))
    for bits in args.bits or []:
        p = str(default_spec_path(d=256, k=4, n_levels=1 << (bits * 4)))
        specs.append((f"fq-b{bits}", FibQuantSpec.from_checkpoint(load_spec(p))))
    for p in args.spec or []:
        spec = FibQuantSpec.from_checkpoint(load_spec(p))
        specs.append((f"fq-N{spec.n_levels}", spec))
    if not specs:
        parser.error("pass --bits, --spec, or remove --no-baseline")

    depths = _int_list(args.depths)
    if not depths:
        parser.error("--depths must list at least one depth")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16).to(args.device).eval()

    if len(tokenizer.encode(args.marker)) != 1 or len(tokenizer.encode(f" {args.marker}")) != 1:
        parser.error(f"--marker {args.marker!r} is not a single token in context; pick another word")
    if args.marker.lower() in CORPUS.lower():
        parser.error("--marker must not appear in the filler corpus")

    config_text = model.config.text_config
    max_depth = args.max_length - 16
    print(f"configs: {', '.join(name for name, _ in specs)}")
    print(f"trials={args.trials} depths={depths} (filler capped at {max_depth} tokens) marker={args.marker!r}")

    # Build trial inputs once, reused for every config (same contexts, only cache differs).
    inputs_by_depth: dict[int, torch.Tensor] = {}
    for depth in depths:
        d = min(depth, max_depth)
        inputs_by_depth[d] = build_trials(tokenizer, args.marker, d, args.trials, args.seed)

    table: dict[str, dict[int, float]] = {}
    for name, spec in specs:
        t0 = time.time()
        table[name] = run_config(
            model, tokenizer, inputs_by_depth, args.marker, args.max_new_tokens, args.device, config_text, spec
        )
        print(f"[{name}] done in {time.time() - t0:.0f}s")

    print()
    print("depth   " + "".join(f"{name:>10}" for name, _ in specs))
    for depth in sorted(inputs_by_depth):
        row = "".join(f"{100 * table[name][depth]:9.0f}%" for name, _ in specs)
        print(f"{depth:<8}{row}")

    payload = {
        "marker": args.marker,
        "trials": args.trials,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "recall": {name: {str(d): table[name][d] for d in table[name]} for name, _ in specs},
    }
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
