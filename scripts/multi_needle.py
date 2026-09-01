"""Buried multi-needle KV recall probe for FibQuant on Qwen3.5-0.8B.

Harder sibling of scripts/key_recall.py. Two changes make it harder while
keeping the wall time identical (the cost is context-length x trials x
configs; needle layout is free):

  1. Buried positions: each needle is placed at a *random absolute position*
     in [NEEDLE_MIN_POS, depth - NEEDLE_MIN_TAIL] (uniform, min spacing),
     instead of immediately after the system prompt. The model must
     content-match mid-context with filler on both sides; key_recall's
     near-start placement let it take a position shortcut.
  2. Multiple needles: with N needles, success = ALL markers appear in the
     continuation, so miss rates multiply (a 5% per-needle tax becomes ~23%
     at 5 needles).

Needle positions are re-sampled per trial (never fixed per depth): recall at a
depth is then the expectation over both placements and filler — the quantity
that transfers to the long-horizon question — and per-trial positions are
recorded in the OUT payload, so placement effects can be analyzed post-hoc
(e.g. recall vs distance-to-query) without another run.

Everything else matches key_recall: deterministic seeded filler reused
verbatim across configurations (paired comparisons), greedy decode, one fresh
FibQuantCache per trial batch, no CLI args -- constants at the top.

Databricks-only (models and codebook checkpoints live on the UC volume); run
as a notebook or `python scripts/multi_needle.py` on the cluster.
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
INCLUDE_BASELINE = True
BITS = []  # e.g. [2] -- use [] when SPEC_PATHS is set (see key_recall.py)
SPEC_PATHS = [
    "/Volumes/security_engineering/nbutton/q34b/models/fibquant/fibquant_d256_k4_N256.pt",
    "/Volumes/security_engineering/nbutton/q34b/models/fibquant/fibquant_d256_k4_N4096.pt",
]

TRIALS = 50  # trials per depth
TRIAL_BATCH = 4  # peak activation memory scales batch * seq^2; 4 at 16k
DEPTHS = [4096, 8192, 16384]  # total tokens before the query (filler + needles)
MAX_LENGTH = 16384  # cap on filler; DEPTHS entries are clamped like key_recall
# Max generated tokens per trial. None = auto: the token cost of the verbose
# marker listing + ANSWER_BUDGET_SLACK (see required_answer_tokens). With
# multi-token markers a fixed small budget can cut the continuation off
# mid-list, and a cut tail counts as forgotten markers -- a silent recall
# underestimate.
MAX_NEW_TOKENS = None
ANSWER_BUDGET_SLACK = 16  # framing/prefix allowance beyond the marker list
MARKERS = ["rabbit", "whale", "sable", "lark", "wren"]  # one marker per needle (len(MARKERS) needles);
# Markers may be multi-token words or short phrases: they are spliced in as
# " " + frame and matched against the whitespace-normalized continuation.
# Only empty markers, self-overlapping markers, or text reachable from the
# filler pools fail at startup (see validate_markers). Prefer words over long
# phrases: a phrase can be glued mid-word by the tokenizer ("blue whale" ->
# "bluewhale") and then score as a miss even when the model recalled it.
# Needle geometry (tokens, relative to the start of the user message):
NEEDLE_MIN_POS = 1024  # at least this much filler before the first needle
NEEDLE_MIN_TAIL = 1024  # at least this much filler after the last needle
NEEDLE_SPACING = 256  # minimum token gap between consecutive needles (adapted
# for 5 needles at the shortest depth; raise it only if DEPTHS are all long)
SEED = 0
OUT = None  # JSON results path, or None to only print tables

SYSTEM_PROMPT = "You answer in one short phrase, nothing else."
QUESTION = " What were the special tokens? Answer:"
NEEDLE_FRAME = "Special token: {word}."

# --- filler: template-generated unique sentences -------------------------
# Templates, word pools, and the generator live in fibquant.probes (shared
# with scripts/key_recall.py and scripts/logit_kl.py).


def required_answer_tokens(tokenizer: AutoTokenizer, markers: list[str], slack: int = ANSWER_BUDGET_SLACK) -> int:
    """Thin wrapper over fibquant.probes.required_answer_tokens binding this
    script's own ANSWER_BUDGET_SLACK default and NEEDLE_FRAME."""
    return _required_answer_tokens(tokenizer, markers, slack, frame=NEEDLE_FRAME)


def _sentence(
    tokenizer: AutoTokenizer,
    rng: random.Random,
    avoid: list[str],
    used: set[str],
) -> tuple[str, list[int]]:
    """One fresh, unique filler sentence as (text, token ids); avoids every marker word."""
    return unique_filler_sentence(tokenizer, rng, used, avoid=avoid)


def _needle_positions(
    rng: random.Random,
    n_needles: int,
    lo: float,
    hi: float,
    spacing: float,
) -> list[float]:
    """n_needles sorted uniform positions in [lo, hi] with min spacing."""
    span = hi - lo
    for _ in range(200):
        pts = sorted(lo + rng.random() * span for _ in range(n_needles))
        if all(b - a >= spacing for a, b in zip(pts, pts[1:])):
            return pts
    raise ValueError(f"could not place {n_needles} needles in [{lo:.0f}, {hi:.0f}] with spacing {spacing}")


def build_trials(
    tokenizer: AutoTokenizer,
    markers: list[str],
    depth: int,
    trials: int,
    seed: int,
) -> tuple[torch.Tensor, list[list[int]]]:
    """Build (trials, seq) input_ids for one depth; also per-trial needle positions.

    Layout per trial: chat-template prefix + filler, with each needle frame
    ("Special token: <word>.") spliced in at its sampled target position
    (rounded up to the next sentence boundary), then the query suffix. The
    content string is built from fragments, tokenized once canonically, and
    trimmed to exactly `depth` tokens before the query is appended -- needles
    are >= NEEDLE_MIN_TAIL tokens from the end, so trimming only ever cuts tail
    filler.

    Positions are drawn fresh per trial (the estimate at a depth is then the
    expectation over placements and filler); the returned positions are the
    sampled targets in token units, for post-hoc analysis.
    """
    lo, hi = NEEDLE_MIN_POS, depth - NEEDLE_MIN_TAIL
    if hi - lo < (len(markers) - 1) * NEEDLE_SPACING:
        raise ValueError(
            f"depth {depth} too short for {len(markers)} needles: span {hi - lo:.0f} < "
            f"{(len(markers) - 1) * NEEDLE_SPACING}; lower NEEDLE_SPACING/NEEDLE_MIN_POS or increase depth"
        )

    needle_ids = [tokenizer.encode(" " + NEEDLE_FRAME.format(word=m), add_special_tokens=False) for m in markers]
    question_ids = tokenizer.encode(QUESTION, add_special_tokens=False)

    # Prefix/suffix around the (variable-length) user content, from the
    # empty-content template. Positionally valid for any content: the user
    # turn renders "user<\\n>" + content + "<|im_end|>...assistant<\\n>".
    tmpl = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": ""}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )["input_ids"]
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    split = len(tmpl) - 1 - tmpl[::-1].index(im_end)  # last <|im_end|> = end of user turn
    prefix, suffix = tmpl[:split], tmpl[split:]

    rng = random.Random(seed + depth)  # same windows for every config
    used: set[str] = set()
    avoid = [m.lower() for m in markers]
    rows = []
    positions: list[list[int]] = []
    for _ in range(trials):
        targets = _needle_positions(rng, len(markers), lo, hi, NEEDLE_SPACING)
        positions.append([int(round(t)) for t in targets])
        frags: list[str] = []
        next_ix = 0
        cum = 0
        while next_ix < len(targets):
            while cum < targets[next_ix]:
                text, ids = _sentence(tokenizer, rng, avoid, used)
                frags.append(" " + text)
                cum += len(ids)
            frags.append(" " + NEEDLE_FRAME.format(word=markers[next_ix]))
            cum += len(needle_ids[next_ix])
            next_ix += 1
        while cum < depth:
            text, ids = _sentence(tokenizer, rng, avoid, used)
            frags.append(" " + text)
            cum += len(ids)
        content = tokenizer.encode("".join(frags), add_special_tokens=False)[:depth]
        rows.append(prefix + content + question_ids + suffix)

    # Sanity on the first row only: the prompt must actually contain every
    # marker and the query in decoded form (else tokenization ate something).
    # Content length can differ from `depth` by ~1 token (isolated vs
    # in-context tokenization of the first unit); all rows share the same
    # offset, so the length assert is a range check, not an equality.
    first = rows[0]
    content_len = len(first) - len(prefix) - len(question_ids) - len(suffix)
    assert depth - 2 <= content_len <= depth + 1, f"content length {content_len} off depth {depth}"
    dec = normalize_continuation(tokenizer.decode(first))
    for m in markers:
        assert m.lower() in dec, f"marker {m!r} missing from trial 0; check MARKERS/tokenizer"
    assert normalize_continuation(QUESTION) in dec, "query text missing from trial 0"

    return torch.tensor(rows, dtype=torch.long), positions


def run_config(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    inputs_by_depth: dict[int, torch.Tensor],
    markers: list[str],
    max_new_tokens: int,
    device: str,
    config_text,
    spec: FibQuantSpec | None,
) -> tuple[dict[int, float], dict[str, dict[int, float]]]:
    """Run every depth for one cache configuration.

    Returns (all-recall per depth, per-marker recall per depth), so a partial
    failure (2 of 3 markers reproduced) is visible in the diagnostics.
    """
    all_recall: dict[int, float] = {}
    per_marker: dict[str, dict[int, float]] = {m: {} for m in markers}
    for depth in sorted(inputs_by_depth):
        inputs = inputs_by_depth[depth].to(device)
        hits_all = 0
        hits_marker = {m: 0 for m in markers}
        # A fresh cache per chunk: generate() appends the continuation into the
        # passed cache, so reusing one cache would mix chunks.
        with torch.no_grad():
            for chunk in inputs.split(TRIAL_BATCH, dim=0):
                cache = FibQuantCache(config=config_text, spec=spec)
                out = model.generate(
                    input_ids=chunk,
                    attention_mask=torch.ones_like(chunk),
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    past_key_values=cache,
                )
                conts = [normalize_continuation(tokenizer.decode(row[chunk.shape[-1]:])) for row in out]
                for c in conts:
                    got = [m.lower() in c for m in markers]
                    if all(got):
                        hits_all += 1
                    for m, g in zip(markers, got):
                        if g:
                            hits_marker[m] += 1
        all_recall[depth] = hits_all / inputs.shape[0]
        for m in markers:
            per_marker[m][depth] = hits_marker[m] / inputs.shape[0]
    return all_recall, per_marker


def main() -> None:
    specs = build_spec_matrix(INCLUDE_BASELINE, BITS, SPEC_PATHS)
    if not specs:
        raise ValueError("set BITS/SPEC_PATHS or INCLUDE_BASELINE so at least one configuration runs")
    if not DEPTHS:
        raise ValueError("DEPTHS must list at least one depth")
    if not MARKERS:
        raise ValueError("MARKERS must list at least one marker word")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = load_model(MODEL_DIR, DEVICE)

    # Marker invariants: non-empty, mutually non-overlapping, and unreachable
    # from the filler pools. Single-token-ness is NOT required: multi-token
    # markers are supported (see normalize_continuation).
    validate_markers(MARKERS, SENTENCE_POOLS)
    required = required_answer_tokens(tokenizer, MARKERS)
    if MAX_NEW_TOKENS is not None and MAX_NEW_TOKENS < required:
        raise ValueError(
            f"MAX_NEW_TOKENS={MAX_NEW_TOKENS} below required {required} for MARKERS={MARKERS}; "
            "raise it or set MAX_NEW_TOKENS=None (auto)"
        )
    max_new = required if MAX_NEW_TOKENS is None else MAX_NEW_TOKENS

    config_text = model.config.text_config
    max_depth = MAX_LENGTH - 16
    print(f"configs: {', '.join(name for name, _ in specs)}")
    print(
        f"trials={TRIALS} (batched {TRIAL_BATCH}) depths={DEPTHS} (filler capped at {max_depth} tokens) "
        f"markers={MARKERS} max_new={max_new}"
    )
    print(
        f"needles: buried in [{NEEDLE_MIN_POS}, depth-{NEEDLE_MIN_TAIL}], "
        f"min spacing {NEEDLE_SPACING}; success = all {len(MARKERS)} markers in the continuation"
    )

    inputs_by_depth: dict[int, torch.Tensor] = {}
    positions_by_depth: dict[int, list[list[int]]] = {}
    for depth in DEPTHS:
        d = min(depth, max_depth)
        inputs_by_depth[d], positions_by_depth[d] = build_trials(tokenizer, MARKERS, d, TRIALS, SEED)

    table: dict[str, dict[int, float]] = {}
    per_marker_table: dict[str, dict[str, dict[int, float]]] = {}
    for name, spec in specs:
        t0 = time.time()
        all_recall, per_marker = run_config(
            model, tokenizer, inputs_by_depth, MARKERS, max_new, DEVICE, config_text, spec
        )
        table[name] = all_recall
        per_marker_table[name] = per_marker
        print(f"[{name}] done in {time.time() - t0:.0f}s")

    print()
    print("depth   " + "".join(f"{name:>10}" for name, _ in specs))
    for depth in sorted(inputs_by_depth):
        row = "".join(f"{100 * table[name][depth]:9.0f}%" for name, _ in specs)
        print(f"{depth:<8}{row}")

    payload = {
        "markers": MARKERS,
        "trials": TRIALS,
        "max_new_tokens": max_new,
        "needle_min_pos": NEEDLE_MIN_POS,
        "needle_min_tail": NEEDLE_MIN_TAIL,
        "needle_spacing": NEEDLE_SPACING,
        "seed": SEED,
        "positions": {str(d): positions_by_depth[d] for d in sorted(positions_by_depth)},
        "recall_all": {name: {str(d): table[name][d] for d in table[name]} for name, _ in specs},
        "recall_per_marker": {
            name: {m: {str(d): v for d, v in pm.items()} for m, pm in per_marker_table[name].items()}
            for name, _ in specs
        },
    }
    if OUT:
        out = Path(OUT)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
