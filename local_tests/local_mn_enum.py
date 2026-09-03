"""Local (MPS) reproduction of the multi-needle run, pre-fix prompt bug toggleable.

Constants mirror the reported run: trials=50 (batched 10), depth 2048,
filler cap 3080, needles in [200, depth-200] spacing 200, markers
rabbit/whale/sable/lark/wren, max_new 43, seed 0.
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from fibquant import FibQuantCache, FibQuantSpec
from fibquant.eval_harness import load_model
from fibquant.probes import (
    SENTENCE_POOLS,
    build_spec_matrix,
    marker_hits,
    normalize_continuation,
    required_answer_tokens as _required_answer_tokens,
    unique_filler_sentence,
    validate_markers,
)

IM_END = "<" + "|im_end|" + ">"

MODEL_DIR = "models/Qwen3.5-0.8B"
DEVICE = "mps"

INCLUDE_BASELINE = True
BITS = []
SPEC_PATHS = [
    "models/fibquant/fibquant_d256_k4_N256.pt",
    "models/fibquant/fibquant_d256_k4_N4096.pt",
]

TRIALS = int(sys.argv[sys.argv.index("--trials") + 1]) if "--trials" in sys.argv else 50
TRIAL_BATCH = 10
DEPTH = 2048
MAX_LENGTH = 3096  # filler capped at MAX_LENGTH - 16 = 3080
MAX_NEW_TOKENS = 150  # generous allowance; eos cuts decoding when the model is done
ANSWER_BUDGET_SLACK = 16
MARKERS = ["rabbit", "whale", "sable", "lark", "wren"]
NEEDLE_MIN_POS = 200
NEEDLE_MIN_TAIL = 200
NEEDLE_SPACING = 200
SEED = 0
ASK_QUESTION = "--fixed" in sys.argv  # False => reproduce the pre-fix bug
OUT = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else Path("local_mn_out.json")

SYSTEM_PROMPT = "You answer in one short phrase, nothing else."
QUESTION = " List all five special tokens, separated by commas."
NEEDLE_FRAME = "Special token: {word}."


def _needle_positions(rng, n_needles, lo, hi, spacing):
    span = hi - lo
    for _ in range(200):
        pts = sorted(lo + rng.random() * span for _ in range(n_needles))
        if all(b - a >= spacing for a, b in zip(pts, pts[1:])):
            return pts
    raise ValueError("could not place needles")


def build_trials(tokenizer, markers, depth, trials, seed, ask_question):
    lo, hi = NEEDLE_MIN_POS, depth - NEEDLE_MIN_TAIL
    needle_ids = [tokenizer.encode(" " + NEEDLE_FRAME.format(word=m), add_special_tokens=False) for m in markers]
    question_ids = tokenizer.encode(QUESTION, add_special_tokens=False)
    tmpl = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": ""}],
        tokenize=True, add_generation_prompt=True, enable_thinking=False,
    )["input_ids"]
    im_end = tokenizer.convert_tokens_to_ids(IM_END)
    split = len(tmpl) - 1 - tmpl[::-1].index(im_end)
    prefix, suffix = tmpl[:split], tmpl[split:]

    rng = random.Random(seed + depth)
    used = set()
    avoid = [m.lower() for m in markers]
    rows, positions = [], []
    for _ in range(trials):
        targets = _needle_positions(rng, len(markers), lo, hi, NEEDLE_SPACING)
        positions.append([int(round(t)) for t in targets])
        frags, next_ix, cum = [], 0, 0
        while next_ix < len(targets):
            while cum < targets[next_ix]:
                text, ids = unique_filler_sentence(tokenizer, rng, used, avoid=avoid)
                frags.append(" " + text)
                cum += len(ids)
            frags.append(" " + NEEDLE_FRAME.format(word=markers[next_ix]))
            cum += len(needle_ids[next_ix])
            next_ix += 1
        while cum < depth:
            text, ids = unique_filler_sentence(tokenizer, rng, used, avoid=avoid)
            frags.append(" " + text)
            cum += len(ids)
        content = tokenizer.encode("".join(frags), add_special_tokens=False)[:depth]
        if ask_question:
            rows.append(prefix + content + question_ids + suffix)
        else:
            rows.append(prefix + content + suffix)  # the pre-fix bug
    return torch.tensor(rows, dtype=torch.long), positions


def run_config(model, tokenizer, inputs, markers, max_new_tokens, device, config_text, spec, ask):
    hits_all, per_marker_conts = 0, []
    all_conts = []
    with torch.no_grad():
        for chunk in inputs.split(TRIAL_BATCH, dim=0):
            cache = FibQuantCache(config=config_text, spec=spec)
            out = model.generate(
                input_ids=chunk,
                attention_mask=torch.ones_like(chunk),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                past_key_values=cache,
            )
            for row in out:
                c = normalize_continuation(tokenizer.decode(row[chunk.shape[-1]:]))
                all_conts.append(c)
                if all(marker_hits(c, markers)):
                    hits_all += 1
    return hits_all, all_conts


def main():
    specs = build_spec_matrix(INCLUDE_BASELINE, BITS, SPEC_PATHS)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = load_model(MODEL_DIR, DEVICE)
    validate_markers(MARKERS, SENTENCE_POOLS)
    max_new = _required_answer_tokens(tokenizer, MARKERS, ANSWER_BUDGET_SLACK, frame=NEEDLE_FRAME) if MAX_NEW_TOKENS is None else MAX_NEW_TOKENS
    print(f"ask_question={ASK_QUESTION} trials={TRIALS} max_new={max_new}")

    inputs, positions = build_trials(tokenizer, MARKERS, DEPTH, TRIALS, SEED, ASK_QUESTION)
    inputs = inputs.to(DEVICE)
    # show the tail of trial 0 so the prompt structure is visible
    tail = tokenizer.decode(inputs[0, -60:])
    print("trial0 tail:", repr(tail))

    payload = {"ask_question": ASK_QUESTION, "trials": TRIALS, "recall": {}, "conts": {}}
    for name, spec in specs:
        t0 = time.time()
        hits, conts = run_config(model, tokenizer, inputs, MARKERS, max_new, DEVICE,
                                 model.config.text_config, spec, ASK_QUESTION)
        rec = hits / TRIALS
        payload["recall"][name] = rec
        payload["conts"][name] = conts
        print(f"[{name}] recall={100*rec:.0f}%  ({hits}/{TRIALS}) in {time.time()-t0:.0f}s")
    OUT.write_text(json.dumps(payload, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
