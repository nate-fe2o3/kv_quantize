"""Option-2 check: k sampled continuations per trial, marker union scoring.

Same trials/prompts as the original run (original QUESTION, brevity system
prompt); do_sample=True, temperature 1.0, k samples per trial via row
expansion. A marker counts as retrieved if ANY sample contains it.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

from fibquant import FibQuantCache
from fibquant.eval_harness import load_model
from fibquant.probes import SENTENCE_POOLS, build_spec_matrix, marker_hits, normalize_continuation, validate_markers
import local_mn as M

K = 5
TRIALS = int(sys.argv[sys.argv.index("--trials") + 1]) if "--trials" in sys.argv else 25
TEMPERATURE = 1.0
SEED_SAMPLER = 1234
OUT = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else Path("local_mn_union.json")


def main():
    specs = build_spec_matrix(M.INCLUDE_BASELINE, M.BITS, M.SPEC_PATHS)
    tokenizer = M.AutoTokenizer.from_pretrained(M.MODEL_DIR)
    model = load_model(M.MODEL_DIR, M.DEVICE)
    validate_markers(M.MARKERS, SENTENCE_POOLS)
    max_new = 150  # generous allowance; eos cuts decoding when the model is done

    inputs, _ = M.build_trials(tokenizer, M.MARKERS, M.DEPTH, TRIALS, M.SEED, ask_question=True)
    inputs = inputs[:TRIALS].to(M.DEVICE)

    payload = {"k": K, "trials": TRIALS, "per_marker_union": {}, "all5_union": {}, "mean_markers": {}}
    for name, spec in specs:
        t0 = time.time()
        union_hits = {m: 0 for m in M.MARKERS}
        all5 = 0
        tot_markers = 0
        with torch.no_grad():
            for chunk in inputs.split(max(1, M.TRIAL_BATCH // K), dim=0):  # -> <=10 rows after xK expansion
                exp = chunk.repeat_interleave(K, dim=0)  # (rows*K, seq)
                cache = FibQuantCache(config=model.config.text_config, spec=spec)
                torch.manual_seed(SEED_SAMPLER)
                out = model.generate(
                    input_ids=exp,
                    attention_mask=torch.ones_like(exp),
                    max_new_tokens=max_new,
                    do_sample=True,
                    temperature=TEMPERATURE,
                    eos_token_id=tokenizer.eos_token_id,
                    past_key_values=cache,
                )
                conts = [normalize_continuation(tokenizer.decode(row[exp.shape[-1]:])) for row in out]
                n = chunk.shape[0]
                for i in range(n):
                    got = {m: any(marker_hits(conts[i * K + j], [m])[0] for j in range(K)) for m in M.MARKERS}
                    tot_markers += sum(got.values())
                    if all(got.values()):
                        all5 += 1
                    for m, g in got.items():
                        if g:
                            union_hits[m] += 1
        payload["per_marker_union"][name] = {m: union_hits[m] / TRIALS for m in M.MARKERS}
        payload["all5_union"][name] = all5 / TRIALS
        payload["mean_markers"][name] = tot_markers / TRIALS
        print(f"[{name}] all5(union)={100*all5/TRIALS:.0f}% mean_markers={tot_markers/TRIALS:.2f}/5 "
              f"per-marker={{{', '.join(f'{m}:{union_hits[m]}' for m in M.MARKERS)}}} in {time.time()-t0:.0f}s")
    OUT.write_text(json.dumps(payload, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
