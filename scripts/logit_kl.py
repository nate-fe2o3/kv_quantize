"""Long-context logit-level fidelity: KL and top-1 agreement vs fp16 KV.

For each context length and each FibQuant spec, forward the same seeded
filler documents through the model twice -- once with the fp16 DynamicCache
(reference) and once with the FibQuant cache -- and compare the per-position
next-token distributions:

  - mean / p95 / max KL(pi_reference || pi_quantized) over positions
    [WARMUP, length)  (nats per token)
  - top-1 agreement: fraction of positions where the greedy prediction is
    identical in both passes

Interpretation: the "practically lossless" heuristic thresholds below -- mean
KL <= ~0.02 nats/token with top-1 agreement >= 99% -- mean the quantization
error is smaller than the model's own next-token noise; a task-level probe
(key_recall, multi_needle) then measures whether residual drift matters. This
probe is the one that catches *distribution* drift even when retrieval tasks
still pass, and it costs seconds per forward.

Databricks-only (models and codebook checkpoints live on the UC volume); run
as a notebook or `python scripts/logit_kl.py` on the cluster.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, DynamicCache

from fibquant import FibQuantCache, FibQuantSpec
from fibquant.eval_harness import load_model
from fibquant.probes import build_spec_matrix, unique_filler_sentence

# --- paths (Databricks volume layout) ------------------------------------
MODEL_DIR = "/Volumes/security_engineering/nbutton/q34b/models/Qwen3.5-0.8B/"
DEVICE = "cuda"

# --- run configuration ---------------------------------------------------
# Quantized configurations to compare against the fp16 reference.
BITS = []  # e.g. [2] -- use [] when SPEC_PATHS is set
SPEC_PATHS = [
    "/Volumes/security_engineering/nbutton/q34b/models/fibquant/fibquant_d256_k4_N256.pt",
    "/Volumes/security_engineering/nbutton/q34b/models/fibquant/fibquant_d256_k4_N4096.pt",
]

CONTEXT_LENGTHS = [4096, 8192, 16384]  # token positions per document (predictions up to length)
DOCS = 4  # seeded filler documents; same docs reused at every length
CHUNK = 1024  # forward-chunk size; transient logits tensors scale as CHUNK x vocab
WARMUP = 512  # skip the first positions (attention-sink region, not long-range)
SEED = 0
OUT = None  # JSON results path, or None to only print tables

# "Practically lossless" gates (documented heuristic, not a spec):
KL_MEAN_PASS = 0.02  # nats/token
KL_P95_PASS = 0.12
TOP1_PASS = 0.99

# --- filler: template-generated unique sentences -------------------------
# Templates, word pools, and the generator live in fibquant.probes (shared
# with scripts/key_recall.py and scripts/multi_needle.py).


def _sentence(tokenizer: AutoTokenizer, rng: random.Random, used: set[str]) -> list[int]:
    """One fresh, unique filler sentence as token ids (no marker constraint)."""
    _, ids = unique_filler_sentence(tokenizer, rng, used)
    return ids


def build_doc(tokenizer: AutoTokenizer, n_tokens: int, seed: int) -> torch.Tensor:
    """One seeded filler document of `n_tokens` ids (sentence-aligned start)."""
    rng = random.Random(seed)
    used: set[str] = set()
    ids: list[int] = []
    while len(ids) < n_tokens:
        ids.extend(_sentence(tokenizer, rng, used))
    return torch.tensor(ids[:n_tokens], dtype=torch.long)


def _run_length(
    model: torch.nn.Module,
    config_text,
    doc_ids: torch.Tensor,
    length: int,
    specs: list[tuple[str, FibQuantSpec]],
    device: str,
) -> dict[str, dict]:
    """One document, one context length: compare all specs to the fp16 pass.

    Both passes are chunked identically. The reference pass runs once; each
    quantized pass advances its own cache in lockstep, so result pairs share
    exact input positions.
    """
    inputs = doc_ids[: length + 1].unsqueeze(0).to(device)  # (1, length+1); position i predicts token i+1
    cache_b = DynamicCache()
    fq_caches = {name: FibQuantCache(config=config_text, spec=spec) for name, spec in specs}
    stats = {name: {"sum_kl": 0.0, "kl_list": [], "disagree": 0, "n": 0} for name, _ in specs}

    # The attention_mask passed alongside a persistent past_key_values must
    # cover the whole sequence the cache has seen so far (past + current
    # chunk), not just the current chunk: Transformers' causal-mask builder
    # derives kv_length from past_key_values.get_seq_length() + the new
    # chunk and right-pads/right-slices a shorter mask, which would silently
    # mark the *earlier* real tokens as padding instead of the (nonexistent)
    # trailing ones. There is no padding here (batch size 1, one dense
    # document), so a full ones mask sliced to the running length is exact.
    full_mask = torch.ones_like(inputs)
    with torch.no_grad():
        for start in range(0, length, CHUNK):
            chunk = inputs[:, start : start + CHUNK]
            am_chunk = full_mask[:, : start + chunk.shape[1]]
            out_b = model(chunk, attention_mask=am_chunk, past_key_values=cache_b)
            la = torch.log_softmax(out_b.logits.float(), dim=-1)  # (1, C, V)
            pa = la.exp()
            for name, _ in specs:
                out_f = model(chunk, attention_mask=am_chunk, past_key_values=fq_caches[name])
                lb = torch.log_softmax(out_f.logits.float(), dim=-1)  # (1, C, V)
                kl = (pa * (la - lb)).sum(dim=-1)[0]  # (C,)
                for p, k in enumerate(kl.tolist()):
                    pos = start + p
                    if pos < WARMUP or pos >= length:
                        continue
                    s = stats[name]
                    s["sum_kl"] += k
                    s["kl_list"].append(k)
                    s["disagree"] += int(la[0, p].argmax() != lb[0, p].argmax())
                    s["n"] += 1
    return stats


def _summarize(stats: dict) -> dict:
    kl = sorted(stats["kl_list"]) or [0.0]
    n = max(stats["n"], 1)
    return {
        "mean_kl": stats["sum_kl"] / n,
        "p95_kl": kl[min(len(kl) - 1, int((len(kl) - 1) * 0.95))] if kl else 0.0,
        "max_kl": kl[-1],
        "top1_agree": 1.0 - stats["disagree"] / n,
        "n": stats["n"],
    }


def main() -> None:
    specs = build_spec_matrix(include_baseline=False, bits=BITS, spec_paths=SPEC_PATHS)
    if not specs:
        raise ValueError("set BITS/SPEC_PATHS so at least one quantized configuration runs")
    if not CONTEXT_LENGTHS:
        raise ValueError("CONTEXT_LENGTHS must list at least one length")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = load_model(MODEL_DIR, DEVICE)
    config_text = model.config.text_config

    max_len = max(CONTEXT_LENGTHS) + 1
    docs = [build_doc(tokenizer, max_len, SEED + i) for i in range(DOCS)]
    print(f"configs: {', '.join(name for name, _ in specs)}  (reference: fp16 KV)")
    print(f"docs={DOCS} lengths={CONTEXT_LENGTHS} chunk={CHUNK} warmup={WARMUP}")

    results: dict[str, dict[int, dict]] = {}
    for length in CONTEXT_LENGTHS:
        t0 = time.time()
        agg: dict[str, dict] = {name: {"sum_kl": 0.0, "kl_list": [], "disagree": 0, "n": 0} for name, _ in specs}
        for doc in docs:
            stats = _run_length(model, config_text, doc, length, specs, DEVICE)
            for name, _ in specs:
                s = agg[name]
                s["sum_kl"] += stats[name]["sum_kl"]
                s["kl_list"].extend(stats[name]["kl_list"])
                s["disagree"] += stats[name]["disagree"]
                s["n"] += stats[name]["n"]
        print(f"[length {length}] done in {time.time() - t0:.0f}s")
        for name, _ in specs:
            summary = _summarize(agg[name])
            results.setdefault(name, {})[length] = summary
            ok = (
                summary["mean_kl"] <= KL_MEAN_PASS
                and summary["p95_kl"] <= KL_P95_PASS
                and summary["top1_agree"] >= TOP1_PASS
            )
            print(
                f"  {name:<10} mean={summary['mean_kl']:.4f} p95={summary['p95_kl']:.4f} "
                f"max={summary['max_kl']:.4f} top1={100 * summary['top1_agree']:6.2f}% "
                f"(n={summary['n']}) {'PASS' if ok else 'CHECK'}"
            )

    print()
    print(f"lossless gates: mean KL <= {KL_MEAN_PASS}, p95 <= {KL_P95_PASS}, top-1 >= {100 * TOP1_PASS:.0f}%")

    if OUT:
        out = Path(OUT)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2))
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
