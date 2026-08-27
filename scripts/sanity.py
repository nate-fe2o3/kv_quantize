"""Sanity checks for the FibQuant cache on Qwen3.5-0.8B.

1. Roundtrip: encode/decode random Gaussian vectors, report cosine similarity.
2. Logits: fp16 baseline vs FibQuant (per the given --spec) on a fixed prompt,
   report max abs diff and KL divergence over the last-token logits.
3. Memory: persistent stored bytes per full-attention layer vs fp16.

Usage:  .venv/bin/python scripts/sanity.py --spec models/fibquant/fibquant_d256_k4_N256.pt
        .venv/bin/python scripts/sanity.py --spec models/fibquant/fibquant_d256_k4_N65536.pt  # b=4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer, DynamicCache

# Make the repo root importable even when run as "python scripts/foo.py".
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fibquant import FibQuantCache, FibQuantSpec, decode, encode, enable_fibquant, load_spec

MODEL_DIR = "models/Qwen3.5-0.8B"


def roundtrip(spec: FibQuantSpec) -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 2, 1024, spec.d)
    indices, norms = encode(x, spec.codebook, spec.rotation, spec.k)
    x_hat = decode(indices, norms, spec.codebook, spec.rotation, dtype=torch.float32)

    cos = torch.nn.functional.cosine_similarity(x, x_hat, dim=-1)
    rel_mse = ((x - x_hat).pow(2).mean(-1) / x.pow(2).mean(-1)).mean()
    print(f"[roundtrip] cosine sim: mean={cos.mean():.4f} min={cos.min():.4f}  rel-mse={rel_mse:.4f}")


def _load_model(device: str) -> torch.nn.Module:
    model = AutoModelForImageTextToText.from_pretrained(MODEL_DIR, dtype=torch.bfloat16)
    model.to(device)
    model.eval()
    return model


def logit_diff(spec: FibQuantSpec, device: str) -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = _load_model(device)

    prompt = "The quick brown fox jumps over the lazy dog. The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    cont = torch.tensor([[143]]).to(device)  # one dummy continuation token

    with torch.no_grad():
        base_cache = DynamicCache(config=model.config.text_config)
        out_base = model(input_ids=input_ids, past_key_values=base_cache, use_cache=True)
        base_logits = out_base.logits if hasattr(out_base, "logits") else out_base
        out_base_next = model(input_ids=cont, past_key_values=base_cache, use_cache=True)
        base_next = out_base_next.logits if hasattr(out_base_next, "logits") else out_base_next

        cache = FibQuantCache(config=model.config.text_config, spec=spec)
        out_fq = model(input_ids=input_ids, past_key_values=cache, use_cache=True)
        fq_logits = out_fq.logits if hasattr(out_fq, "logits") else out_fq
        out_fq_next = model(input_ids=cont, past_key_values=cache, use_cache=True)
        fq_next = out_fq_next.logits if hasattr(out_fq_next, "logits") else out_fq_next

    last = -1
    base_t, fq_t = base_logits[:, last], fq_logits[:, last]
    max_abs = (base_t - fq_t).abs().max().item()
    kl = torch.nn.functional.kl_div(
        fq_t.float().log_softmax(-1), base_t.float().log_softmax(-1), log_target=True, reduction="sum"
    ).item()

    b_next, f_next = base_next[:, 0], fq_next[:, 0]
    max_abs_next = (b_next - f_next).abs().max().item()
    arg_base, arg_fq = b_next.argmax(-1).item(), f_next.argmax(-1).item()
    print(f"[logits] max abs diff (prefill): {max_abs:.4f}  KL: {kl:.4f}")
    print(f"[logits] max abs diff (decode step): {max_abs_next:.4f}  argmax base={arg_base} fq={arg_fq}")


def memory_accounting(spec: FibQuantSpec, device: str) -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = _load_model(device)
    enable_fibquant(model, spec)

    prompt = "Once upon a time there was a very long story about a small model." * 32
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        cache = FibQuantCache(config=model.config.text_config, spec=spec)
        model(**inputs, past_key_values=cache, use_cache=True)

    stored = cache.stored_bytes()
    fp16 = cache.fp16_bytes()
    print(f"[memory] full-attn KV stored: {stored / 1024:.1f} KiB vs fp16 {fp16 / 1024:.1f} KiB")
    print(f"[memory] compression ratio: {fp16 / stored:.2f}x")
    per_token = stored / cache.get_seq_length()
    print(f"[memory] per token (6 full-attn layers): {per_token:.1f} B (fp16: {fp16 / cache.get_seq_length():.1f} B)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spec",
        type=str,
        required=True,
        help="FibQuant spec checkpoint, e.g. fibquant_d256_k4_N256.pt (b=2) or fibquant_d256_k4_N65536.pt (b=4)",
    )
    parser.add_argument("--device", type=str, default="mps")
    args = parser.parse_args()

    spec = FibQuantSpec.from_checkpoint(load_spec(args.spec))
    print(f"spec: d={spec.d} k={spec.k} N={spec.n_levels} b={spec.bits_per_coord} bits/coord")

    roundtrip(spec)
    logit_diff(spec, args.device)
    memory_accounting(spec, args.device)


if __name__ == "__main__":
    main()
