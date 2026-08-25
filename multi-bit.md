# Multi-bit FibQuant: 3-bit and 4-bit operating points

Status of this document: all issues below were verified empirically in this repo's
venv (torch 2.13.0, transformers 5.15.1, macOS/MPS) on 2026-08-25. Measurements are
reproducible from the snippets quoted inline.

## Target operating points

The current design point is k=4, N=256 (one codebook index per 4-coordinate block,
d=256 → 64 blocks per head vector). Bits per coordinate is `log2(N)/k`, so:

| b (bits/coord) | k | N      | bits per block index | natural container | container waste |
|----------------|---|--------|----------------------|-------------------|-----------------|
| 2 (current)    | 4 | 256    | 8                    | uint8             | none            |
| 3              | 4 | 4096   | 12                   | uint16            | 25% unless packed |
| 4              | 4 | 65536  | 16                   | uint16            | none (exact fit: 65535 = 2^16−1) |

Norms stay fp16 (2 B/head-vector) in all cases. Memory per head vector (payload +
norm, vs 512 B fp16): b=2 → 66 B (7.8x), b=3 unpacked → 130 B (3.9x), b=3 packed →
98 B (5.2x), b=4 → 130 B (3.9x). Note the consequence: **in uint16 containers b=3
and b=4 cost identical memory** — without bit-packing the "b=3" eval is really a
b=4-cost eval with a worse codebook (see issue 3).

---

## Issue 1 (critical, silent corruption): uint8 index cast wraps for N > 256

**Where:** `fibquant/quantize.py:40`

```python
indices = scores.argmax(-1).to(torch.uint8)
```

**Problem:** `argmax` returns values up to N−1. Casting to uint8 wraps mod 256 with
no error or warning. At the b=3 point (N=4096) this corrupts almost every index; the
cache then decodes noise and lm-eval silently reports garbage.

**Measured** (synthetic N=4096 codebook, replicating `encode()`'s math exactly):

- 93.88% of stored indices differ from the true argmax after the cast.
- Cosine similarity between the correctly-decoded and stored-then-decoded vectors:
  mean 0.063 (i.e. statistically indistinguishable from random).

**Why b=2 never sees it:** N=256 → argmax ∈ [0, 255] fits uint8 exactly.

**Fix:** pick the index dtype from N. Verified in torch 2.13 that `torch.uint16`
supports every operation `FibQuantLayer`/`decode` perform on index tensors:
`cat`, slicing (crop), `index_select` (beam reorder), advanced indexing
(batch_select), `repeat_interleave`, `element_size`, `torch.save/load`, and
`.long()` gather against the codebook.

```python
# fibquant/quantize.py
def index_dtype(n_levels: int) -> torch.dtype:
    if n_levels <= 2**8:
        return torch.uint8
    if n_levels <= 2**16:
        return torch.uint16   # b=4 fits exactly: indices in [0, 65535]
    raise ValueError(f"n_levels={n_levels} exceeds uint16; not supported")

# in encode():
indices = scores.argmax(-1).to(index_dtype(codebook.shape[0]))
```

No changes needed downstream: `decode()` already does `indices.long()`, and
`FibQuantLayer.stored_bytes()` uses `element_size()`, so memory accounting adapts
automatically. (Optionally add `assert spec.n_levels == spec.codebook.shape[0]` in
`FibQuantSpec.from_checkpoint` to catch mismatched checkpoints early.)

---

## Issue 2 (OOM): intermediate tensors scale as O(n·N), O(B·H·S·N), or O(N²)

Three separate materializations are harmless at N=256 and fatal above it.

### 2a. `encode()` scores tensor — `fibquant/quantize.py:39`

Shape `(B, H_kv, S, d/k, N)` fp32. At the eval envelope (batch 8, S=2048, H=2,
64 blocks):

| N    | scores tensor | outcome                     |
|------|---------------|-----------------------------|
| 256  | 2.1 GB        | works today (transient)     |
| 4096 | 34.4 GB       | OOM on MPS / most GPUs      |
| 65536| 0.55 TB       | OOM anywhere                |

**Fix:** tile over rows. Flatten to `(B·H·S, 64)` blocks, process in row chunks,
argmax each chunk against the full codebook, write into a preallocated output.
Keep the per-chunk score matrix ≤ ~1 GB (e.g. chunk = 2^12 rows → 1.07 GB at
N=65536). Decode-step calls (S=1) are unaffected; only prefill needs this.

### 2b. `_lloyd_max` training — `fibquant/codebook.py:88-94` (and the `onehot`
scatter at :91-94)

`diff` is `(n, N, k)` fp32 plus `d2` `(n, N)` plus the `onehot` `(n, N)`. With the
build default `m_factor=30` (n = 30·N samples):

| N    | n        | transient per Lloyd iteration        |
|------|----------|--------------------------------------|
| 256  | 7,680    | ~31 MB (why b=2 "just works")        |
| 4096 | 122,880  | ~10.1 GB × 25 iters × 4 restarts     |
| 65536| 1,966,080| ~2.6 TB — impossible                 |

(Measured at n=20k, N=1024, k=4: ~700 MB peak vs 410 MB for diff+d2 alone — the
onehot and subtraction temporaries add ~1.7x on top of the two-tensor formula.)

**Fix:** reuse the augmented-inner-product identity `encode()` already uses — it
avoids materializing distances entirely:

- `score(s, c) = 2⟨s, c⟩ − ‖c‖²`, so assignment = `argmax` of
  `[2s, −1] @ [c, ‖c‖²]ᵀ`, and the assigned distance² is exactly
  `‖s‖² − max_score` (no `(n, N, k)` tensor, no separate `d2`).
- Compute in sample-row chunks so the `(chunk, N)` score matrix stays bounded.
- Replace the `onehot.t() @ samples` centroid update with
  `torch.zeros(N, k).index_add_(0, assign, chunk)` +
  `torch.bincount(assign, minlength=N)` for counts.
- `cell_mse` via `scatter_add_` of `(‖s‖² − max_score)`.

### 2c. `build_codebook()` final-MSE evaluation — `fibquant/codebook.py:132`

Same `(n, N, k)` `diff` per restart. Fix identically (share the chunked scorer
with `_lloyd_max`).

### 2d. min-pairwise-distance diagnostic — `scripts/build_codebook.py:37-40`

Materializes `(N, N, k)`: 268 MB at N=4096 (survives), ~69 GB + 17 GB `d2` at
N=65536 (OOM). Fix: loop over codebook row chunks, keep a running per-row min.

---

## Issue 3: `bytes_per_token` reports packed bits; storage uses whole containers

**Where:** `fibquant/quantize.py:61-72` vs `fibquant/cache.py:155-161`.

`bytes_per_token` computes `bits = (n_levels − 1).bit_length()` (12 for N=4096) and
divides by 8 — idealized packed storage. `FibQuantLayer.stored_bytes()` is honest
(`element_size()`), so after fixing issue 1 the two disagree at b=3: reported
payload 96 B/head-vector, actual 128 B.

**Decision required:** either

- **(a) Accept container granularity.** Then b=3 in uint16 costs exactly what b=4
  costs (130 B incl. norm, 3.9x) — for the eval sweep this means: run the b=4
  point (N=65536) to get best quality at 3.9x, and only build a 3-bit point if
  you implement packing. Update `bytes_per_token` to take the container into
  account (report both `packed` and `container` numbers).
- **(b) Implement true 3-bit packing.** 12-bit fields cross byte boundaries, so
  `FibQuantLayer.update()` (which appends per token, `cache.py:85-92`) needs a
  bitstream writer with per-row carry state, and `crop`/`reorder_cache`/
  `batch_select_indices` must unpack-repack. This is a real chunk of work; defer
  unless the exact-3.0-bits datapoint matters.

Recommendation: (a) now, (b) later if needed.

---

## Issue 4: b=2 baked into script defaults and docstrings (silent mis-eval)

None of these crash; they quietly evaluate the wrong config when `--spec` is
forgotten:

- `scripts/eval.py:69-70` and `scripts/eval_cuda.py:71-72` — `--spec` defaults to
  `models/fibquant/fibquant_d256_k4_N256.pt`; help text says "FibQuant b=2".
- `scripts/sanity.py:103` — same hardcoded default.
- `fibquant/cache.py:5-6` module docstring ("~8x at b=2"), `cache.py:56`
  ("uint8 block indices"), `fibquant/quantize.py:4` ("one uint8 block index").

**Fix:** make `--spec` required when `--fibquant` is set (or derive the default
from a `--bits {2,3,4}` flag mapping to `default_spec_path`), and refresh the
docstrings to say "one packed/unpacked container element per k-block".

---

## Issue 5 (secondary): training knobs worth retuning at large N

- `m_factor=30` is a `build_codebook()` keyword (`codebook.py:118`) not exposed in
  `scripts/build_codebook.py`. At N=65536 that's ~2M samples/restart; with the
  chunked scorer each Lloyd iteration is ~1.3 TFLOP — fine on CUDA, slow on MPS.
  Expose `--m-factor`, and consider `--restarts 2` for the b=4 build.
- Dead codewords get likelier as N grows. `_lloyd_max` has empty-cell repair
  (`codebook.py:101-106`) but no reporting; add a final dead-cell count print to
  `scripts/build_codebook.py` and bump `m_factor`/`lloyd_iters` if it exceeds ~1%.
- `build_radii`'s `beta.ppf` at extreme quantiles (q = 0.5/65536) is computed in
  float64 and stays stable; the existing radius-range print is a sufficient check.

---

## Step-by-step plan

1. `quantize.py`: add `index_dtype()`, cast argmax with it (issue 1); tile the
   score/argmax over rows (issue 2a).
2. `codebook.py`: rewrite `_lloyd_max` assignment with the augmented-score trick +
   row chunking + `index_add_`/`bincount` centroids (2b); share the chunked scorer
   with `build_codebook`'s final-MSE loop (2c).
3. `scripts/build_codebook.py`: chunk the (N, N, k) diagnostic (2d); expose
   `--m-factor`; print dead-codeword count (issue 5).
4. `quantize.py` `bytes_per_token`: report container-based numbers alongside
   packed (issue 3a).
5. Script/doc defaults sweep (issue 4).
6. Build and validate, in this order (sanity gate catches any residual dtype bug —
   a wraparound shows up there as roundtrip cosine ≈ 0.06):

```bash
.venv/bin/python scripts/build_codebook.py --n-levels 65536          # b=4
.venv/bin/python scripts/sanity.py --spec models/fibquant/fibquant_d256_k4_N65536.pt
.venv/bin/python scripts/eval.py --tag fibquant-b4 --fibquant \
    --spec models/fibquant/fibquant_d256_k4_N65536.pt --tasks hellaswag,wikitext

.venv/bin/python scripts/build_codebook.py --n-levels 4096           # b=3 (see issue 3)
.venv/bin/python scripts/sanity.py --spec models/fibquant/fibquant_d256_k4_N4096.pt
```

**Acceptance criteria:** roundtrip cosine increases monotonically b=2 < b=3 < b=4
(b=2 reference from `sanity.py` on the N=256 spec); sanity logit max-abs-diff and
KL shrink as b grows; hellaswag/wikitext land between the b=2 run and baseline;
`stored_bytes`-derived compression matches the table above (3.9x for b=4 and
unpacked b=3).

## Alternative worth knowing: k=2 keeps everything in uint8

b=4 can also be built as k=2, N=256 (128 blocks/vector, 8-bit index → no dtype
change, no packing waste), and b=3 as k=2, N=64 (6-bit index in uint8, 25% waste).
This dodges issues 1 and 3 entirely at the cost of weaker 2-D codebooks; the
FibQuant design point is k=4, so prefer the k=4 + uint16 route for comparable
results, but k=2 is a useful cross-check if the uint16 path misbehaves.
