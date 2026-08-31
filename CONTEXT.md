# CONTEXT.md — kv_quantize domain model

FibQuant: universal vector quantization for random-access KV-cache compression
(KV cache of a transformer's full-attention layers, compressed by quantizing
each head vector against a shared codebook). Domain language below; use these
terms consistently in code, docs, and reviews.

## Core entities

- **Head vector** — one `(d,)` key or value vector of one attention head at one
  token position. The unit of compression: each head vector is encoded
  independently (fp16 norm header + one container element per block index).
- **Block index** — `k` consecutive coordinates of a head vector
  correspond to one codebook entry. `blocks = d / k` per head vector.
- **Codebook** — `(N, k)` codewords on (near) the unit ball, shared across
  layers/heads/prompts, built offline for one operating point. Trained with
  multi-restart Lloyd-Max (empty-cell repair) via the augmented-score identity.
- **Operating point** — the tuple `(d, k, N, bits/coord)` that names one
  FibQuant configuration: `bits = log2(N) / k`, `N = 1 << (bits * k)`.
  Checkpoints are stored as `models/fibquant/fibquant_d{d}_k{k}_N{N}.pt`.
- **Spec checkpoint** — the on-disk artifact: codebook + rotation + metadata
  (`d`, `k`, `n_levels`, `seed`, `mse`). Loaded into a **FibQuantSpec**.
- **Rotation** — one shared Haar-random orthogonal `(d, d)` matrix applied
  before block decomposition, deterministic given the seed.

## Operating points

With `d=256` and `k=4`, each head vector has 64 block indices plus one fp16
norm. The supported operating points are:

| bits/coord | N | index storage | bytes/head vector | compression vs fp16 |
|------------|---|---------------|-------------------|---------------------|
| 2 | 256 | uint8 | 66 | 7.8x |
| 3 | 4,096 | pair-packed 12-bit | 98 | 5.2x |
| 4 | 65,536 | uint16 | 130 | 3.9x |

Unpacked 3-bit indices occupy uint16 containers and therefore cost the same
130 bytes as the 4-bit operating point. Pair-packing is what makes the 3-bit
point distinct in memory. Index dtype must always be selected from `N`; casting
an index for `N > 256` to uint8 silently wraps it.

Large codebooks make unchunked score and pairwise-distance tensors impractical.
All assignment, MSE, and minimum-distance work therefore goes through the
chunked nearest-codeword scorer. Lloyd-Max centroid accumulation uses indexed
reduction rather than a sample-by-codeword one-hot matrix.

The preferred comparison keeps `k=4` across operating points. Alternatives
such as `(k=2, N=64)` for 3 bits/coord and `(k=2, N=256)` for 4 bits/coord keep
uint8 storage, but use weaker two-dimensional codebooks and are best treated
as diagnostic cross-checks rather than equivalent operating points.

## Architecture terms (this repo's seams)

- **Nearest-codeword scorer** (`fibquant/scoring.py`) — the deep primitive
  computing nearest codeword assignment (`(indices, dist2)`) under the
  augmented inner-product identity `score(s, c) = 2⟨s, c⟩ − ‖c‖²`, chunked so
  peak memory stays within a budget. Single implementation shared by offline
  codebook construction (Lloyd-Max assignment, final-MSE evaluation) and
  runtime `encode`. Also owns the chunked pairwise min-distance diagnostics.
- **Prepared codec** (`fibquant/codec.py`) — device-resident codebook, rotation,
  transpose, and augmented scoring state for one spec. All payloads sharing a
  spec reuse one prepared codec per device; the compatibility functions in
  `quantize.py` are one-call adapters over this module.
- **KV payload** (`fibquant/payload.py`) — the compressed storage of one
  layer: (packed) block indices + fp16 norms for keys and values. Owns the
  storage format (container dtype policy, dim conventions, pair-packing,
  within-row packing invariant, byte accounting) and in-place protocol verbs
  (append, crop, reorder, select, repeat, reset). Packing is **within-row**:
  adjacent k-blocks of one head vector are pair-packed, never across tokens,
  so sequence-dim ops never need unpack-repack.
- **Eval harness** (`fibquant/eval_harness.py`) — one deep adapter over
  lm-eval: model loading (incl. the MPS single-worker fix), scoped runtime
  installation, per-row presence-penalty emulation, and results emission.
  `run_eval(EvalConfig)` is the interface for eval scripts.
- **Probe support** (`fibquant/probes.py`) — deterministic unique filler,
  marker invariants, answer budgets, and operating-point matrices shared by
  the recall, logit-fidelity, and LongBench scripts. Each script retains its
  scenario-specific prompt layout, generation loop, and metrics.
- **Runtime install** (`fibquant/runtime.py`) — the explicit lifecycle of the
  FibQuant patches: `FibQuantRuntime(spec).install(model=None|model)` patches
  the model class's `forward()` and its generate cache factory
  (`_prepare_cache_for_generation`) together; idempotent per operating point,
  re-patches with a warning on a different spec, context-managed installs
  restore prior state, `uninstall()` restores the originals, and
  `active_spec`/`active_specs()` expose the current state.
  `enable_fibquant` is a thin back-compat wrapper over it.

## Environment specifics (load-bearing facts)

- Model: Qwen3.5-0.8B (`models/Qwen3.5-0.8B`), full-attention layers get
  FibQuant layers; linear-attention (Gated DeltaNet) layers keep their
  recurrent-state machinery, inherited from `DynamicCache` untouched.
- Dev machine is macOS/MPS; production evals run on Databricks serverless CUDA
  (`scripts/eval_cuda.py` constants + `env.yaml` custom environment).
- transformers 5.x removed `presence_penalty` from `GenerationConfig`; lm-eval
  omits the cache on the generate path, so the harness injects
  `FibQuantCache` explicitly.
- `models/` is gitignored; the Databricks notebook must point `MODELS_DIR` at
  DBFS/UC volume.

## Evaluation evidence

The August 2026 key-recall study compared fp16 KV, the 2-bit operating point,
and pair-packed 3-bit FibQuant on Qwen3.5-0.8B. Across 300 paired trials at
depths from 1,024 to 16,368 tokens, fp16 and 3-bit each achieved 300/300
successful singleton recalls. The 2-bit point achieved 284/300 (94.7%); its
misses were flat across depth, indicating an approximately 5% retrieval tax
rather than degradation that grows with context length. The 16 discordant
pairs against fp16 had a sign-test probability of approximately `1.5e-5`.
These results support pair-packed 3-bit as the default quality/memory tradeoff;
2-bit remains useful when the additional compression is required.

Recall filler must be unique. An earlier probe cycled roughly 250 tokens of
prose, causing repeated filler tokens to aggregate attention mass and produce
an artificial decline with depth. At depth 4,080, replacing repeated prose
with deterministic unique sentences raised 2-bit recall from 70% to 94% while
barely changing shallow-depth results. Shared probe support enforces unique,
deterministic filler so configurations receive identical inputs.

The key-recall result only measures easy singleton retrieval: a fixed-position,
single-word marker followed by short greedy generation. Buried and multi-token
retrieval belong to `scripts/multi_needle.py`; distributional fidelity and
top-1 agreement belong to `scripts/logit_kl.py`.
