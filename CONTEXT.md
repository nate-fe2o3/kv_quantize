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

## Architecture terms (this repo's seams)

- **Nearest-codeword scorer** (`fibquant/scoring.py`) — the deep primitive
  computing nearest codeword assignment (`(indices, dist2)`) under the
  augmented inner-product identity `score(s, c) = 2⟨s, c⟩ − ‖c‖²`, chunked so
  peak memory stays within a budget. Single implementation shared by offline
  codebook construction (Lloyd-Max assignment, final-MSE evaluation) and
  runtime `encode`. Also owns the chunked pairwise min-distance diagnostics.
- **KV payload** (`fibquant/payload.py`) — the compressed storage of one
  layer: (packed) block indices + fp16 norms for keys and values. Owns the
  storage format (container dtype policy, dim conventions, pair-packing,
  within-row packing invariant, byte accounting) and in-place protocol verbs
  (append, crop, reorder, select, repeat, reset). Packing is **within-row**:
  adjacent k-blocks of one head vector are pair-packed, never across tokens,
  so sequence-dim ops never need unpack-repack.
- **Eval harness** (`fibquant/eval_harness.py`) — one deep adapter over
  lm-eval: model loading (incl. the MPS single-worker fix), `enable_fibquant`,
  the generate-path cache injection, presence-penalty emulation, and results
  emission. `run_eval(EvalConfig)` is the interface for eval scripts.
- **Runtime install** (`fibquant/runtime.py`) — the explicit lifecycle of the
  FibQuant patches: `FibQuantRuntime(spec).install(model=None|model)` patches
  the model class's `forward()` and its generate cache factory
  (`_prepare_cache_for_generation`) together; idempotent per operating point,
  re-patches with a warning on a different spec, `uninstall()` restores the
  originals, `active_spec`/`active_specs()` expose the current state.
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
