# Remanentia — Rust Acceleration Status

**Date:** 2026-03-27
**Audited by:** Arcane Sapience

---

## Summary

Two Rust PyO3 modules. Both built, installed, and wired into production code.

| Module | Purpose | PyO3 | Built | Wired In | Measured Speedup |
|--------|---------|:----:|:-----:|:--------:|:----------------:|
| `remanentia_search` | BM25 scoring + cosine batch | 0.25 | cp312 | Gated (50K threshold) | ~2–4x on BM25 |
| `arcane_stdp` | STDP weight update + LIF step | 0.25 | cp312 | Auto-detect in snn_backend + snn_daemon | 1.6–2.9x on SNN |

## remanentia_search (rust_search/)

**Location:** `04_ARCANE_SAPIENCE/rust_search/`
**Exports:**
- `BM25Index` class: `build()`, `search()`, `get_para_info()`, `num_paragraphs()`, `vocab_size()`
- `cosine_batch()` function: parallel cosine similarity for embedding rerank

**Integration in memory_index.py:**
- `_get_rust_bm25_class()` — lazy import, caches result
- `_should_use_rust_bm25()` — env var `REMANENTIA_USE_RUST_BM25` or 50K paragraph threshold
- `_ensure_rust_bm25()` — builds Rust index on demand, rebuilds when dirty
- `_search_rust_bm25()` — calls `bm25.search()`, converts to dict

**State:** Built for cp312. Uses real term frequency (not binary TF). Auto-gated at 50K paragraphs, overridable via env var.

## arcane_stdp (rust_stdp/)

**Location:** `04_ARCANE_SAPIENCE/rust_stdp/`
**Exports:**
- `stdp_batch(w, spiked, last_spike, t_now, mask, a_plus, a_minus, tau, w_max)` — vectorised outer product STDP with rayon
- `lif_step(v, w, i_ext, v_rest, v_thresh, v_reset, tau_m, dt_ms)` — LIF step + spike detection

**Integration:**
- `snn_backend.py:DenseCPULIFNetwork` — both `run()` and `_apply_stdp_batch()` use Rust when available
- `snn_daemon.py:SimpleLIFNetwork` — both `run()` and `_apply_stdp()` use Rust when available
- Auto-detect at import time, silent fallback to numpy

**Benchmarks (measured 2026-03-27):**

| Neurons | Rust | Python | Speedup | Spikes (match) |
|---------|------|--------|---------|:--------------:|
| 1,000 | 156ms | 453ms | 2.9x | 8,314 = 8,314 |
| 2,000 | 907ms | 1,422ms | 1.6x | 16,633 = 16,633 |
| 5,000 | 3,531ms | 7,000ms | 2.0x | 41,527 = 41,527 |

Spike counts and weight means identical at all sizes. Rust path is bit-exact with Python.

At N<500, Python is faster (FFI overhead dominates). The daemon defaults to 1000 neurons, where Rust gives 2.9x.

---

## What's NOT Worth Porting to Rust

| Hotpath | Time | Why Not |
|---------|------|---------|
| Tokenization (regex) | ~50ms/build | Python regex is C-backed, marginal gain |
| Paragraph splitting | ~20ms/build | String split, IO-bound |
| Answer extraction | 5 paragraphs/query | Runs on tiny data, not bottleneck |
| Entity extraction | GLiNER (GPU) | Neural model, not Python code |
| Temporal graph | Built once | Not query-path |
| Consolidation clustering | O(n log n), n<500 | Timsort is excellent, runs rarely |
| Embedding/cross-encoder | PyTorch models | GPU-bound, not CPU |

## Build

```bash
# Both modules
cd 04_ARCANE_SAPIENCE/rust_search && maturin build --release --interpreter path/to/python3.12
cd 04_ARCANE_SAPIENCE/rust_stdp && maturin build --release --interpreter path/to/python3.12

# Install
pip install target/wheels/remanentia_search-*.whl
pip install target/wheels/arcane_stdp-*.whl
```

- rustc 1.93.0
- maturin 1.12.6
- PyO3 0.25 (both modules)
