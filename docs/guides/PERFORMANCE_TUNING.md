# Performance Tuning

## Index Build Time

Default build scans all configured sources (~2000 documents, ~20K paragraphs).
Build time: ~10–30s depending on GPU availability for embeddings.

### Reduce Build Time

- Set `use_gpu_embeddings=False` in `build()` to skip embedding computation
- Set `use_gliner=False` to skip entity extraction
- Build time without either: ~3–5s

### Incremental Updates

After initial build, use `add_file()` for new documents:

```python
idx.add_file(Path("new_trace.md"))
```

This adds paragraphs to the index without rebuilding. IDF values update
incrementally.

## Query Latency

Typical query pipeline timing:

| Stage | Cold Start | Warm |
|-------|-----------|------|
| BM25 (Python, 20K paras) | 5–20ms | 5–20ms |
| BM25 (Rust, 20K paras) | 1–5ms | 1–5ms |
| Bi-encoder rerank | 100–200ms | 50–100ms |
| Cross-encoder rerank | 200–400ms | 100–200ms |
| Answer extraction | 1–5ms | 1–5ms |
| Total (no models) | 10–30ms | 10–30ms |
| Total (with models) | 300–600ms | 150–300ms |

### Rust BM25

Automatically activates at 50K+ paragraphs. Force with:

```bash
export REMANENTIA_USE_RUST_BM25=1
```

Requires `remanentia_search` wheel installed.

### Model Warmup

Call `idx.warm_models()` after loading to start background model loading.
First query with models takes 2–5s (model download/load). Subsequent
queries use cached models.

### Cross-Encoder Timeout

The cross-encoder loads in a background thread with 5s timeout. If loading
takes longer, BM25 results are returned immediately. The model continues
loading — next query gets reranking.

## SNN Performance

| Neurons | Rust | Python | Speedup |
|---------|------|--------|---------|
| 1,000 | 156ms | 453ms | 2.9x |
| 2,000 | 907ms | 1,422ms | 1.6x |
| 5,000 | 3,531ms | 7,000ms | 2.0x |

Install `arcane_stdp` wheel for automatic Rust acceleration.

## Memory Usage

| Component | Memory |
|-----------|--------|
| Index (20K paragraphs) | ~50 MB |
| Embedding model (MiniLM) | ~90 MB |
| Cross-encoder model | ~90 MB |
| SNN (1K neurons) | ~4 MB |
| SNN (5K neurons) | ~100 MB |
| SNN (20K neurons, GPU) | ~1.5 GB VRAM |
