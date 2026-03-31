# Performance Tuning

Measured on AMD Ryzen 5 3600, 32 GB DDR4, NTFS disk. All benchmarks use
`time.perf_counter()` and are asserted in CI via budget tests in
`tests/test_pipeline_performance.py` (43 tests).

## Regex Pipeline (core retrieval path)

The regex pipeline is the hot path for every query — no LLM, no GPU, no
network. Measured on a 47K-character document (100× repeated paragraph).

| Component | Avg (ms) | Budget |
|-----------|---------|--------|
| `parse_dates` (47K chars) | 0.164 | <50ms |
| `regex_entities` (47K) | 0.132 | <50ms |
| `extract_answer` (47K) | 0.029 | <20ms |
| `normalize_answer` | 0.001 | <1ms |
| `answers_match` | 0.001 | <1ms |
| `fuzzy_match` | 0.000 | <1ms |
| `normalize_number` | 0.001 | <1ms |
| `resolve_backend` | 0.009 | <5ms |
| `NullBackend.complete` | 0.000 | <0.5ms |
| `KnowledgeStore.add_note` | 0.023 | <50ms |
| **TOTAL (regex pipeline)** | **0.360** | **<200ms** |

## Rust vs Python

7 PyO3 crates built with maturin. Python fallback preserved in every module.
Measured on same 47K-character workload.

| Module | Python | Rust | Speedup |
|--------|--------|------|---------|
| `parse_dates` (47K) | 0.168ms | 0.169ms | ~1.0× |
| `extract_answer` (47K) | 0.030ms | 0.030ms | ~1.0× |
| `regex_entities` (47K) | 0.125ms | 0.061ms | **2.1×** |
| Full pipeline (470K, large) | 9.07ms | 0.60ms | **14.1×** |

On small workloads (47K), PyO3 FFI overhead and Python regex caching
neutralise the Rust advantage. On large workloads (470K+), Rust's compiled
regex engine dominates — 14.1× speedup.

### All Rust crates

| Crate | Module | Speedup |
|-------|--------|---------|
| `remanentia_temporal` | temporal_graph, date_normalizer | 14.2× (large) |
| `remanentia_answer_extractor` | answer_extractor | 11.4× (large) |
| `remanentia_fact_decomposer` | fact_decomposer, arcane_retriever | ~7× |
| `remanentia_answer_normalizer` | answer_normalizer | ~6× |
| `remanentia_search` | memory_index (BM25, Rayon) | ~3-5× |
| `arcane_stdp` | snn_backend | ~2-3× |
| `remanentia_entity_extractor` | entity_extractor | ~2× |
| `remanentia_knowledge_store` | knowledge_store | NEW |
| `remanentia_consolidation` | consolidation_engine | NEW |
| `remanentia_skill_extractor` | skill_extractor | NEW |
| `remanentia_active_retrieval` | active_retrieval | NEW |

## v0.4 Feature Performance

New features added in v0.4 (Hermes/OpenClaw/Engram-inspired). All measured
with `time.perf_counter()`, asserted in CI.

### Per-operation benchmarks

| Operation | Measured | Budget | Notes |
|-----------|---------|--------|-------|
| `classify_fact` (9 types) | **2.7 µs** | <10 µs | Priority-ordered regex cascade |
| `_recency_weight` | **0.5 µs** | <5 µs | Single `pow()` call |
| SHA-256 hash (47K chars) | **0.014 ms** | <0.5 ms | Content-hash indexing |
| `capacity_report` (5 cats, 50 files) | **1.9 ms** | <10 ms | Per-category char counting |
| `age_memories` (50 files) | **1.4 ms** | <50 ms | Frontmatter parse + state transition |
| `build_summary_dag` (100 traces) | **0.21 ms** | <5 ms | Multi-level DAG construction |
| `search_summary_dag` (100 traces) | **0.25 ms** | <1 ms | Top-down score-based expansion |
| `heartbeat` (empty dirs) | **14 ms** | <20 ms | observe + consolidate + age + capacity |
| **Full v0.4 pipeline** (50 turns) | **3.1 ms** | <500 ms | decompose → ArcaneRetriever(decay) → context |

### Feature details

**Extended fact types (9 types)**: Priority-ordered regex cascade classifying
sentences into decision, correction, principle, commitment, skill, plan,
preference, state, event. Adds <3µs overhead per sentence over the original
4-type classifier. Wired through `decompose_sessions()` → `FactIndex` →
`ArcaneRetriever` → `build_context()`.

**Temporal recency decay**: Exponential decay `weight = 2^(-age/half_life)`
applied in RRF fusion. 30-day half-life default. Adds 0.5µs per fact in the
fusion step. Recent facts score higher for knowledge-update queries.

**Content-hash incremental indexing**: SHA-256 per file, cached in
`snn_state/content_hashes.json`. Second build with no changes: 0 documents
indexed (all skipped). Hash check: 0.014ms per 47K-char file.

**Bounded memory capacity**: `capacity_report()` scans semantic memory
categories and reports chars/limit/usage%. 1.9ms for 50 files across 5
categories. Displayed in `remanentia status`.

**Memory lifecycle aging**: `age_memories()` transitions validity_state
(active→stale after 90 days, stale→archived after 365 days). 1.4ms for 50
files. Called by heartbeat every ~5 minutes.

**Hierarchical summary DAGs**: Multi-level compression with DAG_FANOUT=4.
100 traces → ~125 nodes (100 leaves + ~25 internal). Build: 0.21ms. Search
(top-down expansion to leaves): 0.25ms.

**Heartbeat**: Combined maintenance tick (observe + consolidate + age +
capacity). 14ms on empty directories, variable with pending traces.

## Index Build Time

Default build scans all configured sources (~2,000 documents, ~20K paragraphs).

| Mode | Time | When to use |
|------|------|-------------|
| Full build (no embeddings) | 5-8s | First build or structure change |
| Incremental build (no changes) | 0.3-0.5s | Subsequent builds |
| Incremental build (10 changed) | 1-2s | After editing a few files |
| Full build (with embeddings) | 10-30s | When using embedding rerank |

### Content-hash incremental builds

```python
idx = MemoryIndex()
stats = idx.build(incremental=True)  # skips unchanged files
# stats["hash_hits"]   → files skipped
# stats["hash_misses"] → files (re-)indexed
```

Force full rebuild with `incremental=False`.

### Incremental single-file updates

```python
idx.add_file(Path("new_trace.md"))  # adds without rebuild
```

IDF values update incrementally.

## Query Latency

| Stage | Cold Start | Warm |
|-------|-----------|------|
| BM25 (Python, 20K paras) | 5-20ms | 5-20ms |
| BM25 (Rust, 20K paras) | 1-5ms | 1-5ms |
| Bi-encoder rerank | 100-200ms | 50-100ms |
| Cross-encoder rerank | 200-400ms | 100-200ms |
| Answer extraction | 1-5ms | 1-5ms |
| Total (no models) | 10-30ms | 10-30ms |
| Total (with models) | 300-600ms | 150-300ms |

### Rust BM25

Automatically activates at 50K+ paragraphs. Force with:

```bash
export REMANENTIA_USE_RUST_BM25=1
```

Requires `remanentia_search` wheel installed.

### Model warmup

Call `idx.warm_models()` after loading to start background model loading.
First query with models takes 2-5s (model download/load). Subsequent
queries use cached models.

### Cross-encoder timeout

The cross-encoder loads in a background thread with 5s timeout. If loading
takes longer, BM25 results are returned immediately. The model continues
loading — next query gets reranking.

## ArcaneRetriever Performance

4-channel parallel retrieval with RRF fusion + recency decay.

| Operation | Time | Notes |
|-----------|------|-------|
| Single retrieve (20 sessions) | <50ms | All 4 channels parallel |
| RRF fusion (4 channels) | <1ms | Dedup + recency weight |
| `_recency_weight` per fact | 0.5µs | `pow(2, -age/half_life)` |
| `build_context` | <1ms | String assembly |
| Full pipeline (50 turns) | 3.1ms | decompose + retrieve + context |

### Recency decay curve

| Age (days) | Weight (half_life=30) |
|------------|----------------------|
| 0 (today) | 1.000 |
| 15 | 0.707 |
| 30 | 0.500 |
| 60 | 0.250 |
| 90 | 0.125 |
| 365 | ~0.000 |

## Consolidation Performance

| Operation | 10 traces | 100 traces | Budget |
|-----------|----------|-----------|--------|
| `consolidate()` | <50ms | <200ms | 500ms |
| `build_summary_dag()` | <0.1ms | 0.21ms | <5ms |
| `search_summary_dag()` | <0.1ms | 0.25ms | <1ms |
| `age_memories()` (50 files) | 1.4ms | — | <50ms |
| `capacity_report()` (50 files) | 1.9ms | — | <10ms |

## SNN Performance

| Neurons | Rust | Python | Speedup |
|---------|------|--------|---------|
| 1,000 | 156ms | 453ms | 2.9× |
| 2,000 | 907ms | 1,422ms | 1.6× |
| 5,000 | 3,531ms | 7,000ms | 2.0× |

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
| Content hash cache (2K files) | ~0.5 MB |
| Summary DAG (100 traces) | ~50 KB |

## Complete Rust Function Benchmark (11 crates, 27 functions)

All functions measured on AMD Ryzen 5 3600, 1,000 iterations, `time.perf_counter()`.

### Crate 1: remanentia_temporal

| Function | Measured | Wired into |
|----------|---------|------------|
| `parse_dates` (short text) | 11.8 µs | temporal_graph.parse_dates |
| `normalise_vague_date` | 5.1 µs | date_normalizer._rule_based_normalise |

### Crate 2: remanentia_answer_extractor

| Function | Measured | Wired into |
|----------|---------|------------|
| `extract_answer` (short) | 3.2 µs | answer_extractor.extract_answer |
| `fuzzy_match` | 0.3 µs | answer_extractor.fuzzy_match |
| `normalize_number` | 5.8 µs | answer_extractor.normalize_number |
| `extract_best_sentence` | — | answer_extractor.extract_best_sentence |

### Crate 3: remanentia_answer_normalizer

| Function | Measured | Wired into |
|----------|---------|------------|
| `normalize_answer` | 0.4 µs | answer_normalizer.normalize_answer |
| `answers_match` | 0.7 µs | answer_normalizer.answers_match |
| `extract_answer_items` | 1.6 µs | answer_normalizer.extract_answer_items |

### Crate 4: remanentia_entity_extractor

| Function | Measured | Wired into |
|----------|---------|------------|
| `regex_entities` (short) | 7.2 µs | entity_extractor._regex_entities |
| `extract_relations` | — | entity_extractor.extract_relations |

### Crate 5: remanentia_fact_decomposer

| Function | Measured | Wired into |
|----------|---------|------------|
| `classify_fact_type` (9 types) | 0.3 µs | fact_decomposer._classify_fact |
| `split_sentences` | 0.3 µs | fact_decomposer._split_sentences |
| `has_change_verb` | 0.2 µs | fact_decomposer._build_fact |
| `tokenize_words` | 7.4 µs | arcane_retriever._check_sufficiency |

### Crate 6: remanentia_search

| Function | Measured | Wired into |
|----------|---------|------------|
| `BM25Index.search` | 1-5 ms (20K paras) | memory_index.search (≥50K) |
| `cosine_batch` | — | memory_index._compute_embeddings |
| `tokenize` | 1.6 µs | memory_index._tokenize |
| `classify_paragraph` | 6.1 µs | memory_index._classify_paragraph |
| `split_paragraphs` | — | (available, Python windowing used) |
| `token_counts` | — | (available) |

### Crate 7: arcane_stdp

| Function | Measured | Wired into |
|----------|---------|------------|
| `stdp_batch` | ~2-3× speedup | snn_backend.stdp_update |
| `lif_step` | ~2-3× speedup | snn_backend.lif_step |

### Crate 8: remanentia_knowledge_store

| Function | Measured | Wired into |
|----------|---------|------------|
| `tokenize` | 1.5 µs | knowledge_store._tokenize |
| `extract_keywords` | 13.2 µs | knowledge_store._extract_keywords |
| `extract_entities` | 1.7 µs | knowledge_store._extract_entities |
| `extract_person_names` | 7.1 µs | knowledge_store.extract_person_names |

### Crate 9: remanentia_consolidation

| Function | Measured | Wired into |
|----------|---------|------------|
| `extract_entities` | 15.7 µs | consolidation_engine._extract_entities |
| `extract_key_lines` | 1.6 µs | consolidation_engine._extract_key_lines |
| `extract_typed_relations` | — | consolidation_engine._extract_typed_relations |
| `parse_frontmatter` | 6.8 µs | consolidation_engine._parse_frontmatter |

### Crate 10: remanentia_skill_extractor

| Function | Measured | Wired into |
|----------|---------|------------|
| `tokenize_lower` | 1.9 µs | skill_extractor._tokenize_lower |
| `matches_skill_marker` | 2.9 µs | skill_extractor.extract_skills |
| `rank_skills_by_overlap` | — | (available for query_skills) |

### Crate 11: remanentia_active_retrieval

| Function | Measured | Wired into |
|----------|---------|------------|
| `extract_decision_points` | 1.0 µs | active_retrieval.extract_decision_points |

## Test Coverage

1,430+ tests, 100% coverage, 43+ dedicated performance tests with budget
assertions. All benchmarks run in CI and fail if budgets are exceeded.
