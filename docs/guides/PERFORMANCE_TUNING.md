# Performance Tuning

All benchmarks measured 2026-03-31 on verified hardware:

- **CPU:** Intel Core i5-11600K @ 3.90GHz (from `/proc/cpuinfo`)
- **RAM:** 31 GB DDR4 (from `free -h`)
- **Disk:** Local workstation SSD/HDD dataset checkout
- **Kernel:** 6.17.0-19-generic (from `uname -r`)
- **Rust:** Optional PyO3 extension crates installed locally; CI uses Python fallbacks
- **Method:** `time.perf_counter()`, budget assertions in CI

## Regex Pipeline (core retrieval path)

The regex pipeline is the hot path for every query — no LLM, no GPU, no
network. Measured on a 47K-character document (100× repeated paragraph).

| Component | Time (ms) | Budget |
|-----------|----------|--------|
| `parse_dates` (47K chars) | 0.285 | <50ms |
| `regex_entities` (47K) | (included above) | <50ms |
| `extract_answer` (47K) | (included above) | <20ms |
| **TOTAL regex pipeline (47K)** | **0.285** | **<200ms** |

### Large workload (470K chars)

| Component | Time (ms) |
|-----------|----------|
| `parse_dates` (470K) | 3.6 |
| `regex_entities` (470K) | 3.6 |
| `extract_answer` (470K) | <0.1 |
| **TOTAL regex pipeline (470K)** | **7.2** |

## Rust vs Python — Measured Speedups

### Short texts (~50-100 chars)

Measured by blocking the Rust import (`sys.modules[crate] = None`),
reloading the Python module, benchmarking, then restoring. Same inputs,
same iteration count (2000).

| Function | Python µs | Rust µs | Speedup |
|----------|----------|---------|---------|
| `classify_fact_type` (9 types) | 2.4 | 0.3 | **9.2×** |
| `regex_entities` | 12.2 | 1.4 | **8.5×** |
| `ce._extract_entities` | 27.0 | 3.2 | **8.3×** |
| `ks._extract_entities` | 10.0 | 2.2 | **4.6×** |
| `ks._extract_keywords` | 10.2 | 2.9 | **3.5×** |
| `parse_dates` | 12.8 | 4.4 | **2.9×** |

On short texts, 6 of 36 functions show meaningful speedup (2.9-9.2×).
The remaining functions show 0.8-1.1× (parity) because PyO3 FFI
overhead dominates when there is very little regex work per call.

### Large workloads (470K+ chars)

On large text inputs, FFI overhead is amortised across thousands of
regex matches. Previously measured (2026-03-31 morning session):

| Pipeline | Python | Rust | Speedup |
|----------|--------|------|---------|
| Full regex pipeline (470K) | 9.07ms | 0.60ms | **14.1×** |

The cross-over point where Rust beats Python is typically ~1K-5K
characters per call.

### Rust acceleration crates

| Crate | Module(s) | Short-text speedup |
|-------|-----------|-------------------|
| `remanentia_temporal` | temporal_graph, date_normalizer | **2.9×** |
| `remanentia_answer_extractor` | answer_extractor | ~1.0× short, 11.4× large |
| `remanentia_fact_decomposer` | fact_decomposer, arcane_retriever | **9.2×** |
| `remanentia_answer_normalizer` | answer_normalizer | ~1.0× |
| `remanentia_search` | memory_index (BM25, Rayon) | ~3-5× at 50K+ paras |
| `arcane_stdp` | snn_backend | ~2-3× |
| `remanentia_entity_extractor` | entity_extractor | **8.5×** |
| `remanentia_knowledge_store` | knowledge_store | **3.5-4.6×** |
| `remanentia_consolidation` | consolidation_engine | **8.3×** |
| `remanentia_skill_extractor` | skill_extractor | ~1.0× |
| `remanentia_active_retrieval` | active_retrieval | ~1.0× |
| `remanentia_retrieve` | retrieve, memory_index | **26.7×** hash_encode, **7.9×** RRF |

## Per-Function Rust Benchmarks (36 exported functions)

Absolute Rust call times, 1000 iterations each.

| Function | µs/call | Crate |
|----------|--------:|-------|
| `has_change_verb` | 0.13 | fact_decomposer |
| `classify_fact_type` (9 types) | 0.20 | fact_decomposer |
| `classify_paragraph` | 0.27 | search |
| `ar.extract_decision_points` | 0.43 | active_retrieval |
| `answers_match` | 0.60 | answer_normalizer |
| `split_sentences` | 0.62 | fact_decomposer |
| `normalize_answer` | 0.63 | answer_normalizer |
| `ce.parse_frontmatter` | 0.65 | consolidation |
| `extract_answer` | 0.82 | answer_extractor |
| `extract_relations` | 0.85 | entity_extractor |
| `tokenize_words` | 0.86 | fact_decomposer |
| `tokenize (index)` | 0.87 | search |
| `ce.extract_typed_relations` | 0.90 | consolidation |
| `ce.extract_key_lines` | 0.93 | consolidation |
| `fuzzy_match` | 1.03 | answer_extractor |
| `ks.extract_keywords` | 1.24 | knowledge_store |
| `normalize_number` | 1.30 | answer_extractor |
| `extract_answer_items` | 1.32 | answer_normalizer |
| `se.tokenize_lower` | 1.32 | skill_extractor |
| `ks.tokenize` | 1.49 | knowledge_store |
| `regex_entities` | 1.56 | entity_extractor |
| `ks.extract_entities` | 1.65 | knowledge_store |
| `ks.extract_person_names` | 2.72 | knowledge_store |
| `se.matches_skill_marker` | 3.10 | skill_extractor |
| `normalise_vague_date` | 3.47 | temporal |
| `ce.extract_entities` | 3.70 | consolidation |
| `extract_best_sentence` | 4.95 | answer_extractor |
| `parse_dates` | 8.79 | temporal |

All measured functions under 9 µs per call. 36 functions exported total
(28 measured individually, 8 via class methods or untested paths).

### remanentia_retrieve (12th crate, added 2026-04-04)

Measured 2026-04-04, 2000 iterations per function.

| Function | Python µs | Rust µs | Speedup |
|----------|----------|---------|---------|
| `hash_encode` (1000 neurons) | 1468 | 55 | **26.7×** |
| `reciprocal_rank_fusion` (5×100) | 571 | 72 | **7.9×** |
| `stem` | 0.50 | 0.18 | **2.7×** |
| `spike_feature` (100 neurons, 50 steps) | 2934 | 1719 | **1.7×** |
| `tokenize` (short text) | 22 | 27 | ~1.0× (FFI overhead) |
| `bigrams` (50 tokens) | 9 | 63 | 0.1× (FFI overhead) |

Key wins: `hash_encode` (called per trace during index build) and
`reciprocal_rank_fusion` (called per query in multi-source search)
are the dominant hot-path accelerations.

### Tier 2 extensions (added 2026-04-04)

Measured 2026-04-04. Tier 2 uses `#[pyclass]` persistent Rust objects
to avoid FFI serialisation overhead on each query.

| Function | Scale | Python µs | Rust µs | Speedup |
|----------|-------|----------|---------|---------|
| `FactIndex.query` (RustFactIndex pyclass) | 2000 facts | 9024 | 1025 | **8.8×** |
| `score_temporal_query` | 1000 events | 11410 | 4946 | **2.3×** |
| `build_temporal_edges` | 500 events | 18402 | 23546 | ~0.8× (FFI overhead) |
| `knowledge_search` (stateless) | 500 notes | ~7000 | ~7400 | ~1.0× (FFI overhead) |

Key lesson: stateful class methods benefit from `#[pyclass]` (index
built once, queried many times). Stateless one-shot functions with
complex data structures (dict-of-sets, dict-of-lists) pay FFI
serialisation cost that dominates for <10K items.

### Tier 3: consolidation + reflection + SNN (added 2026-04-04)

Measured 2026-04-04.

| Function | Scale | Python µs | Rust µs | Speedup |
|----------|-------|----------|---------|---------|
| `cluster_traces` | 500 traces | 42043 | 553 | **76.1×** |
| `homeostatic_scaling` | 200×200 | 25705 | 566 | **45.4×** |
| `cluster_notes` | 300 notes | 19471 | 1548 | **12.6×** |
| `build_summary_dag` | 100 traces | 1448 | 4525 | 0.3× (FFI overhead) |

Key findings: `cluster_traces` massive win from avoiding Python datetime
parsing. `homeostatic_scaling` eliminates numpy row-loop overhead.
`cluster_notes` benefits from HashSet intersection in Rust.
`build_summary_dag` loses due to FFI cost of constructing ~130 Python
dicts with nested lists — the actual computation is minimal.

## End-to-End Pipeline Benchmarks

Full pipeline exercising every major subsystem, ML model excluded
(regex/heuristic path only).

| Pipeline | Workload | Time |
|----------|---------|-----:|
| Regex pipeline | 47K chars | **0.29 ms** |
| Decompose + FactIndex + query | 50 turns (5 sessions) | **1.6 ms** |
| ArcaneRetriever (with recency decay) | 5 sessions, 50 turns | **6.0 ms** |
| Summary DAG build | 100 traces | **1.26 ms** |
| Summary DAG search | 100-trace DAG | **1.26 ms** |
| KnowledgeStore.add_note | per note | **0.18 ms** |
| KnowledgeStore.search | 50 notes | **0.01 ms** |
| Heartbeat | empty dirs | **26 ms** |
| **Full end-to-end** | retrieve + store + search | **6.6 ms** |
| Regex pipeline (large) | 470K chars | **7.2 ms** |

### Full end-to-end flow

```
Sessions (5 × 10 turns)
    │
    ▼
ArcaneRetriever(sessions, recency_decay=30d)
    ├── decompose_sessions() [Rust: classify_fact_type, split_sentences, has_change_verb]
    ├── FactIndex() [keyword + entity index]
    ├── _parallel_retrieve() [4 channels: BM25, entity, temporal, session]
    ├── _rrf_fusion() [with recency_weight per fact]
    │
    ▼
build_context() → LLM-ready context string
    │
    ▼
KnowledgeStore.add_note() × 5 [Rust: tokenize, extract_keywords, extract_entities]
    │
    ▼
KnowledgeStore.search() [token overlap scoring]
    │
    ▼
Total: 6.6 ms (50 conversation turns → ranked context + stored notes)
```

### Interpretation

- **6.6 ms end-to-end** = ~150 queries/second on a single core, no GPU/LLM
- Regex pipeline (0.29 ms on 47K) is dominated by `parse_dates` and
  `regex_entities` — both Rust-accelerated (2.9-8.5× over Python)
- ArcaneRetriever 4-channel parallel retrieval: 6.0 ms for 50 turns,
  including recency decay (0.5 µs per fact = negligible)
- KnowledgeStore: sub-millisecond add and search
- Heartbeat (26 ms): runs every ~5 minutes in background, cost amortised

## v0.4 Feature Performance

| Operation | Measured | Budget |
|-----------|---------|--------|
| `classify_fact` (9 types, Rust) | 0.20 µs | <500 µs (CI) |
| `_recency_weight` | 0.5 µs | <5 µs |
| SHA-256 hash (47K chars) | 14 µs | <500 µs |
| `capacity_report` (5 cats, 50 files) | 1.9 ms | <50 ms (CI) |
| `age_memories` (50 files) | 1.4 ms | <50 ms |
| `build_summary_dag` (100 traces) | 1.26 ms | <5 ms |
| `search_summary_dag` (100 traces) | 1.26 ms | <5 ms (CI) |
| `heartbeat` (empty) | 26 ms | <50 ms |
| Full v0.4 pipeline (50 turns) | 6.6 ms | <500 ms |

## Index Build Time

| Mode | Time | When to use |
|------|------|-------------|
| Full build (no embeddings) | 5-8s | First build or structure change |
| Incremental build (no changes) | 5-8s | Complete rebuild with hash-hit statistics |
| `add_file()` update | <5ms | Already-loaded index plus one changed file |
| Full build (with embeddings) | 10-30s | When using embedding rerank |

### Content-hash incremental builds

```python
idx = MemoryIndex()
stats = idx.build(incremental=True)
# stats["hash_hits"]   → files unchanged since the previous snapshot
# stats["hash_misses"] → files new or changed since the previous snapshot
```

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

Automatically activates at 50K+ paragraphs:

```bash
export REMANENTIA_USE_RUST_BM25=1
```

### Recency decay curve

| Age (days) | Weight (half_life=30) |
|------------|----------------------|
| 0 (today) | 1.000 |
| 15 | 0.707 |
| 30 | 0.500 |
| 60 | 0.250 |
| 90 | 0.125 |
| 365 | ~0.000 |

## SNN Performance

| Neurons | Rust | Python | Speedup |
|---------|------|--------|---------|
| 1,000 | 156ms | 453ms | 2.9× |
| 2,000 | 907ms | 1,422ms | 1.6× |
| 5,000 | 3,531ms | 7,000ms | 2.0× |

## Memory Usage

| Component | Memory |
|-----------|--------|
| Index (20K paragraphs) | ~50 MB |
| Embedding model (MiniLM) | ~90 MB |
| Cross-encoder model | ~90 MB |
| SNN (1K neurons) | ~4 MB |
| Content hash cache (2K files) | ~0.5 MB |
| Summary DAG (100 traces) | ~50 KB |

## Test Coverage

Canonical test and coverage evidence lives in the repository's `VALIDATION.md`;
point-in-time counts are not duplicated here. The CI coverage job enforces the
repository's 100% coverage gate. Performance
budget tests and benchmark checks should be rerun before publishing new latency
or throughput numbers.
