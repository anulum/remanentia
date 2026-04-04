# Rustification Plan

All compute-bound Python functions must have a Rust path. This document
tracks what exists, what is planned, and the priority order.

**Policy:** Rust is the default for any regex, scoring, graph traversal,
or numerical loop. Python fallback is kept for CI (which lacks the Rust
crates) and for development convenience.

**Build system:** maturin develop --release, crate sources in
`workspace-internal/rust_*/` (outside git repo, in project-workspace collection).

---

## Existing Rust crates (11)

| # | Crate | Python module(s) | Functions | Short-text speedup |
|---|-------|-------------------|-----------|-------------------|
| 1 | `remanentia_temporal` | temporal_graph, date_normalizer | parse_dates, normalise_vague_date, extract_temporal_events | 2.9x |
| 2 | `remanentia_answer_extractor` | answer_extractor | extract_answer, extract_best_sentence, fuzzy_match, normalize_number | ~1x short, 11.4x large |
| 3 | `remanentia_fact_decomposer` | fact_decomposer, arcane_retriever | classify_fact_type, has_change_verb, split_sentences, tokenize_words | 9.2x |
| 4 | `remanentia_answer_normalizer` | answer_normalizer | normalize_answer, extract_answer_items, answers_match | ~1x |
| 5 | `remanentia_search` | memory_index | BM25Index, classify_paragraph, cosine_batch, tokenize, split_paragraphs, token_counts | 3-5x at 50K+ |
| 6 | `arcane_stdp` | snn_backend, snn_daemon | stdp_batch, lif_step | 2-3x |
| 7 | `remanentia_entity_extractor` | entity_extractor | regex_entities, extract_relations | 8.5x |
| 8 | `remanentia_knowledge_store` | knowledge_store | extract_entities, extract_keywords, extract_person_names, tokenize | 3.5-4.6x |
| 9 | `remanentia_consolidation` | consolidation_engine | extract_entities, extract_key_lines, extract_typed_relations, parse_frontmatter | 8.3x |
| 10 | `remanentia_skill_extractor` | skill_extractor | tokenize_lower, matches_skill_marker, rank_skills_by_overlap | ~1x |
| 11 | `remanentia_active_retrieval` | active_retrieval | extract_decision_points | ~1x |

---

## Tier 1 — CRITICAL: retrieve.py hot-path (new crate: `remanentia_retrieve`)

Every function below runs on every query. The retrieval pipeline is the
single most latency-sensitive path in Remanentia.

| # | Function | File:line | What it does | Rust gain |
|---|----------|-----------|-------------|-----------|
| 1 | `_spike_feature()` | retrieve.py:416 | Deterministic LIF simulation, 50 steps x N neurons, matrix ops | Eliminate Python loop overhead on numpy, 5-20x expected |
| 2 | `_snn_affinity()` | retrieve.py:438 | 2x spike_feature + cosine similarity | Composites #1 |
| 3 | `_cosine_sim()` | retrieve.py:408 | Cosine similarity of two numpy arrays | Small win, batch-friendly |
| 4 | `_encode()` | retrieve.py:182 | Hash-encode text to neuron activation pattern (MD5 + primes) | Eliminate Python loop, MD5 in Rust is 3-10x faster |
| 5 | `_tokenize()` | retrieve.py:119 | Lowercase regex tokenizer with stopword filter | Regex in Rust, 2-8x |
| 6 | `_stem()` | retrieve.py:156 | Suffix-stripping stemmer, 25 suffixes | Tight loop, 5-10x |
| 7 | `_expand_query()` | retrieve.py:164 | Tokenize + stem + set diff | Composites #5 + #6 |
| 8 | `_bigrams()` | retrieve.py:178 | Adjacent token pairs | Trivial but hot |
| 9 | `_build_idf()` | retrieve.py:219 | IDF computation over all docs (tokenize + bigrams + Counter) | O(D x T), 3-5x |
| 10 | `_tfidf_score()` | retrieve.py:236 | TF-IDF with sublinear TF, bigrams, filename boost | Per-trace scoring, 3-5x |
| 11 | `_entity_graph_score()` | retrieve.py:702 | Weighted entity-graph matching between query and trace | Loop over relations, 2-5x |
| 12 | `_filename_bonus()` | retrieve.py:739 | IDF-weighted filename overlap | Simple scoring loop |
| 13 | `_reciprocal_rank_fusion()` | memory_index.py:1771 | RRF across ranked lists, O(n x m) | Dict aggregation + sort |

### Rust crate structure

```
workspace-internal/rust_retrieve/
  Cargo.toml          # pyo3, rayon, md5, regex
  src/
    lib.rs            # PyO3 module entry
    tokenize.rs       # tokenize, stem, expand_query, bigrams
    tfidf.rs          # build_idf, tfidf_score
    snn.rs            # spike_feature, snn_affinity, cosine_sim
    encode.rs         # hash_encode (MD5 + primes)
    fusion.rs         # reciprocal_rank_fusion
    graph.rs          # entity_graph_score, filename_bonus
```

### Exported Python API

```python
from remanentia_retrieve import (
    tokenize,           # (text: str, stopwords: set[str]) -> list[str]
    stem,               # (word: str) -> str
    expand_query,       # (query: str, stopwords: set[str]) -> str
    bigrams,            # (tokens: list[str]) -> list[str]
    build_idf,          # (docs: dict[str, str], stopwords: set[str]) -> dict[str, float]
    tfidf_score,        # (query, doc_name, doc_text, idf, stopwords) -> float
    spike_feature,      # (w: ndarray, stim: ndarray, steps: int) -> ndarray
    snn_affinity,       # (w: ndarray, q_stim: ndarray, t_stim: ndarray) -> float
    cosine_sim,         # (a: ndarray, b: ndarray) -> float
    hash_encode,        # (text: str, n_neurons: int, hash_primes: list[int], stopwords: set[str]) -> ndarray
    reciprocal_rank_fusion,  # (ranked_lists: list[list[tuple[int, float]]], k: int) -> list[tuple[int, float]]
    entity_graph_score, # (q_entities: list[str], t_entities: list[str], relations: list[tuple]) -> float
    filename_bonus,     # (query: str, name_lower: str, idf: dict[str, float], stopwords: set[str]) -> float
)
```

---

## Tier 2 — HIGH: index build + knowledge graph

Extend existing crates with new functions.

| # | Function | File:line | Target crate |
|---|----------|-----------|-------------|
| 14 | `FactIndex.query()` | fact_decomposer.py:172 | remanentia_fact_decomposer |
| 15 | `TemporalGraph.add_events()` | temporal_graph.py:108 | remanentia_temporal |
| 16 | `TemporalGraph.query_temporal()` | temporal_graph.py:178 | remanentia_temporal |
| 17 | `KnowledgeStore.search()` | knowledge_store.py:538 | remanentia_knowledge_store |
| 18 | `KnowledgeStore.get_related()` | knowledge_store.py:567 | remanentia_knowledge_store |
| 19 | `KnowledgeStore.graph_search()` | knowledge_store.py:601 | remanentia_knowledge_store |

---

## Tier 3 — MEDIUM: consolidation + reflection

| # | Function | File:line | Target crate |
|---|----------|-----------|-------------|
| 20 | `_cluster_traces()` | consolidation_engine.py:315 | remanentia_consolidation |
| 21 | `build_summary_dag()` | consolidation_engine.py:846 | remanentia_consolidation |
| 22 | `_cluster_notes()` | reflector.py:28 | remanentia_consolidation |
| 23 | `_homeostatic_scaling()` | snn_daemon.py:530 | arcane_stdp |

---

## Implementation checklist (per function)

1. Rust implementation with identical logic
2. PyO3 binding with numpy support
3. Wire into Python with `try: from crate import fn` + fallback
4. Test: Rust output == Python output (exact or within tolerance)
5. Benchmark: measure Rust vs Python speedup
6. Document speedup in PERFORMANCE_TUNING.md
7. pragma: no cover on Rust fast-path (CI has no crates)
