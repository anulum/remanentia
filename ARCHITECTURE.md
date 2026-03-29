# Architecture

Remanentia is a three-stage memory system: retrieval engine, consolidation
pipeline, and MCP server. No LLM required for core operations.

## Directory Map

```
remanentia/
├── memory_index.py         Unified BM25 + embedding index, all scoring/ranking
├── memory_recall.py        Deep recall: retrieval + graph + temporal context
├── mcp_server.py           Thread-safe MCP server (stdio JSON-RPC)
├── consolidation_engine.py Episodic → semantic compression, typed relations
├── knowledge_store.py      Zettelkasten atomic notes, prospective triggers
├── temporal_graph.py       Temporal event graph, relative date resolution
├── entity_extractor.py     GLiNER2 NER + regex fallback, 11 relation types
├── answer_extractor.py     Query-proximity answer extraction, LLM fallback
├── answer_normalizer.py    Hedging strip, yes/no polarity, semantic similarity
├── observer.py             Filesystem watcher → incremental index updates
├── reflector.py            Periodic cluster summarisation + gap detection
├── arcane_retriever.py     4-channel parallel retrieval with RRF fusion
├── fact_decomposer.py      Atomic fact decomposition with temporal validity
├── date_normalizer.py      C4: Rule-based + ML vague date normalisation
├── temporal_relation.py    C3: Temporal relation classifier (6 Allen-interval classes)
├── fact_validity_model.py  C5: Fact type + supersession detection model
├── cli.py                  Command-line interface
├── api.py                  FastAPI REST server
├── api_server.py           Lightweight HTTP API for cross-service integration
├── snn_backend.py          Dual-backend LIF network (GPU/CPU) with Rust STDP
├── snn_daemon.py           Persistent SNN background daemon
│
├── reasoning_traces/       Raw session decisions (episodic memory)
├── memory/
│   ├── semantic/           Consolidated facts with YAML frontmatter
│   ├── graph/              Entity-entity relations (JSONL)
│   └── knowledge_notes.jsonl  Zettelkasten atomic notes
├── snn_state/              SNN checkpoint, index pickle, weight snapshots
├── docs/                   Documentation (MkDocs)
└── tests/                  844 tests, 100% coverage
```

## Retrieval Pipeline

```
Query
  │
  ▼
Query Classification ──────────────── type, boost_types, recency, temporal
  │
  ▼
Query Decomposition ───────────────── multi-hop → sub-queries
  │
  ▼
BM25 Scoring ──────────────────────── inverted index, real TF-IDF
  │                                   (Rust: remanentia_search at 50K+)
  ▼
Entity Graph Boost ────────────────── 11 typed relations, person-centric gating
  │
  ▼
Bi-encoder Rerank ─────────────────── MiniLM-L6-v2, cosine similarity
  │
  ▼
Reciprocal Rank Fusion ────────────── scale-invariant BM25+embedding fusion
  │
  ▼
Cross-encoder Rerank ──────────────── ms-marco-MiniLM-L-6-v2 (background load)
  │
  ▼
Temporal Augmentation ─────────────── C4 date normaliser → event graph → TReMu
  │
  ▼
Answer Extraction ─────────────────── regex patterns + LLM fallback (optional)
  │
  ▼
Confidence Scoring ────────────────── normalised [0,1] with cross-reference
  │
  ▼
Knowledge Store ───────────────────── multi-hop graph search, prospective triggers
```

## Memory Types

| Type | Storage | Write Path | Read Path |
|------|---------|------------|-----------|
| Episodic | `reasoning_traces/*.md` | `remanentia_remember` | BM25 + embedding |
| Semantic | `memory/semantic/**/*.md` | `consolidation_engine` | BM25 + embedding |
| Graph | `memory/graph/*.jsonl` | `consolidation_engine` | Entity boost in retrieval |
| Knowledge | `memory/knowledge_notes.jsonl` | `knowledge_store.add_note()` | Multi-hop graph search |
| Procedural | `skills/*.json` | `skill_extractor` | Keyword lookup |

## Consolidation Pipeline

```
New reasoning trace written
  │
  ▼
Debounced trigger (10s) ───────────── background thread in mcp_server
  │
  ▼
Detect unprocessed traces ─────────── scan reasoning_traces/ vs processed set
  │
  ▼
Cluster by project + date gap ─────── >2 day gap = new cluster
  │
  ▼
Extract structured facts ──────────── heuristic, no LLM
  │
  ▼
Build/update entity graph ─────────── GLiNER2 or regex fallback
  │
  ▼
Detect conflicts ──────────────────── compare with existing semantic memories
  │
  ▼
Write semantic memory files ───────── YAML frontmatter + markdown
```

## Temporal Training Pipeline (C1–C5)

Five locally-trained models address the temporal-reasoning bottleneck
(45.9% → target 65–70% on LongMemEval).

```
haystack_dates[session_idx]
  │
  ▼
decompose_sessions(session_dates=...) ── C4 resolves "3 weeks ago" → ISO date
  │
  ▼
_build_fact(reference_date=...) ──────── C5 classifies type + supersession
  │
  ▼
FactIndex.temporal_query() ───────────── C3 boosts temporally relevant results
  │
  ▼
ArcaneRetriever._ch_temporal() ───────── C3 relation-aware score boost (1.3×)
  │
  ▼
memory_index (C1 + C2) ──────────────── Fine-tuned embedding + cross-encoder
```

| Component | Base Model | Role | Size |
|-----------|-----------|------|------|
| C1 | `all-MiniLM-L6-v2` fine-tuned | Temporal-aware bi-encoder embedding | 88 MB |
| C2 | `ms-marco-MiniLM-L-6-v2` fine-tuned | Temporal cross-encoder reranker | 88 MB |
| C3 | `bert-small` + 6-class head | Temporal relation classifier | 112 MB |
| C4 | `bert-mini` + 8-digit heads | Vague date → ISO date normaliser | 44 MB |
| C5 | `bert-mini` + 3 heads | Fact type + supersession detection | 44 MB |

C4 also includes a **rule-based engine** (12 patterns, 0.95 confidence) that
handles the majority of vague expressions without requiring the ML model.
All models degrade gracefully — missing checkpoints fall back to regex-only
or pre-trained HuggingFace defaults.

Training: 5 jobs in parallel on 5× AMD RX 6600 XT (ROCm 6.2), ~25 min wall time.
Checkpoints stored in `models/` (gitignored).

## SNN Substrate

A 1000-neuron LIF (Leaky Integrate-and-Fire) network with STDP runs as a
background daemon. Its role is consolidation orchestration and novelty
detection — not retrieval (validated negative result from 60+ experiments).

**Backends:**
- Rust (`arcane_stdp`): `stdp_batch` + `lif_step` via PyO3/rayon — 2–3x speedup
- GPU (PyTorch CUDA): 20,000 neurons, cuBLAS GEMV
- CPU (NumPy dense): vectorised outer product STDP

The SNN detects when new information is surprising relative to stored
patterns and triggers consolidation cycles. Complementary learning
(fast W + slow W) consolidates recurring patterns across traces.

## MCP Server

Stdio JSON-RPC server exposing four tools:

| Tool | Function |
|------|----------|
| `remanentia_recall` | Search memory (BM25 + embedding + knowledge store) |
| `remanentia_remember` | Persist a memory as reasoning trace |
| `remanentia_status` | System status summary |
| `remanentia_graph` | Entity relationship query |

Thread-safe with singleton index. Async consolidation on every write
(debounced). Compatible with Claude Code, Cursor, and any MCP client.

## Rust Acceleration

| Module | Function | Threshold | Speedup |
|--------|----------|-----------|---------|
| `remanentia_search` | BM25 scoring | 50K paragraphs | 2–4x |
| `arcane_stdp` | STDP + LIF step | Auto-detect | 2–3x |

Both modules are PyO3 0.25 with rayon parallelisation. Silent fallback
to Python if wheels not installed.
