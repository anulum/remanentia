# Architecture Overview

Remanentia is a three-stage memory system: retrieval engine, consolidation
pipeline, and MCP server. No LLM required for core operations.

## Module Map

| Module | Stmts | Role |
|--------|------:|------|
| `memory_index.py` | 888 | Unified BM25 + embedding index, query decomposition, RRF |
| `knowledge_store.py` | 414 | Zettelkasten notes, prospective queries, graph search |
| `consolidation_engine.py` | 499 | Episodic → semantic compression, typed relation extraction |
| `answer_extractor.py` | 306 | Regex answer extraction (dates, numbers, names), LLM fallback |
| `fact_decomposer.py` | 216 | Atomic fact decomposition with temporal validity windows |
| `memory_recall.py` | 207 | Deep recall: retrieval + graph + temporal context |
| `temporal_graph.py` | 341 | Temporal event graph, date resolution, code execution |
| `mcp_server.py` | 185 | Thread-safe MCP server (stdio JSON-RPC), async consolidation |
| `cli.py` | 173 | Command-line interface |
| `arcane_retriever.py` | 138 | 4-channel parallel retrieval with RRF fusion |
| `api_server.py` | 190 | Stdlib HTTP API for cross-service integration |
| `reflector.py` | 108 | Periodic cluster summarisation + gap detection |
| `api.py` | 224 | FastAPI REST server |
| `observer.py` | 86 | Filesystem watcher → incremental index updates |
| `entity_extractor.py` | 73 | GLiNER2 NER + regex fallback, 11 typed relations |
| `answer_normalizer.py` | 72 | Hedging strip, polarity match, semantic similarity |
| **Validation** | **current** | **2,143 passed, 3 skipped in the 2026-05-12 local full-suite run; CI coverage gate set to 100%** |

## Retrieval Pipeline

```
Query
  │
  ▼
Query Classification ──────────── type, boost_types, recency, temporal
  │
  ▼
Query Decomposition ───────────── multi-hop → sub-queries (optional)
  │
  ▼
BM25 Scoring ──────────────────── inverted index, real TF-IDF
  │                                (Rust: remanentia_search at 50K+)
  ▼
Entity Graph Boost ────────────── 11 typed relations, person-centric gating
  │
  ▼
Bi-encoder Rerank ─────────────── MiniLM-L6-v2, cosine similarity (optional)
  │
  ▼
Reciprocal Rank Fusion ────────── scale-invariant BM25+embedding fusion
  │
  ▼
Cross-encoder Rerank ──────────── ms-marco-MiniLM-L-6-v2 (optional)
  │
  ▼
Temporal Augmentation ─────────── date parsing, event graph, code execution
  │
  ▼
Answer Extraction ─────────────── regex patterns + LLM fallback (optional)
  │
  ▼
Confidence Scoring ────────────── normalised [0,1] with cross-reference
  │
  ▼
Knowledge Store ───────────────── multi-hop graph search, prospective triggers
```

## Memory Types

| Type | Storage | Write Path | Read Path |
|------|---------|------------|-----------|
| Episodic | `reasoning_traces/*.md` | `remanentia_remember` | BM25 + embedding |
| Semantic | `memory/semantic/**/*.md` | `consolidation_engine` | BM25 + embedding |
| Graph | `memory/graph/*.jsonl` | `consolidation_engine` | Entity graph boost |
| Knowledge | `knowledge_notes.jsonl` | `knowledge_store` | Multi-hop graph search |
| Temporal | In-memory graph | `temporal_graph` | Date arithmetic |

## Consolidation Pipeline

```
New reasoning trace written
  │
  ▼
Observer detects change ───────── filesystem poll or remember trigger
  │
  ▼
Text extraction ───────────────── paragraphs, entities, metadata
  │
  ▼
Clustering ────────────────────── by project + date proximity
  │
  ▼
Semantic memory write ─────────── YAML frontmatter, type, confidence
  │
  ▼
Graph update ──────────────────── entity nodes + typed relation edges
  │
  ▼
Knowledge store update ────────── atomic notes, contradiction detection
```

## Data Flow

All data lives in the filesystem. No database required.

- **Input**: markdown files in `reasoning_traces/`, any MCP `remember` call
- **Index**: in-memory BM25 inverted index + optional embedding vectors, persisted as pickle
- **Output**: search results with snippets, extracted answers, entity context

For full module-level reference, see the [API Reference](../api/memory_index.md) section.
