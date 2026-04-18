# memory_index

Unified BM25 + embedding index over all knowledge sources with
content-hash incremental builds.

## Purpose

The MemoryIndex is Remanentia's primary search engine. It scans all
configured knowledge sources (reasoning traces, semantic memories, code,
research documents, handovers), builds a BM25 inverted index with TF-IDF
scoring, and optionally adds GPU-accelerated embedding reranking via
sentence-transformers and cross-encoder models.

In v0.4, the index gained **content-hash incremental builds** inspired
by memsearch (Zilliz): files whose SHA-256 hash has not changed since
the last build are skipped entirely, reducing rebuild time by 50-90% on
large corpora.

## Architecture

```
SOURCES (18 directories)
        │
        ▼
  build(incremental=True)
    ├── Load previous content hashes from snn_state/content_hashes.json
    ├── For each source file:
    │     ├── Read text
    │     ├── SHA-256 hash
    │     ├── Skip if hash unchanged (hash_hit)
    │     ├── Split into paragraphs (_split_paragraphs)
    │     ├── Classify paragraph type (_classify_paragraph)
    │     ├── Generate prospective queries
    │     └── Build token counts + inverted index
    ├── Compute IDF
    ├── Build temporal graph
    ├── Save content hashes
    └── Optional: GPU embeddings + cross-encoder
        │
        ▼
  search(query, top_k)
    ├── Query classification (8 intent types)
    ├── BM25 scoring via inverted index
    ├── Optional: bi-encoder rerank
    ├── Reciprocal Rank Fusion (BM25 + embedding scores)
    ├── Optional: cross-encoder rerank
    ├── Entity graph boost
    ├── Temporal graph query
    └── Answer extraction (regex + optional LLM)
```

## Content-Hash Incremental Builds

### Mechanism

On every build, each file's content is hashed with SHA-256. The hash is
compared against the stored hash from the previous build:

```
File read → SHA-256(content) → compare with stored hash
    │
    ├── Match → skip file (hash_hit)
    └── Mismatch / new → index file (hash_miss)
```

After the build completes, all hashes are saved to
`snn_state/content_hashes.json` for the next build.

### Hash cache format

```json
{
    "/path/to/reasoning_traces/2024-03-15_decision.md": "a1b2c3d4...",
    "/path/to/memory/semantic/decision/2024-03-15_retrieval.md": "e5f6g7h8..."
}
```

### Build stats

The build return value includes hash statistics:

```python
stats = idx.build(incremental=True)
# stats["hash_hits"]   → files skipped (unchanged)
# stats["hash_misses"] → files indexed (new or changed)
```

### Performance impact

| Corpus size | Full build | Incremental (no changes) | Speedup |
|-------------|-----------|-------------------------|---------|
| 100 files | 0.5s | 0.05s | 10× |
| 1,000 files | 5.0s | 0.3s | 17× |
| 15,000 paragraphs | 8.0s | 0.5s | 16× |

The hash check itself costs ~0.01ms per file (SHA-256 of in-memory string).

### Disabling incremental builds

```python
stats = idx.build(incremental=False)  # forces full rebuild
```

This is useful when the index structure changes (new source directories,
changed paragraph splitting logic) and all files need reprocessing.

## Knowledge Sources

The index scans 18 configured source directories:

| Source | Path | Content |
|--------|------|---------|
| traces | `reasoning_traces/` | Raw session decisions |
| paper | `paper/` | Research papers and drafts |
| semantic | `memory/semantic/` | Consolidated semantic memories |
| disposition | `disposition/` | Agent disposition and identity |
| sessions_as | `.coordination/sessions/arcane-sapience/` | Session states |
| handovers_as | `.coordination/handovers/arcane-sapience/` | Handovers |
| qc_research | `.coordination/handovers/scpn-quantum-control/` | Quantum research |
| po_research | `.coordination/handovers/scpn-phase-orchestrator/` | Phase orchestrator |
| nc_research | `sc-neurocore/docs/internal/` | SC-NeuroCore internals |
| claude_memory | `~/.claude/projects/.../memory/` | Claude persistent memory |
| indexer | `INDEXER/` | Catalog files |
| code_remanentia | `.` (this repo) | Python source code |
| code_orchestrator | `scpn-phase-orchestrator/src/` | Orchestrator code |
| code_quantum | `scpn-quantum-control/src/` | Quantum control code |
| code_neurocore | `sc-neurocore/src/` | SC-NeuroCore code |
| code_director | `DIRECTOR_AI/src/` | Director-AI code |

Code sources index `.py` and `.rs` files. All other sources index `.md`
files by default.

## BM25 Scoring

Real TF-IDF with inverted index. Each paragraph is tokenised, and
token frequencies are stored per-paragraph for BM25 scoring:

```
score(q, p) = Σ IDF(t) × (tf(t,p) × (k1+1)) / (tf(t,p) + k1 × (1 - b + b × |p|/avgdl))
```

With k1=1.5, b=0.75 (standard BM25 parameters).

### Rust BM25

When the `remanentia_search` Rust crate is installed and the index exceeds
`RUST_BM25_MIN_PARAGRAPHS` (50,000), the BM25 scoring delegates to the
Rust engine for ~3-5× speedup on large workloads. Below this threshold,
the Python implementation is used (no FFI overhead).

## Search Pipeline

8 query intent types determine the search strategy:

| Intent | Strategy |
|--------|----------|
| general | BM25 + entity graph boost |
| decision | BM25 + graph boost + key-line preference |
| temporal | BM25 + temporal graph + date extraction |
| debugging | BM25 + code paragraph boost |
| explanation | BM25 + graph boost |
| comparison | BM25 + entity co-occurrence |
| counting | BM25 + temporal range |
| identity | BM25 + entity exact match |

## Incremental File Addition

`add_file(path, source)` adds a single file to the running index without
rebuilding. Used by the observer for live updates:

1. Read and split into paragraphs
2. Tokenise and compute term frequencies
3. Update inverted index and IDF
4. Update paragraph length statistics
5. Compute embeddings for new paragraphs (if model loaded)

## Persistence

### Index file

`snn_state/memory_index.pkl` — serialised MemoryIndex with quantised
embeddings (int8, ~4× smaller than float32).

### Hash cache

`snn_state/content_hashes.json` — SHA-256 hashes of all indexed files
for incremental builds.

## Performance

Measured on the project-workspace corpus (15,938 paragraphs, 1,217 documents):

| Operation | Time | Budget |
|-----------|------|--------|
| Full build (no embeddings) | 5-8s | 30s |
| Incremental build (no changes) | 0.3-0.5s | 5s |
| BM25 search (warm) | <10ms | 50ms |
| BM25 + cross-encoder rerank | 50-200ms | 500ms |
| `add_file()` single file | <5ms | 50ms |
| SHA-256 hash per file | <0.01ms | negligible |

### Rust acceleration

| Component | Python | Rust | Speedup |
|-----------|--------|------|---------|
| BM25 scoring (50K+ paragraphs) | 15ms | 3-5ms | ~3-5× |
| Below 50K paragraphs | 8ms | N/A (Python used) | — |

## Data Structures

### Document

```python
@dataclass
class Document:
    name: str           # Filename
    source: str         # Source category (traces, semantic, code_*)
    path: str           # Full filesystem path
    paragraphs: list[str]  # Extracted paragraphs
    tokens: set[str]    # All unique tokens in document
    date: str           # Extracted date (ISO or "")
    doc_type: str       # "code" or source category
```

### SearchResult

```python
@dataclass
class SearchResult:
    name: str           # Document name
    source: str         # Source category
    snippet: str        # Matching paragraph
    score: float        # BM25 + reranking score
    answer: str         # Extracted answer (if any)
    date: str           # Document date
    doc_type: str       # Document type
    paragraph_type: str # function, decision, finding, etc.
```

## Usage

```python
from memory_index import MemoryIndex

idx = MemoryIndex()

# Incremental build (skips unchanged files)
stats = idx.build(use_gpu_embeddings=False, incremental=True)
print(f"Indexed {stats['documents']} docs, "
      f"skipped {stats['hash_hits']} unchanged, "
      f"processed {stats['hash_misses']} new/changed")

# Search
results = idx.search("what did we decide about authentication", top_k=5)
for r in results:
    print(f"{r.name} (score={r.score:.3f})")
    if r.answer:
        print(f"  Answer: {r.answer}")

# Force full rebuild
stats = idx.build(incremental=False)

# Save/load
idx.save()
idx2 = MemoryIndex()
idx2.load()
```

## Test Coverage

Tests in `tests/test_memory_index.py` (relevant to new features):

- **Hash cache**: save/load roundtrip, nonexistent path, corrupt JSON
- **Incremental build**: first build (all misses), second build (all hits),
  changed file detected, non-incremental ignores cache
- **Build stats**: hash_hits and hash_misses in return value

Plus all existing tests for BM25, search, reranking, incremental add,
temporal graph integration, etc. (200+ tests total in the file).

All 6 STRONG dimensions: empty, error, negative, pipeline, roundtrip, performance.

## API Reference

::: memory_index.MemoryIndex
    options:
      show_source: true
      members_order: source

::: memory_index.SearchResult

::: memory_index.Document
