# memory_index

Unified BM25 + embedding index over all knowledge sources with
content-hash incremental builds.

## Purpose

The MemoryIndex is Remanentia's primary search engine. It scans all
configured knowledge sources (reasoning traces, semantic memories, code,
research documents, handovers), builds a BM25 inverted index with TF-IDF
scoring, and optionally adds GPU-accelerated embedding reranking via
sentence-transformers and cross-encoder models.

The index tracks SHA-256 content hashes for rebuild observability and
watcher-driven update decisions. `build(incremental=True)` reports
unchanged files as hash hits while still materializing a complete
searchable in-memory index; use `add_file()` on an already-loaded index
for true single-file incremental updates.

## Architecture

```
Configured source roots
        │
        ▼
  build(incremental=True)
    ├── Load previous content hashes from snn_state/content_hashes.json
    ├── For each source file:
    │     ├── Read text
    │     ├── SHA-256 hash
    │     ├── Record unchanged file as hash_hit
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
    ├── Match → record hash_hit
    └── Mismatch / new → record hash_miss
```

After the build completes, all hashes are saved to
`snn_state/content_hashes.json` for the next build. The file is always
indexed during a fresh `build()` so a hash hit cannot erase memory from
the current process.

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
# stats["hash_hits"]   → files unchanged since the previous hash snapshot
# stats["hash_misses"] → files new or changed since the previous snapshot
```

### Performance impact

The hash check itself costs ~0.01ms per file (SHA-256 of in-memory string).
Search latency is reduced by a lazy Python BM25 term-weight cache, which
precomputes each posting's normalized BM25 contribution once per index
version and reuses it across repeated queries.

### Disabling incremental builds

```python
stats = idx.build(incremental=False)  # forces full rebuild
```

This is useful when the index structure changes (new source directories,
changed paragraph splitting logic) and all files need reprocessing.

## Knowledge Sources

The index is organized around named source roots loaded by
`memory_sources.load_source_config(BASE)` when `memory_index.py` is imported.
The public defaults are repository-local and neutral. Deployments add external
repositories, shared archives, or application data with JSON configuration
instead of editing `memory_index.py`.

| Source | Path | Content |
|--------|------|---------|
| traces | `reasoning_traces/` | Raw session decisions |
| paper | `paper/` | Research papers and drafts |
| semantic | `memory/semantic/` | Consolidated semantic memories |
| compiled | `memory/compiled/` | Built memory snapshots |
| code_remanentia | `.` (this repo) | Python source code |

Configured sources use operator-defined labels such as `decision_archive` or
`external_code`. Relative paths in a config file resolve from that file's
directory; relative paths in inline JSON resolve from the repository root.
Sources without an explicit extension list use text-document suffixes
(`.md`, `.txt`, `.json`, `.jsonl`, `.yaml`, `.yml`).

### Source configuration

`REMANENTIA_MEMORY_SOURCES_CONFIG` points at a JSON file:

```json
{
  "sources": {
    "decision_archive": {
      "path": "../archives/decisions",
      "extensions": [".md", ".jsonl"]
    },
    "external_code": {
      "path": "/srv/project/src",
      "extensions": [".py", ".rs"]
    }
  }
}
```

`REMANENTIA_MEMORY_SOURCES_JSON` accepts the same JSON inline. Set
`"extends_defaults": false` to replace the default roots with only configured
sources.

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

Multi-hop prompts are decomposed into bounded sub-queries, and each
sub-query runs through the same non-decomposing search stack as a normal
query: compiled facts, filters, BM25/Rust scoring, graph boosts,
embedding rerank, answer extraction, and confidence handling. Results
are then deduplicated and reranked against the original prompt.

Entity graph data is cached in-process, keyed by the graph file paths,
sizes, and mtimes for `entities.jsonl` and `relations.jsonl`. If the
consolidation pipeline updates either file, the next search reloads the
graph before applying boosts. Typed relation boosts use a precomputed
entity-neighbor adjacency map, so scoring only visits relations adjacent
to query entities instead of scanning the full relation list for every
candidate paragraph.

## Incremental File Addition

`add_file(path, source)` adds a single file to the running index without
rebuilding. Used by the observer for live updates:

1. Read and split into paragraphs
2. Replace any existing document with the same filesystem path
3. Tokenise and compute term frequencies
4. Update inverted index, IDF, and paragraph length statistics
5. Invalidate cached BM25/Rust state
6. Compute embeddings for new paragraphs if the model is loaded; if an
   existing path was replaced, rebuild embeddings or drop the stale matrix

## Persistence

### Index file

`snn_state/memory_index.json.gz` — serialized MemoryIndex metadata.

`snn_state/memory_index_embeddings.npz` — optional embedding sidecar,
quantized to int8 by default (~4× smaller than float32). Loading validates
that the sidecar is two-dimensional, finite, and row-aligned with the
paragraph index. If the sidecar is stale or corrupt, the sparse BM25 index
still loads and embedding rerank is disabled for that process.

### Hash cache

`snn_state/content_hashes.json` — SHA-256 hashes of all indexed files
for incremental builds.

## Performance

Measured on the internal calibration corpus (15,938 paragraphs, 1,217 documents):

| Operation | Time | Budget |
|-----------|------|--------|
| Full build (no embeddings) | 5-8s | 30s |
| Incremental build (no changes) | 5-8s | 30s |
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

# Incremental build with hash-hit statistics
stats = idx.build(use_gpu_embeddings=False, incremental=True)
print(f"Indexed {stats['documents']} docs, "
      f"unchanged {stats['hash_hits']}, "
      f"new/changed {stats['hash_misses']}")

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
- **Incremental build**: first build (all misses), second build preserves
  searchable unchanged documents, changed file detected, non-incremental ignores cache
- **Build stats**: hash_hits and hash_misses in return value
- **Configured source roots**: `tests/test_memory_sources.py` validates neutral
  defaults, JSON/env configuration, validation errors, and production
  `MemoryIndex.build()` wiring

Plus all existing tests for BM25, search, reranking, incremental add,
temporal graph integration, etc. (200+ tests total in the file).

Test dimensions: empty, error, negative, pipeline, roundtrip, and performance.

## API Reference

::: memory_index.MemoryIndex
    options:
      show_source: true
      members_order: source

::: memory_index.SearchResult

::: memory_index.Document
