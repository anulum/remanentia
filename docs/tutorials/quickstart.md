# Tutorial: First Search in 5 Minutes

This tutorial walks through installing Remanentia, creating your first memory trace, building the index, and running a search query.

## Prerequisites

- Python 3.10 or later
- numpy (installed automatically)

## Step 1: Install

```bash
pip install -e .
```

Or for all optional features (embeddings, REST API, LLM):

```bash
pip install -e ".[all]"
```

## Step 2: Create Directory Structure

```bash
remanentia init
```

This creates:

```
reasoning_traces/     # Your session notes go here
memory/
  semantic/           # Auto-populated by consolidation
  graph/              # Entity relations (auto-populated)
snn_state/            # SNN checkpoints (auto-populated)
```

## Step 3: Write a Reasoning Trace

Create a markdown file in `reasoning_traces/`:

```bash
cat > reasoning_traces/2026-03-29_first_trace.md << 'EOF'
# Decision: Use BM25 for retrieval

## Context

We evaluated three retrieval approaches:
1. TF-IDF with cosine similarity
2. BM25 with real term frequency
3. Dense embedding retrieval (MiniLM)

## Decision

BM25 was chosen because it handles term frequency saturation
naturally (the k1 parameter) and doesn't require GPU inference.

## Result

P@1 improved from 71% to 85% on our internal benchmark.
EOF
```

## Step 4: Search

```bash
remanentia search "which retrieval approach did we choose"
```

On first run, the index builds automatically from all files in the configured source directories. Subsequent searches reuse the cached index.

## Step 5: Search with Python

```python
from memory_index import MemoryIndex

idx = MemoryIndex()
if not idx.load():
    idx.build(use_gpu_embeddings=False, use_gliner=False)
    idx.save()

results = idx.search("BM25 retrieval decision", top_k=3)
for r in results:
    print(f"{r.name} (score={r.score:.3f})")
    if r.answer:
        print(f"  Answer: {r.answer}")
    print(f"  {r.snippet[:150]}")
```

`MemoryIndex.search()` returns a list of `SearchResult` objects:

| Field | Type | Description |
|-------|------|-------------|
| `name` | str | Document filename |
| `source` | str | Source directory name |
| `score` | float | BM25 relevance score |
| `snippet` | str | Matching paragraph text |
| `answer` | str | Extracted answer (if found) |
| `confidence` | float | 0.0–1.0 confidence score |

## Step 6: MCP Integration

Add to `.mcp.json` in your project:

```json
{
  "mcpServers": {
    "remanentia": {
      "command": "python",
      "args": ["path/to/remanentia/mcp_server.py"]
    }
  }
}
```

Now any MCP-compatible tool (Cursor and others) can search your memory via `remanentia_recall`.

## Next Steps

- [Knowledge Store Tutorial](knowledge_store_tutorial.md) — atomic notes, graph search, triggers
- [Temporal Queries Tutorial](temporal_tutorial.md) — date reasoning, event ordering
- [User Manual](../guides/USER_MANUAL.md) — full feature reference
