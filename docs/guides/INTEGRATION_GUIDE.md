# Integration Guide

## MCP Server (Claude Code, Cursor)

### Configuration

Add to `.mcp.json` in your project root:

```json
{
  "mcpServers": {
    "remanentia": {
      "command": "python",
      "args": ["/path/to/remanentia/mcp_server.py"],
      "env": {
        "REMANENTIA_BASE": "/path/to/remanentia"
      }
    }
  }
}
```

### Tools

#### remanentia_recall

```json
{
  "query": "what did we decide about authentication",
  "top_k": 5,
  "project": "my-project",
  "after": "2026-03-01",
  "llm": false
}
```

Set `llm: true` to enable Anthropic API answer synthesis (requires
`ANTHROPIC_API_KEY` environment variable).

#### remanentia_remember

```json
{
  "content": "Switched from JWT to session tokens for auth",
  "memory_type": "decision",
  "project": "my-project",
  "trigger": "authentication token"
}
```

The `trigger` parameter enables prospective memory — future queries
matching the trigger will automatically surface this memory.

#### remanentia_status

No parameters. Returns index statistics, memory counts, SNN state.

#### remanentia_graph

```json
{
  "entity": "STDP",
  "relation_type": "depends_on",
  "hops": 2
}
```

## REST API (FastAPI)

```bash
pip install -e ".[api]"
remanentia serve --port 8001
```

### Endpoints

```python
import httpx

# Search
r = httpx.get("http://localhost:8001/recall", params={"q": "STDP learning"})

# Remember
r = httpx.post("http://localhost:8001/remember", json={
    "content": "Fixed the STDP weight update bug",
    "memory_type": "fix",
    "project": "remanentia"
})

# Status
r = httpx.get("http://localhost:8001/status")

# Graph
r = httpx.get("http://localhost:8001/graph", params={"entity": "STDP"})
```

## Python API

```python
from memory_index import MemoryIndex

# Build or load index
idx = MemoryIndex()
if not idx.load():
    idx.build()
    idx.save()

# Search
results = idx.search("STDP learning rule", top_k=5)
for r in results:
    print(f"{r.name} (score={r.score:.2f}): {r.answer or r.snippet[:100]}")

# Incremental add
from pathlib import Path
idx.add_file(Path("reasoning_traces/new_trace.md"))
idx.save()
```

## Knowledge Store

```python
from knowledge_store import KnowledgeStore

ks = KnowledgeStore()
ks.load()

# Add a note
ks.add_note("STDP learning rule was fixed in v0.3.0", source="session")

# Graph search
notes = ks.graph_search("STDP", top_k=5, hop_depth=2)

# Prospective trigger
ks.add_trigger("authentication", "We switched to session tokens")
ks.save()
```

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `REMANENTIA_BASE` | Script directory | Base path for all data |
| `REMANENTIA_USE_RUST_BM25` | `0` | Force Rust BM25 (`1` to enable) |
| `REMANENTIA_LLM_ANSWERS` | unset | Enable LLM answer synthesis |
| `ANTHROPIC_API_KEY` | unset | Required for LLM features |
