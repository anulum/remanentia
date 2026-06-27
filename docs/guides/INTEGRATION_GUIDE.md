# Integration Guide

## MCP Server (Cursor and other MCP-compatible clients)

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

Set `llm: true` to enable hosted-LLM answer synthesis (requires the
hosted-provider API key in the environment — the default backend
reads `ANTHROPIC_API_KEY`).

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
  "top": 10
}
```

Parameters: `entity` (filter by entity name, optional), `top` (max relations to return, default 10).

## REST API (FastAPI)

```bash
pip install -e ".[api]"
remanentia serve --host 127.0.0.1 --port 8001
```

Set `REMANENTIA_API_TOKEN` for private endpoints. Production launchers can fail
closed and load the token from a local secret file:

```bash
remanentia serve --require-auth --token-file /run/secrets/remanentia_api_token
```

### Endpoints

```python
import httpx

# Search (POST, not GET)
r = httpx.post("http://localhost:8001/recall", json={
    "query": "STDP learning",
    "top_k": 5,
    "format": "summary",
})

# Consolidate
r = httpx.post("http://localhost:8001/consolidate", json={"force": False})

# Status
r = httpx.get("http://localhost:8001/status")

# Entities
r = httpx.get("http://localhost:8001/entities")

# Graph (top relationships)
r = httpx.get("http://localhost:8001/graph", params={"top": 15})

# Entity detail
r = httpx.get("http://localhost:8001/graph/entity/stdp")
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
| `REMANENTIA_API_TOKEN` | unset | Bearer token required by private REST endpoints when configured |
| `REMANENTIA_API_BODY_LIMIT_BYTES` | `1048576` | Maximum declared REST request body size |
| `REMANENTIA_API_RATE_PER_MINUTE` | `60` | Sustained REST API requests per minute per client |
| `REMANENTIA_API_RATE_BURST` | `10` | REST API request burst before throttling |
| `REMANENTIA_MCP_RATE_LIMIT` | enabled | Set to `off`, `0`, `false`, or `no` to disable MCP tool-call throttling |
| `REMANENTIA_MCP_RATE` | `600.0` | Sustained MCP `tools/call` requests per minute |
| `REMANENTIA_MCP_BURST` | `120` | MCP `tools/call` burst size before throttling |
| `REMANENTIA_MCP_CLIENT_ID` | `stdio` | Token-bucket key for this MCP client/session |
| `ANTHROPIC_API_KEY` | unset | Required for LLM features |
