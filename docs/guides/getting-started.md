# Getting Started

## Installation

```bash
# Core (numpy only)
pip install remanentia

# With embedding rerank
pip install "remanentia[embedding]"

# With REST API
pip install "remanentia[api]"

# Everything
pip install "remanentia[all]"

# Development (from source)
pip install -e ".[dev]"
```

## Directory Setup

```bash
export REMANENTIA_BASE="$PWD/.remanentia-data"
remanentia init
```

This creates the directory structure:

```
reasoning_traces/     # Write your session notes here
memory/
  semantic/           # Auto-populated by consolidation
  graph/              # Entity relations (auto-populated)
snn_state/            # SNN checkpoints (auto-populated)
skills/               # Extracted skills (auto-populated)
```

## First Search

```bash
# Build the index from your reasoning traces
remanentia search "what did we decide about authentication"
```

On first run, the index builds automatically from all configured sources.
Subsequent searches use the cached index (~50ms).

## MCP Integration

Add to your `.mcp.json` (Cursor or any MCP-compatible client):

```json
{
  "mcpServers": {
    "remanentia": {
      "command": "/absolute/path/to/.venv/bin/python",
      "args": ["-m", "mcp_server"],
      "env": {"REMANENTIA_BASE": "/absolute/path/to/remanentia-store"}
    }
  }
}
```

Available tools:

| Tool | Purpose |
|------|---------|
| `remanentia_recall` | Search memory with filters |
| `remanentia_remember` | Store a new memory |
| `remanentia_status` | System statistics |
| `remanentia_graph` | Query entity relationships |
| `remanentia_recall_feedback` | Record retrieval usefulness feedback |
| `remanentia_recall_correctness` | Record a correctness-labelled outcome |

## CLI Reference

```bash
remanentia search "query"              # Search memory
remanentia recall "query" --top 10     # Deep recall with context
remanentia status                       # System status
remanentia consolidate --force          # Run consolidation
remanentia init                         # Create directory structure
```

## REST API (FastAPI)

```bash
pip install -e ".[api]"
remanentia serve --host 127.0.0.1 --port 8001
```

Core endpoints (the running service also publishes `/docs` and `/openapi.json`):

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/recall` | Search memory (JSON body: `{"query": "...", "top_k": 3}`) |
| POST | `/recall/correctness` | Record a labelled correctness outcome |
| POST | `/vector/search/public` | Search through the public vector contract |
| POST | `/consolidate` | Run consolidation |
| GET | `/status` | System status |
| GET | `/health` | Health check |
| GET | `/entities` | List all entities |
| GET | `/graph?top=15` | Top entity relationships |
| GET | `/graph/entity/{id}` | Entity detail with connections |

## HTTP API (stdlib, no dependencies)

```bash
python -m api_server --host 127.0.0.1 --port 8001
```

Lighter alternative with no FastAPI dependency. It exposes `health`, `recall`,
`status`, `consolidate`, and `remember`; do not assume full FastAPI route parity.

## Container Deployment

```bash
mkdir -p secrets
printf '%s\n' '<replace-with-a-random-token>' > secrets/remanentia_api_token
docker compose up --build
```

The Compose deployment binds `127.0.0.1:8001`, stores memory under the
`remanentia-data` volume, loads the bearer token from a Docker secret, and uses
`GET /health` as the container healthcheck.

## Next Steps

- [User Manual](USER_MANUAL.md) — full feature reference
- [Integration Guide](INTEGRATION_GUIDE.md) — MCP, REST, Python API
- [Choose an Interface](choose-an-interface.md) — compare CLI, Python, MCP, and HTTP
- [API Guide](../api-guide.md) — requests, routes, tools, and compatibility
- [Performance Tuning](PERFORMANCE_TUNING.md) — index configuration, Rust acceleration
