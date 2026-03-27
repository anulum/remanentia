# Getting Started

## Installation

```bash
# Core (numpy only)
pip install -e .

# With embedding rerank
pip install -e ".[embedding]"

# With REST API
pip install -e ".[api]"

# Everything
pip install -e ".[all]"

# Development
pip install -e ".[dev]"
```

## Directory Setup

```bash
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

Add to your `.mcp.json` (Claude Code, Cursor, or any MCP client):

```json
{
  "mcpServers": {
    "remanentia": {
      "command": "python",
      "args": ["path/to/mcp_server.py"]
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

## CLI Reference

```bash
remanentia search "query"              # Search memory
remanentia recall "query" --top-k 10   # Deep recall with context
remanentia status                       # System status
remanentia consolidate --force          # Run consolidation
remanentia init                         # Create directory structure
```

## REST API

```bash
pip install -e ".[api]"
remanentia serve --port 8001
```

Endpoints:

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/recall?q=query` | Search memory |
| POST | `/remember` | Store memory |
| GET | `/status` | System status |
| GET | `/graph?entity=name` | Entity graph |

## Next Steps

- [User Manual](USER_MANUAL.md) — full feature reference
- [Integration Guide](INTEGRATION_GUIDE.md) — MCP, REST, Python API
- [Performance Tuning](PERFORMANCE_TUNING.md) — index configuration, Rust acceleration
