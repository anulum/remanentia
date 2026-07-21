# Choose an Interface

All interfaces use the same underlying memory components. Choose the smallest
surface that fits the deployment.

| Interface | Best for | Process boundary | Extra dependency |
|---|---|---:|---|
| CLI | Exploration, scripts, operators | New process per command | None for core search |
| Python | In-process applications and custom pipelines | None | None for core search |
| MCP | Desktop and agent-tool integration | Stdio child process | MCP client |
| FastAPI | Shared services and language-neutral clients | HTTP service | `remanentia[api]` |
| Lightweight HTTP | Constrained local integration | HTTP service | Python standard library |

## CLI

```bash
remanentia search "authentication decision"
remanentia recall "authentication decision" --format json
remanentia status
```

Use the CLI for the first evaluation and for human-operated maintenance. Run
`remanentia --help` and `remanentia <command> --help` for the installed version's
canonical options.

## Python

```python
from memory_index import MemoryIndex

index = MemoryIndex()
if not index.load():
    index.build(use_gpu_embeddings=False, use_gliner=False)
    index.save()

for result in index.search("authentication decision", top_k=3):
    print(result.name, result.score, result.snippet)
```

Use Python when retrieval belongs inside an existing process or when you need
custom ranking, filtering, or result handling. The [API guide](../api-guide.md)
links to the public modules.

## MCP

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

Use MCP when an AI client already supports tool servers. Keep stdio reserved for
JSON-RPC; logs belong on stderr or in the configured audit file. See the
[Agent Memory Tutorial](../tutorials/agent-memory.md).

## FastAPI

```bash
pip install "remanentia[api]"
remanentia serve --host 127.0.0.1 --port 8001
curl http://127.0.0.1:8001/health
```

Use FastAPI for shared or language-neutral access. Bind to loopback for local
use. For any broader exposure, require authentication, terminate TLS at a trusted
proxy, and follow the [Integration Guide](INTEGRATION_GUIDE.md).

## Lightweight HTTP

```bash
python -m api_server --host 127.0.0.1 --port 8001
```

Use the lightweight server only when avoiding FastAPI matters. Check its route
set before substituting it for the FastAPI application; the two surfaces are not
promised to be identical.
