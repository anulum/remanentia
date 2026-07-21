# API Guide

Remanentia exposes a Python API, a FastAPI HTTP service, and an MCP tool server.
This page is the stable entry point; the module pages in this section are
generated from current docstrings.

## Python retrieval

Select the store before importing Remanentia modules:

```bash
export REMANENTIA_BASE="/absolute/path/to/remanentia-store"
```

```python
from memory_index import MemoryIndex

index = MemoryIndex()
if not index.load():
    index.build(use_gpu_embeddings=False, use_gliner=False)
    index.save()

results = index.search("what changed in the deployment plan", top_k=5)
for result in results:
    print(result.name, result.score)
    print(result.snippet)
```

Start with [`memory_index`](api/memory_index.md) for direct search or
[`memory_recall`](api/memory_recall.md) for recall enriched with graph and
temporal context. See [`knowledge_store`](api/knowledge_store.md) for atomic
notes and links.

## FastAPI service

```bash
pip install "remanentia[api]"
export REMANENTIA_BASE="/absolute/path/to/remanentia-store"
remanentia serve --host 127.0.0.1 --port 8001
```

```bash
curl -sS http://127.0.0.1:8001/health
curl -sS -X POST http://127.0.0.1:8001/recall \
  -H 'Content-Type: application/json' \
  -d '{"query":"deployment plan","top_k":3}'
```

The FastAPI application currently exposes:

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness and dependency status |
| `POST` | `/recall` | Retrieve relevant memory |
| `POST` | `/recall/correctness` | Record a correctness outcome |
| `POST` | `/vector/search/public` | Public vector-search contract |
| `POST` | `/consolidate` | Run consolidation |
| `GET` | `/status` | Memory-system status |
| `GET` | `/entities` | List known entities |
| `GET` | `/graph` | List top graph relations |
| `GET` | `/graph/entity/{entity_id}` | Inspect one entity's connections |

Interactive Swagger documentation is available at `/docs` while the service is
running. Export the versioned schema without starting a server:

```bash
remanentia openapi --output docs/openapi/remanentia_openapi.json
```

Read [`api`](api/api.md) for request and response models and
[`api_security`](api/api_security.md) before exposing the service beyond loopback.

## MCP tools

Run `python -m mcp_server` and connect over stdio. The server advertises six
tools:

| Tool | Purpose |
|---|---|
| `remanentia_recall` | Retrieve source-grounded context |
| `remanentia_remember` | Store a memory note |
| `remanentia_status` | Inspect memory state |
| `remanentia_graph` | Query entity relationships |
| `remanentia_recall_feedback` | Record usefulness feedback |
| `remanentia_recall_correctness` | Record a correctness-labelled outcome |

See [`mcp_server`](api/mcp_server.md) for rate limits, auditing, and runtime options.

## Errors and compatibility

- Treat response fields documented by request/response models as the contract;
  do not parse human-readable CLI output as an API.
- Pin the package version in production and regenerate the OpenAPI schema when
  upgrading.
- Empty recall queries fail closed rather than triggering unconstrained search.
- Authentication and transport security are deployment responsibilities; use
  the shipped auth controls and a trusted TLS boundary.
