# Architecture Overview

Remanentia combines a filesystem-backed memory store, hybrid retrieval,
consolidation, graph and temporal context, and several integration surfaces.
The core path does not require an LLM or managed vector database.

## System boundaries

```text
Markdown traces / semantic notes / graph artefacts
                     |
                     v
          canonical selected store
                     |
        +------------+-------------+
        |                          |
        v                          v
  MemoryIndex                 consolidation
  BM25 + optional             trace -> semantic memory
  semantic ranking            graph + conflict records
        |
        v
  recall enrichment
  graph + temporal context
        |
        +----------+-----------+------------+
                   |           |            |
                  CLI        MCP stdio   HTTP / Python
```

`REMANENTIA_BASE` selects the durable store for runtime surfaces. The
`store_paths` module derives trace, semantic-memory, graph, state,
consolidation, and vector-index paths from that root. Package resources such as
schemas and optional model metadata remain separate from operator data.

## Main modules

| Module | Responsibility |
|---|---|
| `store_paths.py` | Canonical selected-store layout |
| `memory_sources.py` | Source-root and extension configuration |
| `memory_index.py` | BM25 index, optional semantic ranking, filters, persistence |
| `memory_recall.py` | Recall orchestration with graph and temporal context |
| `consolidation_engine.py` | Episodic-to-semantic conversion and conflict records |
| `knowledge_store.py` | Atomic notes, links, triggers, and graph search |
| `temporal_graph.py` | Date extraction, temporal edges, and date arithmetic |
| `mcp_server.py` | Six-tool stdio JSON-RPC server |
| `api.py` | FastAPI request/response surface |
| `api_server.py` | Lightweight standard-library HTTP surface |
| `cli.py` | Operator commands and service launchers |
| `vector_pipeline.py` | Optional persistent local-vector bridge |

The complete public module surface is linked from the [API Guide](../api-guide.md).

## Retrieval path

```text
query classification and optional decomposition
  -> BM25 candidate scoring
  -> optional bi-encoder reranking
  -> optional cross-encoder reranking
  -> reciprocal-rank fusion
  -> entity and temporal enrichment
  -> rule-based answer extraction or optional model synthesis
```

Python fallbacks preserve functionality when native extensions are absent.
Seventeen Rust/PyO3 modules accelerate selected paths: 16 use the general CI
crate matrix, while temporal-SNN memory has separate installed-wheel verification.

## Memory types

| Type | Typical storage | Purpose |
|---|---|---|
| Episodic | `reasoning_traces/*.md` | Source observations and decisions |
| Semantic | `memory/semantic/**/*.md` | Consolidated durable facts |
| Graph | `memory/graph/*.jsonl` | Evidence-linked entity relations |
| Knowledge | `memory/knowledge_notes.jsonl` | Atomic linked notes and triggers |
| Procedural | `skills/*.json` | Extracted workflows |

## Integration surfaces

- **CLI:** human and script operation; see the [CLI reference](../api/cli.md).
- **Python:** in-process search and custom pipelines.
- **MCP:** stdio tools for recall, remember, status, graph, feedback, and
  correctness labels.
- **FastAPI:** shared HTTP service with OpenAPI and explicit auth controls.
- **Lightweight HTTP:** smaller route set without the FastAPI dependency.

These interfaces share storage, but their route and tool sets are not identical.
Choose the narrowest surface that meets the deployment need.

## Security boundary

Core retrieval has no network requirement. Optional embeddings may download
models during setup, and hosted synthesis can send selected context to a model
provider. API authentication is disabled when no token is configured; any
non-local deployment must use `--require-auth`, a protected token source, and a
trusted TLS boundary. See the [Integration Guide](../guides/INTEGRATION_GUIDE.md).

## SNN research boundary

The legacy SNN daemon and SNN text-retrieval design are not product-core paths.
The current temporal-SNN work is a separately packaged, preregistered experiment
with its own schemas, gates, and ADRs. See
[Temporal SNN Memory](../research/temporal_snn_memory.md) and
[ADR 0006](../adr/0006-preregister-temporal-snn-memory-experiment.md).

## Decisions and validation

- [ADR log](../adr/README.md) records accepted architectural decisions.
- [`VALIDATION.md`](https://github.com/anulum/remanentia/blob/main/VALIDATION.md)
  is the canonical test and coverage surface.
- [Model cards](../models/README.md) document evidence and limitations for
  learned components.
