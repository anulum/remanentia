SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
© Concepts 1996–2026 Miroslav Sotek. All rights reserved.
© Code 2020–2026 Miroslav Sotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li

# Remanentia

**Persistent AI memory with SNN-orchestrated consolidation, entity graphs, and deep contextual recall.**

92.9% retrieval precision | 20,000 LIF neurons | 197 entity relations | 24/7 consolidation

[remanentia.com](https://remanentia.com) | DOI: [10.5281/zenodo.19098792](https://doi.org/10.5281/zenodo.19098792)

---

## What It Does

Remanentia is a memory system for AI agents that goes beyond retrieval.

It encodes experiences as episodic traces, consolidates them into structured semantic knowledge, builds an entity relationship graph with evidence trails, and returns deep contextual recall: matched traces + consolidated knowledge + temporal timeline + cross-project insights.

A 20,000-neuron spiking network runs continuously, detecting novelty and orchestrating consolidation.

## Quick Start

```bash
pip install -e .

# Consolidate existing traces into semantic memories
python cli.py consolidate --force

# Check system status
python cli.py status

# Recall a memory with full context
python cli.py recall "Dimits shift convergence"

# View entity graph
python cli.py graph --top 10

# Start the API server (port 8001)
python api.py

# Start the GPU daemon (requires PyTorch CUDA)
.venv312/Scripts/python.exe gpu_daemon.py --detach
```

## Architecture

```
Query
  |
  v
TF-IDF + best-paragraph embedding ............. retrieval (92.9% P@1)
  |
  v
Consolidation engine .......................... episodic -> semantic
  |
  v
Entity graph .................................. 28 entities, 197 relations
  |
  v
Deep recall ................................... trace + knowledge + timeline + entities
```

### Memory Types

| Type | Storage | Example |
|------|---------|---------|
| Episodic | `reasoning_traces/*.md` | Raw session decisions |
| Semantic | `memory/semantic/**/*.md` | Consolidated facts with YAML frontmatter |
| Procedural | `skills/*.json` | Extracted skills and workflows |
| Graph | `memory/graph/*.jsonl` | Entity-entity relations with evidence |

### Components

| File | Role |
|------|------|
| `memory_recall.py` | Deep recall: retrieval + graph + temporal context |
| `consolidation_engine.py` | Episodic -> semantic compression pipeline |
| `retrieve.py` | Hybrid retrieval: TF-IDF + SNN + embedding |
| `gpu_daemon.py` | 20K neuron SNN, live retrieval IPC, consolidation |
| `monitor.py` | Dashboard at localhost:8888 |
| `cli.py` | Command-line interface |
| `api.py` | FastAPI REST server |
| `mcp_server.py` | MCP integration for Claude Code / Cursor |
| `encoding.py` | Text -> spike pattern encoding (hash/LSH/embedding) |
| `snn_backend.py` | GPU/CPU LIF network with STDP |

## CLI

```bash
remanentia recall "quantum control identity"       # deep recall
remanentia recall "STDP" --format context          # LLM-injectable context
remanentia recall "revenue strategy" --format json # machine-readable
remanentia consolidate                             # run consolidation
remanentia consolidate --force                     # reconsolidate everything
remanentia status                                  # daemon + memory stats
remanentia graph --top 15                          # top entity relationships
remanentia entities                                # list all entities
remanentia daemon start                            # start GPU daemon
remanentia daemon stop                             # stop daemon
```

## REST API

```bash
# Start server
python api.py  # -> http://localhost:8001/docs (auto-generated OpenAPI)

# Recall
curl -X POST http://localhost:8001/recall \
  -H "Content-Type: application/json" \
  -d '{"query": "STDP learning", "top_k": 3}'

# Status
curl http://localhost:8001/status

# Entities
curl http://localhost:8001/entities

# Entity graph
curl http://localhost:8001/graph?top=10

# Entity detail
curl http://localhost:8001/graph/entity/remanentia
```

## MCP Integration

For Claude Code, Cursor, or any MCP-compatible agent:

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

Tools:
- `remanentia_recall` — deep memory recall with full context
- `remanentia_status` — system status
- `remanentia_graph` — entity relationship query

## Example Integration

```python
from memory_recall import recall

# Agent receives a question
query = "What did we decide about the authentication middleware?"

# Get deep context
ctx = recall(query)

# Inject into LLM prompt
prompt = f"""Based on this memory context, answer the question.

{ctx.to_llm_context()}

Question: {query}"""

# ctx.to_llm_context() includes:
#   [Matched trace: auth_middleware_decision.md]
#   ...the actual decision text...
#   [Consolidated: security decisions summary]
#   ...extracted key points...
#   [Related concepts: auth(5), middleware(3), security(2)]
#   [Before: api_design.md, threat_model.md]
#   [After: deployment_plan.md]
#   [Cross-project: backend shares concepts: auth, middleware]
```

## Research

Built on 60+ experiments and honest negative results.

Tested: holographic Kuramoto memory, complex Hebbian coupling, E/I attractor networks, contrastive Hebbian learning, reservoir readouts, 5 architectural variants. Shuffled-W controls at every stage.

Key findings:
- STDP in LIF networks does not create discriminative features for text retrieval (negative result, all configs tested)
- Best-paragraph embedding outperforms document-level (85.7% vs 50.0%)
- TF-IDF(0.4) + best-paragraph(0.6) = 92.9% P@1 without SNN
- The SNN's role: consolidation orchestration + novelty detection, not retrieval

Full logs: `paper/comprehensive_findings.md`, `paper/holographic_memory_investigation.md`

## Stack

Python | PyTorch CUDA | Rust/PyO3 | sentence-transformers | NumPy/SciPy | BM25/TF-IDF | FastAPI

## License

AGPL-3.0-or-later | Commercial license available

## Author

Miroslav Sotek ([Anulum](https://www.anulum.li)) | ORCID: [0009-0009-3560-0851](https://orcid.org/0009-0009-3560-0851)

Part of the [SCPN framework](https://www.anulum.li).
