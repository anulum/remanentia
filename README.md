SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
© Concepts 1996–2026 Miroslav Sotek. All rights reserved.
© Code 2020–2026 Miroslav Sotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li

# Remanentia

**Filesystem knowledge retrieval for persistent AI agent memory.**

74.7% LOCOMO (1,986 questions) | <100ms queries | 243 tests | zero LLM calls

[remanentia.com](https://remanentia.com) | [GitHub](https://github.com/anulum/remanentia)

---

## What It Does

Remanentia indexes your project's existing files — session logs, code, research documents, reasoning traces — into a unified BM25 index with query intelligence. Ask a question, get the relevant paragraph with an extracted answer.

No vector database. No cloud service. No LLM in the retrieval path.

## Quick Start

```bash
# Install
pip install -e .

# Create directory structure
remanentia init

# Add your reasoning traces to reasoning_traces/
# Then consolidate into semantic memories
remanentia consolidate --force

# Search
remanentia search "what did we decide about authentication"
remanentia recall "STDP learning rule" --format context

# System status
remanentia status
```

## Prerequisites

- Python 3.10+
- numpy (required)
- Optional: sentence-transformers (embedding rerank), torch (GPU), fastapi (REST API)

```bash
pip install -e ".[all]"     # everything
pip install -e ".[api]"     # REST API only
pip install -e ".[dev]"     # test dependencies
```

## Search Pipeline

```
Query → Classification (8 intent types)
  ↓
BM25 scoring (15,938 paragraphs)
  ↓
Bi-encoder rerank (MiniLM-L6-v2, optional)
  ↓
Cross-encoder rerank (ms-marco-MiniLM, optional)
  ↓
Answer extraction (dates, numbers, versions, names)
  ↓
Results with snippets + extracted answers
```

## LOCOMO Benchmark

Evaluated on 1,986 questions from the LOCOMO multi-session QA dataset.

| Category | Accuracy |
|----------|----------|
| Multi-hop | 82.6% |
| Adversarial | 79.5% |
| Open-domain | 78.7% |
| Single-hop | 55.7% |
| Temporal | 42.7% |
| **Overall** | **74.7%** |

Method: BM25 + token overlap + answer extraction. No embedding rerank, no LLM.
Context: Hindsight (GPT-4 answer extraction) achieves 91.4%.

## MCP Integration

For Claude Code, Cursor, or any MCP-compatible tool:

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
- `remanentia_recall` — search with full context
- `remanentia_status` — system status
- `remanentia_graph` — entity relationship query

Set `REMANENTIA_BASE` env var to point to a custom memory directory.

## CLI

```bash
remanentia search "query"                    # search (alias for recall)
remanentia recall "query" --format context   # LLM-injectable context
remanentia recall "query" --format json      # machine-readable
remanentia consolidate                       # consolidate new traces
remanentia consolidate --force               # reconsolidate all
remanentia status                            # system stats
remanentia graph --top 15                    # entity relationships
remanentia entities                          # list all entities
remanentia init                              # create directory structure
```

## REST API

```bash
python api.py  # http://localhost:8001/docs

curl -X POST http://localhost:8001/recall \
  -H "Content-Type: application/json" \
  -d '{"query": "STDP learning", "top_k": 3}'

curl http://localhost:8001/status
curl http://localhost:8001/entities
curl http://localhost:8001/graph?top=10
```

## Python API

```python
from memory_index import MemoryIndex

idx = MemoryIndex()
idx.build(use_gpu_embeddings=False)
results = idx.search("what did we decide about auth", top_k=5)

for r in results:
    print(f"{r.name} (score={r.score})")
    if r.answer:
        print(f"  Answer: {r.answer}")
    print(f"  {r.snippet[:100]}")
```

## Architecture

| Module | Role |
|--------|------|
| `memory_index.py` | Unified BM25 index with query intelligence |
| `answer_extractor.py` | Regex answer extraction (dates, numbers, names) |
| `consolidation_engine.py` | Episodic traces → semantic memories |
| `entity_extractor.py` | GLiNER2 NER + typed relations |
| `memory_recall.py` | Rich recall: retrieval + graph + temporal |
| `mcp_server.py` | MCP server (stdio JSON-RPC) |
| `cli.py` | Command-line interface |
| `api.py` | FastAPI REST server |

## Research (Negative Results)

SNN-based retrieval was the original design. After 70+ experiments across 4 learning rules (STDP, BCPNN, Hebbian, E/I balanced), we proved it adds zero discriminative signal. Root cause: 384-dim embeddings hash-encoded into 20K-neuron patterns are too correlated for local learning rules. The current system uses BM25 + optional neural reranking because that's what works.

A Rust BM25 engine (PyO3 + Rayon) was built but is slower than Python at the current 15K-paragraph scale due to FFI overhead.

Full analysis: `paper/remanentia_paper_draft.md`

## Tests

```bash
pip install -e ".[dev]"
pytest tests/ -q
```

243 tests, 77% average coverage on core modules.

## License

AGPL-3.0-or-later | Commercial license available

## Author

Miroslav Sotek ([Anulum](https://www.anulum.li)) | ORCID: [0009-0009-3560-0851](https://orcid.org/0009-0009-3560-0851)
