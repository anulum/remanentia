SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li

# Remanentia

[![CI](https://github.com/anulum/remanentia/actions/workflows/ci.yml/badge.svg)](https://github.com/anulum/remanentia/actions/workflows/ci.yml)
[![CodeQL](https://github.com/anulum/remanentia/actions/workflows/codeql.yml/badge.svg)](https://github.com/anulum/remanentia/actions/workflows/codeql.yml)
[![Docs](https://github.com/anulum/remanentia/actions/workflows/docs.yml/badge.svg)](https://github.com/anulum/remanentia/actions/workflows/docs.yml)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/anulum/remanentia/badge)](https://securityscorecards.dev/viewer/?uri=github.com/anulum/remanentia)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/12340/badge)](https://www.bestpractices.dev/projects/12340)
[![Version](https://img.shields.io/badge/version-0.3.1-blue)](https://github.com/anulum/remanentia)
[![Tests](https://img.shields.io/badge/tests-2143_passed-brightgreen)](VALIDATION.md)
[![Coverage Gate](https://img.shields.io/badge/coverage_gate-100%25-brightgreen)](pyproject.toml)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Rust](https://img.shields.io/badge/engine-Rust-orange)](docs/guides/PERFORMANCE_TUNING.md)
[![Ruff](https://img.shields.io/badge/linter-ruff-261230)](https://github.com/astral-sh/ruff)
[![Bandit](https://img.shields.io/badge/security-bandit-yellow)](https://github.com/PyCQA/bandit)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://pre-commit.com/)
[![MCP](https://img.shields.io/badge/MCP-server-purple)](docs/guides/INTEGRATION_GUIDE.md)
[![CITATION.cff](https://img.shields.io/badge/citation-CFF-blue)](CITATION.cff)
[![REUSE](https://img.shields.io/badge/REUSE-compliant-green)](REUSE.toml)

![Remanentia](docs/assets/remanentia_repo_header.png)

**Persistent AI memory with SNN-orchestrated consolidation, entity graphs, and deep contextual recall.**

> **Active Development** — Remanentia is under intensive development. The core memory engine, BM25+embedding hybrid retrieval, SNN-orchestrated consolidation, temporal reasoning, and the MCP server are functional, tested (2,143 passing tests in the 2026-05-12 local full-suite run), and deployable. The repository keeps a 100% coverage gate in CI via `pyproject.toml`; regenerate a coverage report before publishing module-level coverage numbers. Rust acceleration spans 16 crates with a Python fallback on every path. LongMemEval, **full-S setting** (realistic ~50-session haystacks, retrieval actually exercised; this is what published leaderboards measure): **56.6% overall** (3-run mean on the current `gpt-4o-mini`, spread 2.2). The older 72.2%/~71% figures are the *oracle* setting (gold sessions only, retrieval bypassed) and are not comparable to leaderboards — see Benchmarks for both. Figures are 3-run means, not single runs. APIs may evolve as this work progresses.

BM25+embedding hybrid retrieval with RRF | 11 typed entity relation types | temporal reasoning with date arithmetic | async consolidation | thread-safe MCP server

[remanentia.com](https://www.remanentia.com) | [GitHub](https://github.com/anulum/remanentia)

---

## Integration Positioning

Remanentia is a standalone, local-first memory engine for projects that need
auditable recall over notes, source code, research documents, and structured
findings. It can run as a repository-local CLI, an MCP server, or an HTTP API.

Cross-project retrieval is an integration pattern, not a packaging assumption:
deployments may add additional source roots for external repositories or shared
archives, but the public package does not require a private monorepo layout.

---

## What It Does

Remanentia indexes your project's existing files — session logs, code, research documents, reasoning traces — into a unified BM25 index with query intelligence. Ask a question, get the relevant paragraph with an extracted answer.

No vector database. No cloud service. No LLM in the retrieval path.

## Quick Start

```bash
# Install from PyPI
pip install remanentia

# Or install from source
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

## Retrieval Pipeline

```
Query
  |
  v
BM25 (real TF + inverted index) .............. first-pass retrieval
  |
  v
Bi-encoder rerank (MiniLM-L6-v2) ............. semantic similarity
  |
  v
Reciprocal Rank Fusion ........................ scale-invariant score fusion
  |
  v
Cross-encoder rerank (MiniLM-L-6-v2) ......... fine-grained re-scoring
  |
  v
Entity graph boost ............................ 11 typed relation types
  |
  v
Temporal graph + date arithmetic .............. TReMu code execution
  |
  v
Answer extraction ............................. query-proximity scoring
  |
  v
Knowledge store (multi-hop graph search) ...... Zettelkasten + prospective queries
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
| `memory_index.py` | Unified BM25 + embedding index, all scoring and ranking |
| `memory_sources.py` | Neutral default source roots plus JSON/env source configuration |
| `store_paths.py` | Canonical memory-store paths shared by ingest and freshness monitoring |
| `store_manifest.py` | Store-selection manifest for reconsolidation and freshness operations |
| `store_sources.py` | Selected-store MemoryIndex source config for backfill indexing |
| `memory_recall.py` | Deep recall: retrieval + graph + temporal context |
| `recall_calibration.py` | Recall abstention calibration from correctness-labelled ledger outcomes |
| `compiled_memory.py` | Typed fact-card compiler for durable seed and operational facts |
| `mcp_server.py` | Thread-safe MCP server (stdio JSON-RPC), async consolidation |
| `consolidation_engine.py` | Episodic -> semantic compression, typed relation extraction |
| `knowledge_store.py` | Zettelkasten atomic notes, prospective triggers, graph search |
| `temporal_graph.py` | Temporal event graph, relative date resolution, TReMu |
| `entity_extractor.py` | GLiNER2 NER + regex fallback, 11 typed relation types |
| `llm_backend.py` | Pluggable LLM backend: Auto, Local, hosted, Null |
| `answer_extractor.py` | Query-proximity answer extraction, LLM fallback |
| `answer_normalizer.py` | Hedging strip, yes/no polarity, semantic similarity |
| `observer.py` | Filesystem watcher -> incremental index updates |
| `reflector.py` | Periodic cluster summarisation + gap detection |
| `arcane_retriever.py` | 4-channel parallel retrieval with RRF fusion |
| `fact_decomposer.py` | Atomic fact decomposition with temporal validity windows |
| `claim_axes.py` | Shared finding evidence/status/admission/freshness vocabulary |
| `finding_ingest.py` | Hub-backed finding admission, cursoring, and Markdown persistence |
| `feed_ingest.py` | SYNAPSE feed.ndjson ingest for explicit findings and decisions |
| `feed_normalization.py` | Controlled project, actor, sequence, and timestamp vocabulary for feed ingest |
| `finding_envelope.py` | Signed finding envelopes with recall-gate, validity, void, and lineage closure checks |
| `cli.py` | Command-line interface |
| `api.py` | FastAPI REST server |
| `api_server.py` | Lightweight HTTP API for cross-service integration |

### Prerequisites

- Python 3.10+
- numpy (required)
- Optional: sentence-transformers (embedding rerank), torch (GPU), fastapi (REST API), `anthropic` Python package (hosted LLM)
- Optional: llama.cpp / Ollama for local LLM (any chat-completions-compatible endpoint)

## CLI

```bash
pip install -e ".[all]"     # everything
pip install -e ".[bus]"     # SYNAPSE bus and feed integration
pip install -e ".[api]"     # REST API only
pip install -e ".[seal]"    # signed finding envelopes (Python 3.12+)
pip install -e ".[dev]"     # test dependencies
```

```bash
remanentia-feed-ingest  # consume explicit Finding:/Decision: rows from ~/synapse/feed.ndjson
remanentia-recall-calibration --json  # report calibrated recall abstention evidence
remanentia-store-manifest --write --json  # record the selected memory-store manifest
remanentia-store-sources --write --json   # write MemoryIndex sources for that store
python tools/install_user_services.py --base /path/to/store --stimuli-dir /path/to/snn_stimuli
```

## Search Pipeline

```
Query → Classification (8 intent types)
  ↓
BM25 scoring (inverted index, real TF-IDF)
  ↓
Bi-encoder rerank (MiniLM-L6-v2, optional)
  ↓
Cross-encoder rerank (ms-marco-MiniLM, optional)
  ↓
Answer extraction (dates, numbers, versions, names)
  ↓
Results with snippets + extracted answers
```

## Benchmarks

### LongMemEval (committed, reproducible)

500 questions across 6 categories. GPT-4o-mini generation + judge. Per-run history
is tracked in `benchmarks/longmemeval_history.jsonl`.

LongMemEval has two settings, and the distinction matters: the **oracle** setting gives
the reader only the gold sessions (~2 per question), so retrieval is not exercised; the
**full-S** setting gives ~50 sessions per question (~2 of them gold), so the system must
actually retrieve. Published leaderboard numbers (e.g. Hindsight 91.4 %) are full-S. The
realistic, comparable number for Remanentia is the **full-S** figure below.

#### Full-S (realistic retrieval setting) — headline

~50 sessions/question, retrieved-context reader (top-10 retrieved sessions). 3-run mean
on the 2026-06 `gpt-4o-mini`, cross-encoder rerank on (ledger round `full-S`):

| Category | Full-S 3-run mean |
|----------|-------------------|
| knowledge-update | 79.5% |
| single-session-user | 73.8% |
| single-session-assistant | 71.4% |
| single-session-preference | 58.9% |
| multi-session | 42.1% |
| temporal-reasoning | 41.8% |
| **Overall** | **56.6%** (runs 57.6 / 55.4 / 56.8, spread 2.2) |

This is the number to compare against full-S leaderboards. The hard categories
(multi-session, temporal) carry most of the gap: retrieval recall@10 is high (~88 % /
~79 %), so the remaining loss there is synthesis over the retrieved context, not a
retrieval miss; single-session losses are retrieval misses on the legacy BM25 path.

#### Oracle setting (gold sessions only — retrieval NOT exercised)

The figures below are the oracle setting: the haystack is exactly the gold sessions, fed
to the reader in full. They measure synthesis, not retrieval, and are **not** comparable
to full-S leaderboards. Kept for historical continuity.

**Committed snapshot — R11, April 2026** (single run, `data/longmemeval_hypotheses.results.jsonl`):

| Category | Score |
|----------|-------|
| single-session-preference | 90.0% (27/30) |
| single-session-assistant | 87.5% (49/56) |
| single-session-user | 85.7% (60/70) |
| knowledge-update | 84.6% (66/78) |
| **temporal-reasoning** | **65.4% (87/133)** |
| multi-session | 54.1% (72/133) |
| **Overall** | **72.2% (361/500)** |

**Current model, default config (cross-encoder rerank on), 3-run mean on the 2026-06
`gpt-4o-mini`:** ~71.2% overall, ~60.2% temporal-reasoning, ~89% knowledge-update,
~53% multi-session. The ~1pp overall and ~5pp temporal gap vs the April snapshot is
`gpt-4o-mini` drift between model snapshots — the same code reproduces ~60% temporal on
the current model, so it is not a pipeline regression. Quote ≥3-run means; single-run
swings of ±~10 questions are noise.

Temporal-reasoning improved from 45.9% (R8) to **65.4% (R11, 2026-04-11)**, a +19.5pp / +26 question gain — the 65% target has been achieved. Four rounds shipped pipeline improvements:

- R9 (+14 temporal): session-anchored date resolution, explicit duration arithmetic via TReMu pre-computation, chronological session ordering
- R10 (+2 temporal): intraday HH:MM tiebreak + qtype-aware sort
- R11 (+10 temporal): fuzzy inclusive/exclusive durations, question_date anchoring in LLM prompt + TReMu, multi-event bigram/proximity tuning, narrow multi-hop chain resolution

Single-run LLM-judge noise is ±~10 questions per 500-run; per-round 3-run means are now recorded in `benchmarks/longmemeval_history.jsonl`. Cross-encoder reranking is **on by default** — set `REMANENTIA_ARCANE_CE_DISABLE=1` to skip it for latency-sensitive live/MCP use (it is worth ~8–9 questions on this benchmark).

Hindsight (SOTA with GPT-4 extraction) reports 91.4% on the **full-S** setting —
compare it to Remanentia's full-S **56.6 %**, not the oracle number above. Closing that
gap on full-S is the active work (retrieval recall on single-session, cross-session
synthesis on multi-session/temporal).

### LOCOMO

**1,651 / 1,986 = 83.1 %** on the LOCOMO multi-session QA dataset
(BM25 + cross-encoder rerank + 4-stage answer extraction + LLM
synthesis). Results committed at
[`paper/locomo_results.json`](paper/locomo_results.json).

| Category    | Correct / Total | Accuracy |
|-------------|----------------:|---------:|
| Multi-hop   | 285 / 321       | 88.8 %   |
| Temporal    | 60 / 96         | 62.5 %   |
| Single-hop  | 207 / 282       | 73.4 %   |
| Adversarial | 731 / 841       | 86.9 %   |
| Open-domain | 368 / 446       | 82.5 %   |
| **Overall** | **1 651 / 1 986** | **83.1 %** |

The LOCOMO dataset is distributed separately and must be obtained
by the reproducer; the preprocessed question order is pinned in
`bench_locomo.py`. Run with ``python bench_locomo.py --llm`` to
reproduce. See [`docs/benchmarks/LOCOMO.md`](docs/benchmarks/LOCOMO.md)
for the full methodology and the pre-LLM 74.7 % baseline.

## Trained components — honest status

The temporal-training programme produced five components with
mixed outcomes. We publish a model card for each so downstream
users can decide what to rely on; the rule-based date normaliser is
the only one with a committed, measured benchmark impact. Full
cards live at [`docs/models/`](docs/models/).

| Component | Status | Note |
|-----------|--------|------|
| C1 Embedding fine-tune | UNCERTAIN | 3 464 triplets, no retrieval A/B committed |
| C2 Cross-encoder fine-tune | UNCERTAIN | AP 84.57 % on own split; real-world eval pending (P2-13) |
| C3 Temporal relation classifier | **NON-FUNCTIONAL** | F1-macro 0.178 vs 0.167 random baseline — model did not learn the task, not wired into any default code path |
| C4 Date normaliser (rule engine) | PROVEN | 12 regex patterns drove temporal-reasoning +14.3 pp on LongMemEval |
| C4 Date normaliser (ML) | weak / unused | 24.8 % exact / 65.7 % relaxed — redundant vs rules |
| C5 Fact-validity model | OVERFITTED TO SYNTHETIC | 100 % on templates is a template-memorisation signal; gated behind regex in production |

Source: [`training/HONEST_ASSESSMENT.md`](training/HONEST_ASSESSMENT.md).

## MCP Integration

For Cursor or any MCP-compatible tool:

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

Set `REMANENTIA_BASE` to point ingest, freshness checks, `remanentia init`, and
`remanentia status` at a custom memory store. `remanentia init` creates that
store layout, while `remanentia status` reads its traces, semantic memory, graph
files, state files, and freshness report from the same resolved root. Set
`REMANENTIA_STIMULI_DIR` when the stimuli firehose lives outside that store.
The user-service installer accepts the same selection as `--base` and
`--stimuli-dir`; generated API, vector-worker, and freshness-watchdog units all
export those paths, and the watchdog writes the freshness report under the
selected store.

## CLI

```bash
remanentia search "query"                    # search (alias for recall)
remanentia recall "query" --format context   # LLM-injectable context
remanentia recall "query" --format json      # machine-readable
remanentia consolidate                       # consolidate new traces
remanentia consolidate --force               # reconsolidate all
remanentia status                            # system stats for REMANENTIA_BASE
remanentia store-manifest --write --json     # record selected store paths/counts
remanentia store-sources --write --json      # write selected MemoryIndex source config
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

## Rust Acceleration

16 PyO3 crates built with maturin. Python fallback preserved in every module.
Tiers 1–3 + recall pipeline complete — all compute-bound functions have a Rust path.

| Crate | Peak speedup | Wired into |
|-------|------------:|------------|
| remanentia_retrieve (new) | **26.7×** | retrieve.py, memory_index.py (13 functions) |
| remanentia_consolidation | **76.1×** | consolidation_engine.py (cluster_traces) |
| arcane_stdp | **45.4×** | snn_daemon.py (homeostatic_scaling) |
| remanentia_consolidation | **12.6×** | reflector.py (cluster_notes) |
| remanentia_answer_extractor | 11.4× | answer_extractor |
| remanentia_fact_decomposer | **8.8×** | fact_decomposer (RustFactIndex pyclass) |
| remanentia_entity_extractor | 8.5× | entity_extractor |
| remanentia_search | 3–5× | memory_index (BM25, Rayon) |
| remanentia_knowledge_store | 3.5–4.6× | knowledge_store |
| remanentia_temporal | 2.3× | temporal_graph (score_temporal_query) |
| remanentia_answer_normalizer | ~6× | answer_normalizer |
| remanentia_skill_extractor | ~1× | skill_extractor |
| remanentia_active_retrieval | ~1× | active_retrieval |

Full regex pipeline: **0.60ms** (Rust) vs 9.07ms (Python) on 470K chars = **14.1× on large workloads**.

Details: [RUSTIFICATION_PLAN.md](docs/guides/RUSTIFICATION_PLAN.md) | [PERFORMANCE_TUNING.md](docs/guides/PERFORMANCE_TUNING.md)

## Research (Negative Results)

SNN-based retrieval was the original design. After 70+ experiments across 4 learning rules (STDP, BCPNN, Hebbian, E/I balanced), we proved it adds zero discriminative signal. Root cause: 384-dim embeddings hash-encoded into 20K-neuron patterns are too correlated for local learning rules. The current system uses BM25 + optional neural reranking because that's what works.

Full analysis: `paper/remanentia_paper_draft.md`

## Tests

```bash
pip install -e ".[dev]"
pytest tests/ -q
```

Current local full-suite baseline: 2,143 passed, 3 skipped on 2026-05-12.
The CI coverage job is configured with `--cov-fail-under=100`; refresh the
coverage report before publishing exact module-level coverage counts.

## License

AGPL-3.0-or-later | Commercial license available

## Author

Miroslav Šotek ([Anulum](https://www.anulum.li)) | ORCID: [0009-0009-3560-0851](https://orcid.org/0009-0009-3560-0851)

---

<p align="center">
  <a href="https://www.anulum.li">
    <img src="docs/assets/anulum_logo_company.jpg" width="180" alt="ANULUM">
  </a>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://www.anulum.li">
    <img src="docs/assets/fortis_studio_logo.jpg" width="180" alt="Fortis Studio">
  </a>
  <br>
  <em>Developed by <a href="https://www.anulum.li">ANULUM</a> / Fortis Studio</em>
</p>
