# Remanentia

**Local-first, auditable memory for AI agents and knowledge-intensive software.**

Remanentia turns Markdown traces, structured notes, and selected project files
into durable memory that applications can search through a CLI, Python, MCP, or
HTTP. Its core retrieval path works without a hosted model or vector database;
optional embedding and language-model layers add reranking and synthesis.

Use it when an assistant or team needs to remember decisions, recover evidence,
track how facts changed over time, or carry context across sessions without
sending the source corpus to a managed memory service.

!!! info "Production core and research boundary"
    The retrieval, graph, temporal, consolidation, CLI, MCP, and HTTP surfaces
    are implemented and tested. Temporal SNN memory remains an explicitly
    preregistered research programme; it is not required for normal retrieval.

## Start here

Choose the path that matches your goal:

| Goal | First document |
|---|---|
| Try local search in five minutes | [Quick Start](tutorials/quickstart.md) |
| Connect an AI agent over MCP | [Agent Memory Tutorial](tutorials/agent-memory.md) |
| Choose between CLI, Python, MCP, and HTTP | [Interface Guide](guides/choose-an-interface.md) |
| Integrate the HTTP or Python API | [API Guide](api-guide.md) |
| Evaluate fit, use cases, and operational value | [Use Cases and Value](product/use-cases-and-value.md) |
| Deploy and secure a service | [Container Deployment](guides/container_deployment.md) |
| Understand the implementation | [Architecture](architecture/architecture.md) |

## Five-minute local path

```bash
pip install remanentia
export REMANENTIA_BASE="$PWD/.remanentia-data"
remanentia init

# Add Markdown notes under reasoning_traces/, then search them.
remanentia search "what did we decide about authentication"
remanentia status
```

The first search builds the local index. `remanentia consolidate --force`
can later turn episodic traces into semantic memories and entity relations.

## What it provides

- **Hybrid recall:** BM25 first-pass retrieval with optional embedding,
  cross-encoder, entity, temporal, and reciprocal-rank-fusion stages.
- **Durable memory types:** episodic traces, semantic memories, procedural
  skills, and evidence-linked graph relations.
- **Multiple interfaces:** one engine exposed through CLI, Python, six MCP
  tools, and a FastAPI service with an exportable OpenAPI schema.
- **Sovereign operation:** the core path is local and model-free; hosted model
  use is optional and configured separately.
- **Temporal and provenance controls:** date-aware recall, correctness feedback,
  claim axes, signed finding envelopes, lineage reporting, and audit hooks.
- **Portable acceleration:** 17 Rust/PyO3 modules have Python fallbacks so the
  package remains usable without compiling native extensions.

## Common applications

| Application | What Remanentia contributes |
|---|---|
| Long-running AI assistants | Recall of prior decisions, constraints, and user preferences |
| Engineering and operations | Searchable handovers, incident findings, and architectural decisions |
| Research workflows | Evidence-linked notes, temporal updates, and reproducible retrieval evaluation |
| Privacy-sensitive environments | Local corpus control, explicit auth boundaries, and provenance surfaces |
| Multi-agent systems | Shared MCP/HTTP recall with feedback and correctness recording |

These are deployment patterns, not claims that Remanentia replaces a records
system, access-control layer, or domain-specific safety process. See
[Use Cases and Value](product/use-cases-and-value.md) for fit and non-goals.

## How a query moves through the system

```text
query
  -> BM25 candidate retrieval
  -> optional semantic and cross-encoder reranking
  -> reciprocal-rank fusion
  -> entity and temporal context
  -> answer extraction or optional LLM synthesis
  -> result with source context
```

The [architecture guide](architecture/architecture.md) describes the complete
module map. The [model cards](models/README.md) state which learned components
are proven, uncertain, weak, or non-functional.

## Evidence and readiness

Remanentia publishes its validation rules instead of embedding drifting test
counts in introductory pages:

- [`VALIDATION.md`](https://github.com/anulum/remanentia/blob/main/VALIDATION.md)
  is the canonical test and coverage record.
- [LongMemEval](benchmarks/LongMemEval.md) separates realistic full-S retrieval
  from oracle-only evaluation.
- [Sovereign Memory Evaluation](benchmarks/SOVEREIGN_MEMORY_EVALUATION.md)
  covers no-egress, abstention, write discipline, fleet recall, and lineage.
- [Security policy](https://github.com/anulum/remanentia/blob/main/SECURITY.md)
  explains supported versions and private reporting.

## Licensing and support

The public code is licensed under AGPL-3.0-or-later. Commercial licensing is
available for organisations that need different distribution terms.

Contact: [remanentia@anulum.li](mailto:remanentia@anulum.li) ·
[protoscience@anulum.li](mailto:protoscience@anulum.li) ·
[www.remanentia.com](https://www.remanentia.com)

---

<p align="center">
  <a href="https://www.anulum.li"><img src="assets/anulum_logo_company.jpg" height="70" alt="ANULUM"></a>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://www.anulum.li"><img src="assets/fortis_studio_logo.jpg" height="70" alt="Fortis Studio"></a>
  <br>
  <em>Developed by <a href="https://www.anulum.li">ANULUM</a> / Fortis Studio</em>
</p>
