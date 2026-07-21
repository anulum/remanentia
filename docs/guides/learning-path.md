# Learning Path

Progressive guide from first search to production deployment.

## Beginner: First 10 Minutes

**Goal:** Install, create a trace, run a search.

1. [Quick Start Tutorial](../tutorials/quickstart.md) — install, init, write a trace, search
2. Run `remanentia status` to see what's indexed
3. Run `remanentia search "your query"` to test retrieval

**Key concepts:**
- Reasoning traces are markdown files you write
- The index builds automatically on first search
- BM25 scoring finds relevant paragraphs without GPU or embeddings

**What you'll have:** A working search over your own session notes.

## Intermediate: Build a Knowledge Base

**Goal:** Use the knowledge store, set up MCP, understand consolidation.

1. [Knowledge Store Tutorial](../tutorials/knowledge_store_tutorial.md) — atomic notes, links, graph search
2. Complete the [Agent Memory Tutorial](../tutorials/agent-memory.md)
3. Run `remanentia consolidate --force` to extract semantic memories from traces
4. Browse `memory/semantic/` to see consolidated facts
5. Run `remanentia graph --top 10` to see entity relationships

**Key concepts:**
- Consolidation compresses episodic traces into semantic memories with YAML frontmatter
- The knowledge store auto-links notes by similarity and shared entities
- Prospective queries bridge the vocabulary gap between how you store and how you ask
- The MCP server lets AI agents search your memory

**What you'll have:** A structured knowledge base with entity relationships, searchable by AI agents.

## Intermediate: Temporal Reasoning

**Goal:** Query by date, understand event ordering, use date arithmetic.

1. [Temporal Queries Tutorial](../tutorials/temporal_tutorial.md) — date parsing, graph, code execution
2. Search with date filters: `idx.search("decision", after="2026-03-01")`
3. Try temporal queries: "what was the most recent decision", "how many days since X"

**Key concepts:**
- Dates are extracted automatically from ISO format, written English, and M/D/Y
- The temporal graph tracks event ordering (before/after/same_day edges)
- `temporal_code_execute()` computes answers from dates (durations, ordering, recency)
- Benchmark results depend on the evaluation setting. Use the
  [LongMemEval report](../benchmarks/LongMemEval.md) for current, evidence-linked
  full-S and historical oracle results.

**What you'll have:** Date-aware search with computed temporal answers.

## Advanced: Python API and Custom Pipelines

**Goal:** Build custom retrieval pipelines, use filtered search, integrate with applications.

1. Read the [Integration Guide](INTEGRATION_GUIDE.md) — Python API, REST API, environment variables
2. Use the [Interface Guide](choose-an-interface.md) to select the narrowest integration surface
3. Study the example scripts in `examples/`:
   - `basic_search.py` — MemoryIndex build/load/search
   - `filtered_search.py` — project, date, and doc_type filters
   - `knowledge_notes.py` — KnowledgeStore with graph search and triggers
   - `deep_recall.py` — full recall pipeline (BM25 + graph + temporal)
   - `temporal_queries.py` — TemporalGraph and date arithmetic
   - `consolidation.py` — episodic → semantic pipeline
   - `mcp_config.py` — MCP tool configuration and testing
3. Read the [Performance Tuning Guide](PERFORMANCE_TUNING.md) — embeddings, Rust BM25, model warmup

**Key concepts:**
- `MemoryIndex.search()` combines BM25, query classification, entity boost, and answer extraction
- `recall()` from `memory_recall` adds graph traversal, temporal context, and cross-project search
- Filters (project, after, before, doc_type) are applied before scoring
- Incremental updates via `idx.add_file()` avoid full rebuilds

**What you'll have:** Custom retrieval pipelines tailored to your application.

## Advanced: Multi-Channel Retrieval

**Goal:** Understand the ArcaneRetriever and fact decomposer for complex queries.

1. Read `arcane_retriever.py` — 4-channel parallel retrieval with RRF fusion
2. Read `fact_decomposer.py` — atomic fact decomposition with temporal validity
3. Study how multi-hop queries are decomposed in `memory_index.py` (`_decompose_query`)

**Key concepts:**
- The ArcaneRetriever runs BM25, entity, temporal, and cross-session channels in parallel
- Reciprocal Rank Fusion (RRF) combines results from heterogeneous channels without score normalisation
- The sufficiency loop retries with rewritten queries when results are insufficient
- Atomic facts track validity windows (valid_from/valid_until) for knowledge-update detection

**What you'll have:** Understanding of the full retrieval architecture for contributing or extending.

## Reference

| Resource | Purpose |
|----------|---------|
| [API Reference](../api/memory_index.md) | All public classes and functions |
| [Architecture](../architecture/architecture.md) | System design, module map, pipeline |
| [Benchmarks](../benchmarks/LongMemEval.md) | Measured performance with honest analysis |
| [Performance Tuning](PERFORMANCE_TUNING.md) | Latency, memory, Rust acceleration |
| [User Manual](USER_MANUAL.md) | Full feature reference |
