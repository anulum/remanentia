# memory_recall

Deep recall combining BM25 retrieval, entity graph traversal, temporal context, and cross-project search into a unified RecallContext.

`memory_recall` consumes repository-local trace Markdown, consolidated semantic
Markdown, and JSONL entity-graph rows. Its semantic search and novelty scoring
use the Rust recall extension when available and keep deterministic Python
fallbacks for local-first deployments and test isolation.

::: memory_recall.recall

::: memory_recall.MemoryContext
    options:
      show_source: true
      members_order: source
