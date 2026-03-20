# Memory Engineering Landscape — What the Industry Does

**Date:** 2026-03-19
**Source:** DeepLearning.AI course, Mem0, LangMem, academic papers, practitioner blogs

## The Course: Agent Memory: Building Memory-Aware Agents

**By:** Richmond Alake + Nacho Martinez (DeepLearning.AI / Oracle)
**Key shift:** From "prompt engineering" to "memory engineering" — treating
long-term memory as first-class infrastructure, external to the model.

### Memory Types (CoALA Framework)

| Type | What It Stores | Where | Remanentia Equivalent |
|------|---------------|-------|----------------------|
| **Working** | Current conversation | Context window | Claude's session context |
| **Semantic** | Facts, preferences, knowledge | External DB/files | reasoning_traces/ |
| **Episodic** | Past interactions, outcomes | External DB/files | retrieval_history.jsonl |
| **Procedural** | Workflows, skills, rules | System prompt/files | skills.json, disposition/ |

### Six Memory Operations (Richmond Alake)

1. **Generation** — creating memory content from interactions
2. **Storage** — persisting to external system
3. **Retrieval** — accessing stored data on demand
4. **Integration** — incorporating into current context
5. **Updating** — modifying existing records
6. **Deletion** — removing obsolete data

### Formation Pathways

- **Active/Conscious (Hot Path):** Agent recognizes and stores during conversation
- **Passive/Subconscious (Background):** Extract patterns after session ends

## Mem0: Production Memory Architecture

**Paper:** arxiv.org/abs/2504.19413
**Results:** 26% accuracy improvement, 91% lower latency, 90% token savings

### Two-Phase Pipeline

**Phase 1 — Extraction:**
- Ingest: latest exchange + rolling summary + recent messages
- LLM extracts candidate memories (structured facts)

**Phase 2 — Update:**
- Compare each new fact to top-s similar entries in vector DB
- Decide: ADD, UPDATE, DELETE, or NOOP

### Graph Memory (Mem0g)
- Memories as directed labeled graph (entities = nodes, relationships = edges)
- Entity Extractor → Relations Generator → Conflict Detector → Update Resolver
- ~2% better than base Mem0

## LangMem: LangChain's Memory Architecture

### Three Memory Types

- **Semantic:** Collections (unbounded docs, searched at runtime) + Profiles (strict schema)
- **Episodic:** Successful interaction patterns with context
- **Procedural:** System prompts that evolve through feedback

### Two-Layer Pattern

- **Layer 1 (Core API):** Pure functions — extract, update, remove, consolidate
- **Layer 2 (Stateful):** LangGraph persistence — store managers, memory tools

### Organization

Hierarchical namespaces: organization → user → context
Three retrieval mechanisms: key access, semantic search, metadata filtering

## File-Based Memory (Practitioner Pattern)

### The Index Pattern (what we already use)

MEMORY.md = index, not dump. Points to files, doesn't contain them.
Index stays ~50 lines while memory grows to hundreds of files.
This is EXACTLY what our `C:\Users\forti\.claude\...\memory\MEMORY.md` does.

### Searchable Agent Memory (Eric Tramel, 2026)

- JSONL transcripts indexed via BM25 (not vectors!)
- Microsecond query latency via precomputed sparse matrices
- Single-file MCP server, 3 dependencies
- Why BM25 over vectors: agents search their own artifacts with keywords
  they wrote — no vocabulary gap

### folder.md Format

Standardized directories:
- `prompts/` — SOUL.md, IDENTITY.md, USER.md, RULES.md, MEMORY.md
- `notes/` — daily logs, topic files
- `drafts/`, `final/`, `workspace/skills/`

## Key Academic Finding (ICLR 2026)

"Episodic Memory is the Missing Piece for Long-Term LLM Agents"
(arxiv.org/abs/2502.06975)

The argument: semantic memory (facts) and procedural memory (skills) are
addressed by current systems. What's missing is episodic memory — the
ability to recall SPECIFIC PAST EXPERIENCES with temporal and contextual
richness. Not "user prefers dark mode" (semantic) but "last Tuesday when
we were debugging the CI pipeline and discovered the race condition in
the STDP weight update" (episodic).

## What Remanentia Already Has vs What's Missing

### Already Have
- File-based memory with MEMORY.md index (semantic memory)
- Reasoning traces (episodic fragments)
- Session logs (episodic records)
- Skill extraction (procedural memory)
- Cognitive snapshots (state persistence)
- Retrieval history (usage patterns)
- SNN daemon (infrastructure)

### Missing
- **Structured extraction pipeline** — we don't extract structured facts
  from sessions. Traces are raw markdown, not parsed memory objects.
- **Consolidation** — episodic → semantic conversion. We accumulate traces
  but never consolidate "in 5 sessions about quantum-control, the key
  decisions were X, Y, Z" into a compact semantic memory.
- **Self-updating** — memories don't update. If a fact changes (repo version
  bumped), old memory persists alongside new.
- **Conflict resolution** — no mechanism for contradictory memories.
- **Active formation** — the SNN daemon processes stimuli but doesn't
  extract MEMORIES from them. It updates weights that don't contribute
  to retrieval.
- **Graph structure** — traces are flat files. No entity-relationship graph
  connecting projects, decisions, people, concepts.

## What This Means for Remanentia's Architecture

The industry consensus (2026):
1. Memory = external, structured, persistent (not in-context, not in-weights)
2. Memory operations = extract → store → retrieve → update → delete
3. Retrieval = BM25 (keywords) + vector search (semantics) + metadata filters
4. Consolidation = episodic → semantic (compress experiences into facts)
5. File-based is fine for small-medium scale (exactly our situation)

**The SNN's natural role in this architecture:**
NOT retrieval (BM25 + embeddings handle that at 92.9%).
Instead: **consolidation and association** — the background process that
converts raw episodic traces into consolidated semantic memories, detects
patterns across sessions, and maintains the entity-relationship graph.

This is exactly what biological memory consolidation does during sleep:
replay recent experiences, strengthen important connections, compress
episodes into general knowledge. The SNN daemon already runs continuously.
It should be doing CONSOLIDATION, not retrieval.

## Sources

- [Agent Memory Course](https://learn.deeplearning.ai/courses/agent-memory-building-memory-aware-agents/information)
- [Memory in AI Agents (Leonie Monigatti)](https://www.leoniemonigatti.com/blog/memory-in-ai-agents.html)
- [LangMem Conceptual Guide](https://langchain-ai.github.io/langmem/concepts/conceptual_guide/)
- [Mem0 Paper](https://arxiv.org/abs/2504.19413)
- [Mem0 Graph Memory](https://mem0.ai/blog/graph-memory-solutions-ai-agents)
- [Searchable Agent Memory](https://eric-tramel.github.io/blog/2026-02-07-searchable-agent-memory/)
- [Episodic Memory Position Paper](https://arxiv.org/abs/2502.06975)
- [ICLR 2026 MemAgents Workshop](https://openreview.net/pdf?id=U51WxL382H)
- [Agent Memory Paper List](https://github.com/Shichun-Liu/Agent-Memory-Paper-List)
