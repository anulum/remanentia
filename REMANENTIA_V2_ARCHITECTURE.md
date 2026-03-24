# Remanentia v2: Observe → Reflect → Anticipate

**Date:** 2026-03-24
**Author:** Arcane Sapience
**Status:** EXECUTING
**Research basis:** 15+ competitor systems, 6 web searches, 3 research agents, A-MEM (NeurIPS 2025), Observational Memory (Mastra 94.87%), EverMemOS (92.3%), Zep/Graphiti bitemporal

## The Shift

v1: store text, retrieve text (passive)
v2: observe changes, reflect on knowledge, anticipate needs (active)

Every system scoring 90%+ treats memory as active computation, not passive storage.

## Five Phases

### Phase A: Knowledge Notes (A-MEM Zettelkasten)
**The foundation. Everything else builds on this.**

Replace flat paragraphs with interconnected atomic knowledge notes.

Each note:
- `id`: unique hash
- `title`: one-line summary
- `content`: the atomic fact/decision/finding
- `keywords`: extracted terms
- `source`: origin file + paragraph
- `created`: timestamp
- `updated`: timestamp (mutates when related notes arrive)
- `links`: list of related note IDs with relation types
- `embedding`: vector (optional, for similarity)

When a new note is created:
1. Compute similarity against existing notes
2. Create bi-directional links to top-3 most similar
3. Update the `updated` timestamp on linked notes
4. If similarity > 0.9 to an existing note, MERGE (update existing, don't create duplicate)

**Files:**
- `knowledge_store.py` — new module: KnowledgeNote dataclass, KnowledgeStore class
  - `add_note(content, source, keywords)` → creates note, links, merges if duplicate
  - `search(query, top_k)` → BM25 over notes (reuse existing tokenizer)
  - `get_related(note_id, depth)` → graph traversal
  - `save() / load()` — JSONL persistence
- `memory_index.py` — integrate KnowledgeStore into build() and search()
- `tests/test_knowledge_store.py` — full coverage

**No LLM required.** Similarity via token overlap + optional embeddings.

### Phase B: Contradiction Detection
**When new knowledge contradicts old knowledge, flag it.**

In `KnowledgeStore.add_note()`:
1. After finding similar notes, check for contradiction signals:
   - Same entity + opposite verbs ("removed" vs "added", "killed" vs "started")
   - Same metric + different values ("81.2%" vs "74.7%")
   - Temporal override (newer date supersedes older)
2. When contradiction detected:
   - Mark old note as `superseded_by: new_note_id`
   - Mark new note as `supersedes: old_note_id`
   - Both notes preserved (no deletion)
3. On retrieval, return most recent version + contradiction history

**Files:**
- `knowledge_store.py` — add contradiction detection to `add_note()`
- `tests/test_knowledge_store.py` — contradiction test cases

### Phase C: Observer (Filesystem Watcher)
**Auto-create knowledge notes when files change.**

Watch directories:
- `reasoning_traces/` — new traces
- `memory/semantic/` — consolidated memories
- `.coordination/sessions/` — session logs
- `.coordination/handovers/` — handovers

When a file changes:
1. Extract paragraphs (reuse `_split_paragraphs`)
2. For each paragraph with decision/finding/metric content:
   - Create a KnowledgeNote
   - Let KnowledgeStore handle linking + contradiction detection
3. Run consolidation (already wired)
4. Rebuild index if in watch mode

**Files:**
- `observer.py` — new module: filesystem watcher, paragraph→note extraction
- `memory_index.py` — wire observer into `--watch` mode

### Phase D: Prospective Triggers
**"When context matches X, surface Y." Checked on every recall.**

New data structure:
```python
@dataclass
class Trigger:
    id: str
    condition: str      # keywords/pattern to match
    action: str         # what to surface
    created: str        # when created
    fired: list[str]    # timestamps when triggered
    active: bool        # can be deactivated after firing
```

- `remanentia_remember` gains `trigger` parameter: `{"content": "...", "trigger": "when working on scpn-control"}`
- On every `remanentia_recall`, check all active triggers against query
- If match: prepend trigger action to results
- MCP tool: `remanentia_triggers` to list/manage

**Files:**
- `knowledge_store.py` — add Trigger dataclass and trigger checking
- `mcp_server.py` — add trigger support to recall + new triggers tool
- `cli.py` — add trigger management commands

### Phase E: Reflector (LLM-Powered Consolidation)
**Periodic deep consolidation that compresses, links, and generates insights.**

Runs via `--reflect` CLI command or periodically in watch mode:
1. Scan all knowledge notes from last N days
2. For each cluster of related notes, ask LLM to:
   - Generate a dense summary note
   - Identify the key decision/finding
   - Generate 5 prospective queries (Kumiho technique)
   - Flag any unresolved contradictions
   - Identify knowledge gaps
3. Create summary notes with links to source notes
4. Output a human-readable digest

**Files:**
- `reflector.py` — new module: LLM-powered reflection cycle
- `cli.py` — add `reflect` command
- `memory_index.py` — wire into `--watch` mode (periodic)

## Execution Order

Phase A first — it's the foundation. Each subsequent phase adds a capability.

## Success Criteria

- [ ] Knowledge notes replace flat paragraphs as primary memory unit
- [ ] Contradictions detected and tracked automatically
- [ ] Filesystem changes auto-create notes (observer)
- [ ] Prospective triggers fire on matching queries
- [ ] Reflector generates weekly knowledge digests
- [ ] LOCOMO 85%+ with knowledge-note-augmented retrieval
- [ ] All tests pass, 100% coverage on new modules
- [ ] Zero-LLM mode works for Phases A-D (LLM only in Phase E)
