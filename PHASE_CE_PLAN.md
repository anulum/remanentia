# Phase C + E: Observer & Reflector

**Date:** 2026-03-24
**Status:** EXECUTING

## Phase C: Observer (filesystem watcher → auto knowledge notes)

### What it does
When files change in watched directories, automatically:
1. Extract paragraphs with decisions/findings/metrics
2. Create KnowledgeNotes in the store (with linking + contradiction detection)
3. Update the temporal graph
4. Trigger consolidation

### Watched directories
- `reasoning_traces/` — new reasoning traces
- `memory/semantic/` — consolidated memories
- `.coordination/sessions/arcane-sapience/` — session logs
- `.coordination/handovers/arcane-sapience/` — handovers

### Integration
- Standalone: `python observer.py` (runs forever, polls every 30s)
- Wired into `memory_index.py --watch` mode
- CLI: `remanentia observe` command

### Files
- `observer.py` — new module
  - `ObserverState` — tracks which files have been processed (path → mtime)
  - `observe_once()` — scan all watched dirs, process new/changed files
  - `observe_loop(interval)` — poll forever
  - `_extract_notes_from_file(path)` — split into paragraphs, filter for substantial content, create notes
- `cli.py` — add `observe` command
- `tests/test_observer.py` — full coverage

### Design decisions
- Polling (not inotify/watchdog) — simpler, cross-platform, no extra deps
- State persisted to `memory/observer_state.json` — survives restarts
- Each file processed at most once per mtime — idempotent
- No LLM needed — heuristic extraction from text

## Phase E: Reflector (LLM-powered deep consolidation)

### What it does
Periodically processes recent knowledge notes and:
1. Generates dense summary notes from clusters of related notes
2. Generates prospective queries per note (Kumiho technique)
3. Identifies unresolved contradictions and knowledge gaps
4. Produces a human-readable digest

### Integration
- CLI: `remanentia reflect` (one-shot) or `remanentia reflect --watch` (periodic)
- Wired into observer loop (every N cycles)
- MCP: `remanentia_reflect` tool

### Files
- `reflector.py` — new module
  - `reflect_once(days)` — process notes from last N days
  - `_cluster_notes(notes)` — group related notes
  - `_generate_summary(cluster)` — LLM call to summarize
  - `_generate_prospective_queries(note)` — LLM call for Kumiho queries
  - `_identify_gaps(notes)` — find missing knowledge
  - `_generate_digest(results)` — human-readable output
- `cli.py` — add `reflect` command
- `tests/test_reflector.py` — full coverage

### LLM calls (all optional, Claude Haiku)
- Summary: ~1 per cluster (~$0.001 each)
- Prospective queries: ~1 per note (~$0.001 each)
- Gaps: ~1 per reflection cycle (~$0.002)

## Execution Order

1. Phase C: observer.py + tests + CLI integration
2. Phase E: reflector.py + tests + CLI integration
3. Wire both into --watch mode
4. Run full test suite, maintain 100% coverage

## Success Criteria
- [ ] Observer auto-creates notes when files change
- [ ] Observer state persists across restarts
- [ ] Reflector generates summaries and prospective queries
- [ ] Reflector identifies knowledge gaps
- [ ] CLI: `remanentia observe` and `remanentia reflect` work
- [ ] 100% test coverage maintained
