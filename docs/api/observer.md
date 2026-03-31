# observer

Filesystem watcher for incremental index updates with heartbeat-driven
autonomous maintenance.

## Purpose

The observer monitors project directories for new or changed files, extracts
knowledge notes from substantive paragraphs, and maintains the knowledge store.
In v0.4, the observer gained **heartbeat** capabilities inspired by OpenClaw's
Heartbeat component — periodic maintenance that runs consolidation, memory
aging, and capacity monitoring without manual intervention.

## Architecture

```
Filesystem (reasoning_traces/, memory/semantic/, sessions/, handovers/)
        │
        ▼
  ObserverState (tracks file path → mtime)
        │
        ▼
  observe_once()                      ← quick scan (every 30s)
    ├── Scan all watched directories
    ├── Detect new/changed files (mtime comparison)
    ├── Extract knowledge notes (_split_into_paragraphs + _has_signal)
    ├── Add notes to KnowledgeStore
    └── Incrementally update MemoryIndex (add_file)
        │
        ▼ (every ~5 minutes)
  heartbeat()                         ← full maintenance tick
    ├── observe_once()                → filesystem changes
    ├── consolidate()                 → pending traces → semantic memories
    ├── age_memories()                → lifecycle transitions (stale/archive)
    └── capacity_report()             → per-category overflow detection
```

## Heartbeat

The heartbeat function combines all maintenance operations into a single tick.
It runs every 10 observe cycles (~5 minutes at the default 30-second interval).

### Operations

| Step | Function | Purpose | Typical time |
|------|----------|---------|-------------|
| 1 | `observe_once()` | Detect new/changed files | <10ms |
| 2 | `consolidate()` | Compress pending traces | <50ms (no traces), variable |
| 3 | `age_memories()` | Lifecycle state transitions | <20ms |
| 4 | `capacity_report()` | Check per-category limits | <10ms |

### Return value

```python
{
    "observe": {
        "files_scanned": 47,
        "files_new": 2,
        "notes_created": 5,
    },
    "consolidate": {
        "traces_processed": 2,
        "memories_written": 1,
        "entities_found": 12,
    },
    "aging": {
        "scanned": 23,
        "active_to_stale": 1,
        "stale_to_archived": 0,
    },
    "capacity": {
        "categories_checked": 4,
        "categories_over_threshold": 1,
        "over_capacity": ["decision"],
    },
}
```

### Error isolation

Each step catches exceptions independently. A failure in consolidation
does not prevent aging or capacity checking from running. Error details
are stored in the respective result section:

```python
{"consolidate": {"error": "permission denied on traces dir"}}
```

## Observe Once

The core scanning function. Iterates all watched directories, compares
file mtimes against the stored state, and creates knowledge notes for
new/changed files.

### Signal words

A paragraph must contain at least one signal word to become a knowledge
note. This filters out boilerplate (headers, copyright notices, import
lists) and keeps only substantive content.

```python
_SIGNAL_WORDS = {
    "decided", "decision", "found", "finding", "result", "measured",
    "fixed", "broke", "shipped", "released", "removed", "added",
    "accuracy", "percent", "score", "benchmark", "p@1", "because",
    "root cause", "confirmed", "rejected", "chose", "version",
    "v0.", "v1.", "v2.", "v3.", "critical", "important", "key insight",
    "conclusion",
}
```

### Incremental index update

When new files are detected, they are added to the running `MemoryIndex`
via `add_file()` (if the MCP server's unified index is available). This
enables live search over newly added content without a full rebuild.

## Watched Directories

| Source | Path | Content |
|--------|------|---------|
| traces | `reasoning_traces/` | Raw session decisions and findings |
| semantic | `memory/semantic/` | Consolidated semantic memories |
| sessions | `.coordination/sessions/arcane-sapience/` | Session state files |
| handovers | `.coordination/handovers/arcane-sapience/` | Handover documents |

Custom directories can be passed to `observe_once()` and `heartbeat()`:

```python
state = ObserverState()
result = heartbeat(state, {"custom": Path("/my/watched/dir")})
```

## ObserverState

Tracks which files have been processed and their last modification time.
Persists to `memory/observer_state.json`.

### Methods

| Method | Description |
|--------|-------------|
| `is_new_or_changed(path)` | True if file is new or modified since last scan |
| `mark_processed(path)` | Record file's current mtime |
| `save(path)` | Persist state to JSON |
| `load(path)` | Load state from JSON |

### File format

```json
{
    "/path/to/file1.md": 1711987200.0,
    "/path/to/file2.md": 1712073600.0
}
```

## Pipeline Integration

The observer sits between the filesystem and the knowledge/retrieval layer:

```
Filesystem changes
        │
        ▼
  observer.observe_once()
        │
        ├── KnowledgeStore.add_note()     ← knowledge_store.py
        ├── MemoryIndex.add_file()        ← memory_index.py (via mcp_server)
        │
        ▼ (via heartbeat)
        ├── consolidation_engine.consolidate()
        ├── consolidation_engine.age_memories()
        └── consolidation_engine.capacity_report()
```

The observer is started by:

- `remanentia daemon start` → runs `observe_loop()` continuously
- MCP server → calls `observe_once()` on memory writes
- Manually: `python observer.py` (direct execution)

## Performance

| Operation | Time | Budget |
|-----------|------|--------|
| `observe_once()` (47 files, 2 new) | <10ms | 50ms |
| `heartbeat()` (empty dirs) | <50ms | 50ms |
| `heartbeat()` (full cycle) | <100ms | 200ms |
| `_has_signal()` per paragraph | <0.001ms | negligible |

## Usage

```python
from observer import ObserverState, observe_once, heartbeat

# Quick scan
state = ObserverState()
state.load()
result = observe_once(state)
print(f"New files: {result['files_new']}, Notes: {result['notes_created']}")
state.save()

# Full heartbeat
result = heartbeat(state)
if result["capacity"]["categories_over_threshold"] > 0:
    print(f"WARNING: {result['capacity']['over_capacity']} need consolidation")

# Continuous loop (daemon mode)
from observer import observe_loop
observe_loop(interval=30)  # runs forever, heartbeat every ~5 min
```

## Test Coverage

48 tests in `tests/test_observer.py`:

- **Signal detection**: decision, finding, metric, no signal, empty
- **Paragraph splitting**: multi-paragraph, short filter, no signal
- **Note extraction**: knowledge notes, file not found, restricted, empty
- **State tracking**: new file, unchanged, modified, save/load roundtrip
- **Observe once**: full scan, incremental update, empty dirs
- **Heartbeat**: all 4 sections present, triggers consolidation, empty dirs,
  performance (<50ms)
- **Pipeline integration**: notes flow into KnowledgeStore, index update

All 6 STRONG dimensions: empty, error, negative, pipeline, roundtrip, performance.

## API Reference

::: observer.ObserverState
    options:
      show_source: true
      members_order: source

::: observer.observe_once

::: observer.heartbeat
