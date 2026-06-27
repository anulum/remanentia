# consolidation_engine

Episodic-to-semantic memory compression with typed relation extraction,
memory lifecycle management, bounded capacity tracking, and hierarchical
summary DAGs.

## Purpose

The consolidation engine transforms raw episodic traces (reasoning_traces/*.md)
into structured semantic memories. It is the core of Remanentia's long-term
memory formation — without consolidation, the system operates only on raw
full-text search over unstructured traces.

Three new subsystems were added in v0.4 (inspired by Hermes Agent, OpenClaw,
and Engram):

1. **Memory lifecycle** — validity_state transitions (active → stale → archived)
2. **Bounded capacity** — per-category char limits with overflow warnings
3. **Hierarchical summary DAGs** — multi-level compression for efficient search

## Architecture

```
reasoning_traces/*.md
        │
        ▼
  get_pending_traces()        → find unconsolidated traces
        │
        ▼
  _extract_metadata()         → date, project, type from filename
  _extract_entities()         → 3-layer extraction (project, concept, dynamic)
  _extract_key_lines()        → trigger-based key line extraction
  _extract_paragraphs()       → meaningful paragraph splitting
        │
        ▼
  _cluster_traces()           → group by project + date proximity (2-day gap)
        │
        ▼
  _write_semantic_memory()    → YAML frontmatter + content with lifecycle state
  _update_graph()             → entity-entity typed relations (10 relation types)
  build_summary_dag()         → hierarchical multi-level compression
        │
        ▼
  memory/semantic/**/*.md     → structured semantic memories
  memory/graph/*.jsonl        → entity graph
  consolidation/summary_dag.json → DAG for efficient search
```

## Memory Lifecycle

Every semantic memory has a `validity_state` field in its YAML frontmatter that
progresses through four states:

```
active ──→ validated ──→ stale ──→ archived
  │            │           │
  └────────────┴───────────┘
        (re-referenced → reset to active)
```

### States

| State | Meaning | Search behaviour |
|-------|---------|-----------------|
| **active** | Newly created, not yet confirmed | Normal retrieval weight |
| **validated** | Confirmed by a later trace | Slight boost in retrieval |
| **stale** | No access for >90 days | Reduced retrieval weight |
| **archived** | No access for >365 days, or contradicted | Excluded from default search |

### Transitions

| From | To | Trigger |
|------|----|---------|
| active | validated | Later trace confirms same fact |
| active/validated | stale | `age_memories()` when `last_accessed` > STALE_AFTER_DAYS (90) |
| stale | archived | `age_memories()` when `last_accessed` > ARCHIVE_AFTER_DAYS (365) |
| any | active | Trace re-references the memory (reset) |
| any | archived | Contradicted by a newer fact |

### Configuration

```python
STALE_AFTER_DAYS = 90    # days without access before marking stale
ARCHIVE_AFTER_DAYS = 365 # days without access before archiving
```

### Frontmatter Example

```yaml
---
type: decision
date: 2024-03-15
project: remanentia
source_traces:
  - 2024-03-15T1400_retrieval_decision.md
entities:
  - bm25
  - cross-encoder
confidence: 0.8
validity_state: active
last_validated: 2024-03-15
last_accessed: 2024-03-15
---
```

### Pipeline Integration

`age_memories()` is called by the **heartbeat** function in `observer.py` on
every maintenance tick (every ~5 minutes when the observer loop is running).
It scans all semantic memory files, parses their frontmatter, and transitions
states based on `last_accessed` timestamps.

`_write_semantic_memory()` sets `validity_state: active` and `last_accessed`
on every newly created memory.

## Bounded Capacity Tracking

Each semantic memory category has a configurable character limit. When a
category exceeds 80% of its limit, `capacity_report()` flags it for
consolidation.

### Category Limits

| Category | Char limit | Purpose |
|----------|-----------|---------|
| decision | 50,000 | Architectural/design decisions |
| finding | 100,000 | Experimental findings, measurements |
| strategy | 30,000 | Revenue, competitive strategy |
| technical | 100,000 | Technical details, fixes, migrations |
| continuity | 20,000 | Identity, contribution tracking |
| personal | 20,000 | Personal context, relationships |
| findings | 100,000 | Alias for mixed finding types |
| general | 50,000 | Uncategorised |

Default limit for unlisted categories: 50,000 chars.
Warning threshold: 80% (`CAPACITY_WARN_PERCENT`).

### capacity_report()

Returns a dict mapping each category to:

```python
{
    "chars": 42_350,           # total characters in category
    "limit": 50_000,           # configured limit
    "usage_pct": 84.7,         # percentage used
    "needs_consolidation": True,  # above 80% threshold
    "file_count": 12,          # number of .md files
    "state_counts": {          # breakdown by validity_state
        "active": 8,
        "stale": 3,
        "archived": 1,
    }
}
```

### CLI Integration

`remanentia status` now displays capacity per category:

```
Memory:
  Episodic traces: 47
  Semantic memories: 23
  Entities: 156
  Relations: 412

  Capacity:
   ! decision         84.7% (42,350 / 50,000 chars, 12 files) [active=8, stale=3, archived=1]
     finding          23.1% (23,100 / 100,000 chars, 5 files) [active=5]
     technical        67.2% (67,200 / 100,000 chars, 8 files) [active=6, validated=2]
```

Categories above 80% are marked with `!`.

## Hierarchical Summary DAGs

Multi-level compression of episodic traces, inspired by Engram's Lossless
Context Management (LCM). Enables efficient search over long histories by
searching at high levels first, then drilling down to leaf nodes.

### Structure

```
Level 3:  [Meta-summary]          (~128 traces)
            │
Level 2:  [Super-summary] × 4    (~32 traces each)
            │
Level 1:  [Cluster summary] × 4  (~8 traces each)
            │
Level 0:  [Leaf] × 4             (individual traces)
```

Each node (`DAGNode`) contains:

| Field | Type | Description |
|-------|------|-------------|
| node_id | str | Unique identifier (e.g., `L1_0_2024-01`) |
| level | int | 0 = leaf (raw trace), 1+ = summary |
| summary | str | Extracted key lines (L0) or merged child summaries (L1+) |
| children | list[str] | Child node_ids or trace filenames |
| date_range | (str, str) | Earliest and latest ISO dates covered |
| entities | list[str] | Union of child entities (capped at 30) |
| project | str | Most common project in children |

### Build algorithm

1. Create leaf nodes (L0) from each trace — summary = first 5 key lines
2. Group leaves into batches of `DAG_FANOUT` (default: 4)
3. For each group, create a parent node (L1) with merged summary
4. Repeat grouping until only one root remains (or fewer than DAG_FANOUT nodes)

### Search algorithm

Top-down search with score-based expansion:

1. Score all nodes at the highest level by query token overlap
2. Expand the top-scoring nodes by scoring their children
3. Continue expanding until reaching leaf nodes (L0)
4. Return leaf nodes sorted by score

This avoids scanning all leaf nodes — for 1,000 traces, the search examines
~30 nodes instead of all 1,000.

### Performance

| Operation | 8 traces | 100 traces | Budget |
|-----------|---------|-----------|--------|
| `build_summary_dag()` | 0.02ms | 0.8ms | <100ms |
| `search_summary_dag()` | 0.01ms | 0.05ms | <10ms |

### Integration

`build_summary_dag()` is called inside `consolidate()` after cluster
processing. The DAG is saved to `consolidation/summary_dag.json` and
rebuilt on every consolidation run.

`search_summary_dag()` can be called independently for efficient historical
search without loading the full FactIndex.

## Entity Graph

The consolidation engine maintains an entity-entity relationship graph with
10 typed relations:

| Relation | Trigger patterns |
|----------|-----------------|
| caused_by | "because", "caused by", "due to", "root cause" |
| fixed_by | "fixed", "repaired", "corrected", "patched" |
| replaced | "replaced", "superseded", "instead of" |
| contradicts | "contradicts", "inconsistent with", "conflicts with" |
| version_of | "v0.1", "version" |
| depends_on | "depends on", "requires", "needs" |
| improved | "improved", "from X to Y", "increased" |
| produced | "produced", "created", "generated", "wrote" |
| used_in | "used in", "part of", "component of" |
| tested_with | "tested", "benchmarked", "evaluated" |

Entities are extracted via 3 layers:
1. Project names (from known patterns)
2. Known concepts (60+ domain terms)
3. Dynamic: version numbers, percentages, file paths, function/class names

## Consolidation Pipeline

The main `consolidate()` function orchestrates the full pipeline:

1. `get_pending_traces()` — find unconsolidated .md files
2. Load and analyse each trace (metadata, entities, key lines, paragraphs)
3. `_cluster_traces()` — group by project + date proximity
4. For each cluster:
   - Aggregate metadata and entities
   - Write semantic memory with lifecycle state
   - Update entity graph with typed relations
5. `build_summary_dag()` — create/update hierarchical DAG
6. Mark traces as processed
7. Save consolidation stats

### Return value

```python
{
    "traces_processed": 5,
    "clusters_formed": 2,
    "memories_written": 2,
    "entities_found": 34,
    "entity_list": ["bm25", "cross-encoder", ...],
}
```

## Usage

```python
from consolidation_engine import (
    consolidate,
    age_memories,
    capacity_report,
    build_summary_dag,
    search_summary_dag,
)

# Run consolidation
stats = consolidate(force=True)
print(f"Processed {stats['traces_processed']} traces")

# Age memories (run periodically)
aging_stats = age_memories(reference_date="2024-06-01")
print(f"Transitioned: {aging_stats}")

# Check capacity
report = capacity_report()
for cat, info in report.items():
    if info["needs_consolidation"]:
        print(f"WARNING: {cat} at {info['usage_pct']:.1f}%")

# Search DAG
import json
from pathlib import Path
dag_path = Path("consolidation/summary_dag.json")
if dag_path.exists():
    dag = json.loads(dag_path.read_text())
    results = search_summary_dag(dag, "BM25 retrieval accuracy")
    for r in results:
        print(f"[L{r['level']}] {r['summary'][:80]}")
```

## Test Coverage

110 tests in `tests/test_consolidation_engine.py`:

- **Metadata extraction**: filenames, dates, projects, types
- **Entity extraction**: projects, concepts, dynamic (versions, paths)
- **Key lines**: trigger patterns, multi-line capture
- **Clustering**: project + date proximity, gap detection
- **Consolidation**: full pipeline with traces, empty dirs, force mode
- **Lifecycle**: write with state, active→stale, stale→archived, recent stays
- **Capacity**: basic report, threshold, empty, state counts
- **DAG**: build hierarchy, preserve traces, search, empty, serialisation
- **Performance**: DAG build <100ms, search <10ms
- **Graph**: typed relations, entity DB CRUD
- **Novelty**: cosine detection, degenerate inputs
- **Quality ratchet**: strict-mypy-clean production/test surfaces and audited
  helper docstrings for metadata, graph persistence, DAG, and scoring helpers

All 6 STRONG dimensions: empty, error, negative, pipeline, roundtrip, performance.

## API Reference

::: consolidation_engine.consolidate

::: consolidation_engine.age_memories

::: consolidation_engine.capacity_report

::: consolidation_engine.build_summary_dag

::: consolidation_engine.search_summary_dag
