# lean_context

Lean bi-temporal reader context — the W2 score lever. Replaces the raw-session
dump fed to the reader with a compact, dated, supersession-resolved set of
observations (Mastra OM / Engram mechanism).

## Why this module exists

The realistic-SOTA systems don't dump raw history into the reader; they feed a
lean, dated, compressed context and let bi-temporal ordering — not an LLM —
resolve conflicts (Engram: a ~9.6k-token bi-temporal slice beats full context by
+10.4 pts). REMANENTIA's honest full-S gap is reader-synthesis over a raw dump,
so this turns the retrieved atomic facts into a compact observation set that
**replaces** the reader's raw-session context. It is the failed P1.3
entity-summary done right: the lean context *is* the reader input, not a block
bolted on top of the dump.

## Public surface

```python
from lean_context import build_lean_context, LeanContext, Observation, BiTemporalFact
```

### `build_lean_context(facts, *, max_observations=40, char_budget=8000, drop_superseded=False) -> LeanContext`

`facts` is any sequence of `BiTemporalFact`-shaped objects (the retriever's
`AtomicFact` satisfies it via `text`/`valid_from`/`valid_until`/`date_mentions`/
`session_date`/`confidence`). Returns `LeanContext(observations, rendered)`:

- Duplicates (normalised text) collapse, keeping the highest-confidence instance.
- A fact with `valid_until` set is **superseded** — dropped if `drop_superseded`,
  else kept and marked `[superseded]`.
- Observations are ordered **newest-first** (current ahead of superseded at the
  same date) and capped by `max_observations` and `char_budget` to stay lean.
- `rendered` is the prompt block (header + dated lines), or `""` when empty.

## Wiring

Opt-in in `bench_longmemeval.py` via `REMANENTIA_LEAN_CONTEXT=1`: when on, the
reader's history block becomes the lean observation set instead of the raw
session dump (falls back to raw history when the lean set is empty). Off by
default until the full-S ablation validates it.

## Invariants

- **Deterministic**, no model calls — bi-temporal fields resolve conflicts.
- **Lean by construction** — deduplicated, capped, supersession-collapsed.
- **Never starves the reader** — empty lean set falls back to raw history.

## See also

- `fact_decomposer.AtomicFact` — the bi-temporal fact the builder consumes.
- `world_class_scorecard` — measures whether lean context moves the number.
- `plan_2026-06-29_sota_world_class_roadmap.md` — W2 context and target (≥80%).
