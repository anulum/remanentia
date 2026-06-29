# world_class_scorecard

The honest, comparable, category-defining scorecard for one benchmark run — the
capstone of the world-class evaluation harness (roadmap W1).

## Why this module exists

The field's headline numbers are non-comparable: different reader and judge LLMs
everywhere, and most report the inflated oracle setting (the Zep 84 % → 58 %
dispute is the proof). A world-class claim needs a scorecard that pins what makes
a number comparable and reports the axes no leaderboard scores.

Each scorecard records the **setting** (oracle vs realistic full-S — never
conflated), the **reader** and **judge** models (two runs compare only when both
match), and folds in the new-category metrics: calibrated abstention
(`coverage_accuracy`), sovereign no-egress (`no_egress_audit`), and
lineage-of-belief completeness (`lineage_completeness`). One deterministic,
JSON-serialisable record — the artefact REMANENTIA publishes to prove both a
cloud-comparable and a pure-local number on equal footing.

## Public surface

```python
from world_class_scorecard import RunConfig, Scorecard, build_scorecard, Setting
```

- `RunConfig(setting, reader, judge)` + `.comparable_to(other)` — comparability
  guard (setting ∧ reader ∧ judge must match).
- `build_scorecard(config, outcomes, egress, lineage, *, accuracy_target=0.90)`
  → `Scorecard` with accuracy, AURC, coverage-at-target, pure_local, cloud_calls,
  lineage_completeness, and `.as_dict()`.

## Invariants

- **Comparability is explicit.** Never compare across mismatched setting/reader/judge.
- **Oracle and full-S are distinct settings**, recorded, never conflated.
- **Deterministic**; composes the metric modules without model calls.

## See also

- `coverage_accuracy`, `no_egress_audit`, `lineage_completeness` — the folded axes.
- `plan_2026-06-29_sota_world_class_roadmap.md` — W1 context and targets.
