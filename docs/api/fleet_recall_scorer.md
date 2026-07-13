# fleet_recall_scorer

Score the fleet-fed recall axis (category axis 3) from the recall query-stream
ledger — the measurement that turns "the fleet asks us questions" from a claim
into a number.

## Why this module exists

The query stream has existed twice for a while (`recall_ledger` is the durable
local sink, `bus_recall` mirrors the same events to the hub), but nothing
scored it: the stream was recorded, never measured, so the fleet-fed axis of
the world-class scorecard stayed permanently dark. This module is the scorer;
`world_class_scorecard` and `scorecard_report` fold its report in beside the
abstention, no-egress and lineage axes.

## Public surface

```python
from fleet_recall_scorer import (
    FleetRecallReport, report_from_ledger, score_fleet_recall,
)
```

- `score_fleet_recall(records)` — pure scorer over recall query records (as
  `RecallLedger.queries()` yields them, outcomes merged in).
- `report_from_ledger(path)` — score straight from the production ledger JSONL;
  a missing or empty ledger scores as an unmeasured axis, not an error.
- `FleetRecallReport` — frozen, JSON-serialisable axis report:
  `queries`, `agents`, `fleet_fed`, `answered`/`answered_rate`,
  `usage_labelled`/`usage_rate`, `correctness_labelled`/`fleet_accuracy`,
  `contradictions`, `measured`.

## Invariants

- **Label coverage is explicit.** `was_used` / `was_correct` labels arrive
  later than the query record and may never arrive; the report carries the
  labelled counts, so a rate over three labelled records cannot masquerade as
  a measured axis over a thousand.
- **Usage is not correctness.** `usage_rate` is the loop-closure precision
  proxy (`recall_outcome_tracker`); `fleet_accuracy` scores only the
  verifier-supplied `was_correct` label. Reported separately, never mixed.
- **Fleet-fed is objective.** `fleet_fed` = at least two distinct querying
  identities; a single-agent stream cannot claim the axis by volume.
- **Empty means dark.** No queries ⇒ `measured: false`, mirroring the
  harness's honest null-handling — never a perfect score over nothing.
- **Contradictions surface.** A record that both abstained and returned
  memories is counted (`contradictions`), not silently dropped.

## See also

- `recall_ledger` — the two-record append-only stream this scores.
- `bus_recall` — the fleet-bus mirror of the same stream.
- `world_class_scorecard` / `scorecard_report` — where the axis folds in
  (`--recall-ledger` on the `remanentia-scorecard` CLI).
