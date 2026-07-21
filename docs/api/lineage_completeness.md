# lineage_completeness

Lineage-of-belief completeness: measure *why a fact is believed*. Part of the
multi-axis evaluation harness (roadmap W1/W5) for the auditability axis no public
memory leaderboard scores.

## Why this module exists

Final-answer accuracy cannot say which evidence supported a claim or how a belief
formed. The governance literature formalises it as **Provenance Visibility =
Queryable ∧ LineageComplete**: every memory an answer rests on must resolve to a
record, and that record's lineage must trace back to an originating write. This
module scores the fraction of answers that meet it.

## Public surface

```python
from lineage_completeness import (
    ProvenanceNode, AnswerLineage, LineageReport,
    is_lineage_complete, answer_provenance_visible, lineage_completeness,
)
```

- `ProvenanceNode(id, origin, parent)` — a provenance record linking toward its
  originating write (`origin=True` is a lineage root).
- `is_lineage_complete(node_id, store)` — the parent chain reaches an origin
  without dangling (missing node) or cycling.
- `answer_provenance_visible(answer, store)` — the answer cites ≥1 id and every
  cited id is lineage-complete.
- `lineage_completeness(answers, store) -> LineageReport(total, visible,
  completeness, incomplete_answers)`.

## Invariants

- **Queryable ∧ LineageComplete.** Missing node → not queryable → fails.
- **Cycle-safe.** A provenance cycle is incomplete, not an infinite loop.
- **Deterministic**, no model calls. Empty answer set is vacuously complete (1.0).

## See also

- `world_class_scorecard` — folds completeness into the comparable scorecard.
- `finding_envelope` / claim model — the provenance the lineage walks.
