# discipline_impact

Quantify the write-discipline ceiling on retrieval (roadmap W4): how much recall
is lost when a memory is written without its canonical fields.

## Why this module exists

REMANENTIA's new-category claim is that **write-side discipline caps retrieval** —
a memory written without its canonical fields can never be retrieved as well as
one written with them, however good the retriever. The fleet audit found the
dominant failure is a missing timestamp (~15% of stimuli conform). This module
measures that cost: it compares retrieval recall computed with the canonical
timestamps against recall with the timestamps stripped, and reports the per
question-type drop — the discipline ceiling.

The retrieval runs live in the CPU-bound recall harness
(`tools/discipline_recall.py`); this module holds the pure comparison so the
ceiling is computed and tested deterministically.

## Public surface

```python
from discipline_impact import discipline_impact, RecallDelta, worst_hit
```

- `discipline_impact(canonical, degraded, ns=(1,3,5,10,20)) -> list[RecallDelta]`
  — compares two `aggregate_recall`-shaped maps (`{qtype: {"mean@N": float}}`);
  only shared qtypes, `"overall"` first. Raises `KeyError` on a missing `mean@n`.
- `RecallDelta(qtype, canonical, degraded, delta)` with `.delta_at(n)` — recall
  lost to the degraded write at cutoff `n` (positive = lost).
- `worst_hit(impacts, n=10) -> RecallDelta | None` — the non-overall qtype most
  hurt by the degradation (expected: temporal-reasoning).

## Runner

`tools/discipline_recall.py` runs recall with timestamps on and off and prints
the ceiling (NICED, no API): `nice -n 19 python tools/discipline_recall.py --limit 50`.

## See also

- `write_discipline` — the enforced gate whose ceiling this measures.
- `tools/retrieval_recall.py` — the recall machinery reused by the runner.
- `plan_2026-06-29_sota_world_class_roadmap.md` — W4 context.
