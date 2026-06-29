# coverage_accuracy

Risk–coverage (calibrated-abstention) evaluation: measuring whether the system
**knows what it doesn't know**. Part of the world-class evaluation harness
(roadmap W1/W5) for an axis no public memory leaderboard scores.

## Why this module exists

Every memory leaderboard reports final-answer accuracy and nothing else. A
world-class system must also abstain well — answer confidently where it has
support, and say "I don't have that" where it does not. This module turns
per-question outcomes into a **risk–coverage curve**: sort answers by confidence,
sweep a threshold, and at every *coverage* (fraction answered) measure the
*accuracy* over the answered subset. A well-calibrated system's accuracy rises as
coverage falls — it abstains on exactly what it would get wrong.

It is pure and deterministic — no model calls — so it scores the result records
the benchmark already produces and composes into the honest evaluation harness
with no cloud dependency.

## Public surface

```python
from coverage_accuracy import Outcome, risk_coverage, RiskCoverage, CoveragePoint
```

### `risk_coverage(outcomes) -> RiskCoverage`

`outcomes` is a sequence of `Outcome(correct: bool, confidence: float)`. Returns
the curve plus summaries:

- `accuracy_at_full_coverage` — accuracy when you answer everything.
- `aurc` — area under the risk–coverage curve (lower = better calibrated).
- `points` — `CoveragePoint(coverage, answered, accuracy, risk)` per prefix.
- `coverage_at_accuracy(target)` — the maximum coverage whose answered subset
  stays at or above a target accuracy: "how much can I answer while holding this
  quality bar?" Returns `0.0` if no prefix reaches the target.

## Invariants

- **Deterministic.** Same outcomes → same curve; confidence ties keep input order.
- **No model calls.** Operates on result records only.
- **Empty-safe.** No outcomes → zeroed result, not an error.

## See also

- `recall_calibration` — fits an abstention threshold from `RecallLedger`
  outcomes (recall-side calibration); this module is the benchmark-QA-side curve.
- `plan_2026-06-29_sota_world_class_roadmap.md` — the W1/W5 context.
