# judge_agreement

Measure how well a candidate judge reproduces the reference judge's
correctness labels — the evidence gate a local judge must pass before it may
replace the hosted one.

## Why this module exists

Every committed benchmark anchor was scored by the same hosted LLM-as-judge
(LongMemEval's evaluation protocol), which keeps the numbers comparable but
leaves one cloud dependency in an otherwise sovereign measurement chain: the
system needs an external call to know its own score. The way out is not to
swap judges on faith — a judge swap silently reprices every score — but to
*measure* a local judge against the hosted labels on identical prompts and
identical parsing, and let the agreement statistics decide. This module holds
those statistics; the re-judging loop lives in
`tools/local_judge_agreement.py`.

## Public surface

```python
from judge_agreement import JudgePair, agreement_payload, agreement_stats
```

- `agreement_stats(pairs)` — fold `(reference_label, candidate_label)` pairs
  into raw `agreement`, per-class `positive_agreement` /
  `negative_agreement`, and chance-corrected `cohen_kappa`.
- `agreement_payload(pairs, metadata=...)` — wrap the statistics into the
  committable artefact shape (schema version, benchmark tag, run metadata).
- `JudgePair` — `(bool, bool | None)`; `None` = the candidate produced no
  usable answer.

## Invariants

- **Chance cannot masquerade as skill.** A candidate that answers "yes" to
  everything matches the reference at the base rate; Cohen's kappa reports
  it as 0, so raw agreement alone can never justify a judge swap.
- **Unanswered is counted, never coerced.** A failed or empty candidate
  answer is recorded as unanswered — it neither inflates nor deflates the
  agreement over the answered pairs.
- **Empty means dark.** No answered pairs ⇒ `measured: false` (the honest
  null-handling convention), never a fabricated agreement.
- **Undefined is null.** A single-class comparison (chance = 1) has no
  defined kappa; the field is `null`, not an invented number.

## See also

- `tools/local_judge_agreement.py` — the runner that produces the pairs
  (same `_judge_prompt`, same yes/no rule as the reference pass).
- `benchmark_report` / `benchmark_manifest` — judge identity is part of run
  comparability (headlines pin setting AND reader; scores from different
  judges are claims, not facts).
