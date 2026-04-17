# Model Card — Date Normaliser (C4)

> **Status: RULE ENGINE PROVEN; ML WEAK BUT UNUSED BY DEFAULT.** The
> rule-based 12-pattern regex normaliser is the only component in the
> C1-C5 training programme with a committed, measured impact (+66.9 pp
> date coverage on LongMemEval sessions). The ML companion model
> achieved 24.8 % exact / 65.7 % relaxed (±3 d) and is redundant in
> production because the rules already cover >90 % of observed
> expressions.

## What this component is for

Resolve vague date expressions in conversation text ("three weeks
ago", "last Tuesday", "recently", "yesterday") into concrete
ISO-8601 dates relative to a reference timestamp.

Used by:

- `date_normalizer.normalise_in_context` (production path)
- `bench_longmemeval.py::_build_index_for_question` per-session wiring

## Architecture — Rule Engine (the one that ships)

A 12-pattern regex set, deterministic, zero ML:

| Family | Example | Resolution |
|--------|---------|------------|
| Quantified offset | "3 weeks ago" | ref − 3×7 d |
| Weekday | "last Tuesday" | nearest past Tue |
| Vague | "recently" | ref − 14 d, confidence 0.4 |
| Absolute | "March 15, 2026" | ISO passthrough |
| Relative | "yesterday", "tomorrow" | ref ± 1 d |
| Quarter | "last quarter" | previous 3-month window |
| … | … | … |

Each pattern produces a `(date, confidence)` pair so downstream code
can thresh.

## Architecture — ML companion (weak, unused by default)

- Base: `distilbert-base-uncased`
- Head: regression onto day-offset from reference
- Script: `training/train_date_normalizer.py`

## Evaluation

### Rule engine

| Metric | Value |
|--------|------:|
| Deterministic on training corpus | yes (no stochasticity) |
| LongMemEval dated-fact coverage (before wiring fix) | 23.3 % |
| LongMemEval dated-fact coverage (after rules + wiring) | **90.2 %** |
| Gain | **+66.9 pp** |

The wiring fix (passing `haystack_dates` through
`decompose_sessions`) was an architectural bug; the rule engine was
the thing that then actually consumed the reference timestamps.

### ML model (informational)

| Metric            | Value  |
|-------------------|-------:|
| Exact match       | 24.8 % |
| Relaxed (±3 days) | 65.7 % |

The ML model is redundant: rules already fire on >90 % of encountered
expressions with higher confidence.

## Downstream impact

Per GPT-4o-mini benchmark (2026-03-29), the rule engine drove
temporal-reasoning from **45.9 % → 60.2 % (+14.3 pp)** on LongMemEval.
This is the single biggest measured contribution of the C1-C5
programme.

## Limitations

- English only. No localisation to Slovak / German / French / CJK
  date phrases. `P5` multilingual-support gap tracked separately.
- Vague expressions like "recently" map to a 14-day window with
  low confidence — downstream consumers should treat such outputs
  as soft signals, not exact dates.
- The model has no ambiguity resolution: "last Tuesday" on a Tuesday
  returns the prior week's Tuesday.

## Reproduction

```python
from date_normalizer import normalise_in_context
normalise_in_context("We met 3 weeks ago.", ref_date="2023-04-10")
# → "We met 2023-03-20."
```

```bash
# Re-train the ML companion (optional, not used in production):
cd training
CUDA_VISIBLE_DEVICES=3 python train_date_normalizer.py
```
