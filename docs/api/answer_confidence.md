# answer_confidence

Two per-answer confidence signals for the calibrated-abstention axis (roadmap W1
producer side). Pure parsing/normalisation; no model calls.

## Why this module exists

Calibrated abstention asks whether a system knows when it does not know. Scoring
it (`coverage_accuracy`) needs a per-answer confidence, and the honest design
emits two complementary signals so the scorecard can use either:

1. **Reader self-report** — the reader LLM ends its answer with a
   `CONFIDENCE: 0.83` line, parsed off and clamped to [0, 1]. Principled (the
   reader rates its own certainty) but present only when the model complies.
2. **Retrieval-score proxy** — the top reranker/BM25 score squashed to [0, 1].
   Always available, no prompt change; weaker across heterogeneous score scales
   but robust.

This module holds the logic so the benchmark harness wires thin calls; the logic
is tested here.

## Public surface

```python
from answer_confidence import confidence_suffix, parse_confidence, normalise_score
```

- `confidence_suffix() -> str` — the instruction appended to the reader prompt to
  elicit a `CONFIDENCE: x` rating.
- `parse_confidence(answer) -> (clean_answer, confidence | None)` — splits a
  trailing rating off the answer, clamped to [0, 1]; `None` when absent.
- `normalise_score(score) -> float` — logistic squash of a raw retrieval score
  into [0, 1] (a calibrated relevance probability for a cross-encoder logit; a
  monotone proxy for BM25), overflow-guarded.

## Wiring

In `bench_longmemeval.py`, opt-in via `REMANENTIA_EMIT_CONFIDENCE=1`: appends the
suffix to the reader prompt and emits `confidence`, `retrieval_confidence`, and
`cited_ids` per question. Off by default so runs stay comparable to history.
`scorecard_report` then scores the abstention + lineage axes.

## See also

- `scorecard_report` — consumes these fields to score the axes.
- `coverage_accuracy` — the risk-coverage evaluator behind the abstention axis.
