# reader_budget

Priority-ordered context fitting for bounded reader windows — keeps the
answer-critical sections whole and clips only the raw-history tail.

## Why this module exists

A cloud reader has a ~128 k-token window, so the benchmark can send the whole
retrieved context in one call. A sovereign local reader has a short window (a
llama.cpp server pinned at 8192 tokens); handed a ~110 k-char prompt it does not
answer — llama.cpp stalls on an over-window prompt rather than truncating, which
hangs the run, while Ollama silently truncates and answers from a clipped tail.
Neither is acceptable: the run must neither hang nor lose the sections the
reader needs most.

## Public surface

```python
from reader_budget import fit_context
```

- `fit_context(sections, budget_chars)` — given context sections in priority
  order (most answer-critical first: computed totals, distilled observations,
  retrieved facts, raw history last) and a character budget, keep whole
  sections until the budget, truncate the section that would overflow, drop the
  rest. The budget is the reader window expressed in characters (tokens ×
  chars-per-token, minus instruction and reserved answer).

## Invariants

- **Non-positive budget = unbounded.** The cloud path passes 0/negative and
  receives every section verbatim — cloud behaviour is unchanged by wiring this
  in.
- **Priority survives, tail is clipped.** Answer-critical sections always
  survive intact; only the lowest-priority overflow is truncated or dropped.
- **Deterministic and pure** — no model calls, no environment reads.

## See also

- `observed_context` — produces the high-priority distilled section.
- `bench_longmemeval` — computes the budget from the local reader window and
  assembles the sections in priority order.
