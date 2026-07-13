# observed_context

LLM observe→reflect distillation of the retrieved sessions into a lean, dated,
supersession-aware observation set — the quality-extraction reader context that
replaces the raw transcript dump (roadmap W2-v2).

## Why this module exists

The deterministic sibling `lean_context` collapses retrieved atomic facts by
rule; measured on full-S that hurt (−7.6 pp) because a rule cannot tell which
dropped clause the reader still needs. The realistic-SOTA mechanism distils with
an LLM instead: read the raw exchange, write dated observations, mark
supersessions, feed those. The budget-minimal cloud ablation confirmed the lift
lands exactly on the categories it touches (multi-session +6.5 pp, preference
+16.7 pp). This module supplies that extraction while staying backend-agnostic.

## Public surface

```python
from observed_context import Completer, ObservedContext, build_observed_context
```

- `Completer` — structural protocol for the completion boundary:
  `(prompt, max_tokens) -> str | None`. The bench injects a callable routing to
  the hosted or the local model; this module imports no backend, so the
  sovereign no-egress path never gains a cloud call.
- `ObservedContext` — frozen result: `observations` (tuple of
  `lean_context.Observation`) + `rendered` (the header-prefixed block handed to
  the reader). Falsy when nothing was distilled.
- `build_observed_context(question, sessions, complete, *, max_observations=40,
  char_budget=8000, per_call_char_budget=100_000, max_tokens=700)` — pack
  sessions into chunks of at most `per_call_char_budget` characters, observe
  each chunk in one completion, merge observations across chunks (deduplicated
  by text, arrival order preserved), cap by count and by rendered budget.

## Invariants

- **Window-safe for local readers.** A large per-call budget reproduces the
  validated single-call cloud shape byte-for-byte; a small one yields one call
  per session or two, so a short-window local model is never handed a prompt it
  cannot read.
- **Fail-open to the raw dump.** Empty sessions, or every chunk returning a
  blank/failed/unparsable completion, yield an empty (falsy) context — the
  caller falls back to the raw transcript rather than answering from nothing.
- **Dates verbatim.** Observation dates are kept as the observer wrote them, so
  the reader meets the same date strings the transcript used.

## See also

- `lean_context` — the deterministic sibling and the `Observation` type.
- `reader_budget` — fits the distilled sections into a bounded reader window.
- `bench_longmemeval` — the consumer that wires this into the full-S run.
