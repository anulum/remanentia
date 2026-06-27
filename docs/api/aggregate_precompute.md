# aggregate_precompute

Pre-compute `COMPUTED TOTAL:` and `COMPUTED COUNT:` lines before
the LLM sees the retrieved context. Rust-accelerated hot path; pure-
Python fallback when the extension is absent.

## Why this module exists

Cross-session aggregation is the single largest failure class on
LongMemEval R11: 14 of 61 failures are arithmetic errors or missed
aggregations where the LLM had the right parts but produced the wrong
total (the infamous "YouTube 542 + TikTok 1456 = 2098" case, gold
1998). The LLM knows `a + b = a + b` in principle; it just loses the
thread on long contexts.

The module runs a deterministic precompute *before* the LLM so the prompt can
contain explicit lines such as `COMPUTED TOTAL: YouTube: 542 views + TikTok:
1456 views = 1998 views` or `COMPUTED COUNT: Dr. Smith, Dr. Patel, Dr. Lee = 3
distinct doctors`. The downstream LLM is told to trust the line unless it
clearly contradicts the question.

The module is deliberately conservative — it refuses to emit a total
unless the evidence is unambiguous:

1. The question is phrased with a strong total cue (`total`,
   `combined`, `altogether`, `how much total`).
2. At least two same-unit labelled numbers are in the retrieved text.
3. The units are compatible (views with views, dollars with dollars).

For counts, the module is even narrower: it emits a count only for explicit
`different` / `unique` / `distinct` questions and only when the context exposes a
stable item list such as doctor names or an explicit distinct-item list.
Over-aggressive matching does more harm than good.

## Public surface

```python
from aggregate_precompute import (
    is_sum_question,
    is_count_question,
    extract_numeric_facts,
    extract_count_facts,
    precompute_sum,
    precompute_count,
    precompute_aggregation,
    NumericFact,
    CountFact,
    PrecomputeResult,
)
```

### `is_sum_question(question: str) -> bool`

True for questions phrased as sum aggregations. Matches
`total`, `combined`, `altogether`, `how much total`, `what is the total`,
`adding up/together`. False for every other shape.

### `is_count_question(question: str) -> bool`

True for `how many X` questions **only when** the question does not
also match `is_sum_question` (sum phrasing wins; `total items`
routes to the sum path).

### `extract_numeric_facts(text: str) -> list[NumericFact]`

High-precision, low-recall scan for `<label>: <number> <unit>` triples
and currency shapes. The label must start with an uppercase letter
followed by at least two lowercase letters, so `it was 42` and `YT: 5`
are not extracted. The walk-back window is 80 chars, which covers
`Your YouTube tutorial has 542 views` without drifting into the
previous sentence.

### `extract_count_facts(question, text) -> list[CountFact]`

High-precision distinct-item scan for explicit count questions. It currently
accepts named doctors (`Dr. Smith`) and explicit lists such as `Different
cuisines: Italian, Mexican, Thai`. Generic `how many` questions without
`different`, `unique`, or `distinct` return an empty list.

### `precompute_sum(question, text) -> PrecomputeResult | None`

End-to-end path. Returns `None` unless the question is a sum question
*and* the text yields at least two same-unit facts. On success,
`PrecomputeResult.message` is the `COMPUTED TOTAL: …` string ready
to prepend to the LLM prompt.

### `precompute_count(question, text) -> PrecomputeResult | None`

End-to-end distinct-count path. Returns `None` unless the question has an
explicit distinct-count cue and the context yields at least two stable items. On
success, `PrecomputeResult.message` is the `COMPUTED COUNT: …` string ready to
prepend to the LLM prompt.

### `precompute_aggregation(question, text, *, qtype="") -> PrecomputeResult | None`

Top-level entry point used by the bench. Dispatches to `precompute_sum`, then
`precompute_count`, for non-temporal, non-abstention qtypes and returns `None`
for off-scope qtypes so the caller can short-circuit cleanly.

## Rust fast path

`aggregate_precompute` imports `remanentia_aggregate_precompute` at
module load. When present, `is_sum_question`, `is_count_question`,
`extract_numeric_facts`, and `precompute_sum` all delegate to the Rust
bindings; the Python fallback runs only on `ImportError`. **~11.3×
faster** than Python on LongMemEval-shaped inputs. Both paths return
identical output; parity tests in
`tests/test_aggregate_precompute.py::TestRustPythonParity` compare
them on every CI run.

Distinct-count precompute is currently Python-side only because the repository
does not track a Rust source crate for `remanentia_aggregate_precompute`. The
existing Rust parity surface remains enforced for the sum fast path.

## Out of scope

- **Entity supersession chains** (R11 audit §8). Needs
  `knowledge_store` integration to collapse "my old YouTube" vs. "my
  new YouTube" into a single entity.
- **Abstention gating.** `_abs` questions are skipped upstream; a
  pre-computed total would undermine the abstention signal.
- **Mixed-unit totals.** If a retrieval returns views + followers +
  dollars, the module refuses and lets the LLM sort it out.
- **Implicit count inference.** Generic "how many" questions without an explicit
  distinct/unique cue stay on the LLM fallback unless they are already sum
  questions.
- **Temporal reasoning.** Routed to `_tremu_precompute`, the temporal
  peer module.

## Invariants

- **Deterministic.** Same question + same text → same output, byte for
  byte. No RNG, no floating-point drift (ints are printed as ints).
- **UTF-8 safe.** Walk-back snaps to the nearest char boundary, so
  multi-byte prefixes (¥, €, CJK, emoji) never panic the Rust path.
- **Never fabricates.** If extraction fails, return `None` — never a
  best-guess total. The LLM remains the fallback for everything we
  refuse.
- **Unit coherence.** The dominant unit with ≥ 2 facts wins; single
  outliers are ignored rather than mixed in.

## Usage pattern

```python
from aggregate_precompute import precompute_sum

context = build_retrieval_context(results)
pre = precompute_sum(question, context)
if pre is not None:
    prompt = f"{pre.message}\n\n{context}\n\n{question}"
else:
    prompt = f"{context}\n\n{question}"
```

## See also

- [`pii_redactor`](pii_redactor.md) — sister module, same Rust-first
  + Python-fallback pattern.
- `bench_longmemeval.py` — primary consumer.
- Optional `remanentia_aggregate_precompute` PyO3 extension — Rust
  implementation used when installed.
