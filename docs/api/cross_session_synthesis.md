# cross_session_synthesis

Consolidate the retrieved facts about a question's named entities into a single
`ENTITY SUMMARY` block before the LLM reads the context. Operates only over
facts the retriever already surfaced, so it adds no extra API call; pure-Python
with an optional Rust path for the entity-matching primitive.

## Why this module exists

The deterministic aggregation pre-compute (`aggregate_precompute`) only fires
for SUM and COUNT questions. On the honest full-S LongMemEval benchmark the
larger multi-session failure class is *synthesis*, not arithmetic. The failure
attribution (`benchmark_2026-06-21_p11_failure_attribution.md` §4–§5) measured
recall@10 at 88 % for multi-session — the gold sessions are usually retrieved —
yet accuracy was ~42 %. The reader now sees ~10 sessions (only ~2 of them gold),
and loses the thread when it has to gather and reconcile statements about the
same entity that are scattered across those sessions, sometimes superseded.

This module is the reflection / entity-summary lever for that class (TODO P1.3).
It groups the already-retrieved facts by the entities the question actually
names, deduplicates them, orders them oldest→newest, and marks the latest dated
statement `(most recent)` — the exact signal the knowledge-update reader needs
("use the most recent answer"). The reader then has a compact, organised record
to consult instead of re-aggregating ten raw sessions.

## Design stance

It mirrors `aggregate_precompute`: deliberately conservative, high-precision,
and deterministic. It returns `None` — adding nothing to the prompt — whenever:

1. The question names no entity that the retrieved facts mention.
2. No named entity has at least two distinct statements (a single statement
   carries no cross-session synthesis value).
3. The question type is `temporal-reasoning` (those already receive a dedicated
   dated timeline; an entity summary would be redundant noise).

Over-firing would add the very reader-distracting noise the module targets, so
the bar to emit a block is intentionally high.

## Public surface

```python
from cross_session_synthesis import (
    synthesise,
    focus_entities,
    SynthFact,
    StatementLine,
    EntityDigest,
    SynthesisResult,
)
```

### `synthesise(question, facts, *, qtype="", max_entities=4, max_statements=4, char_budget=1400) -> SynthesisResult | None`

Top-level entry point used by the bench. `facts` is any sequence of
`SynthFact`-shaped objects (the retriever's `AtomicFact` satisfies it). Returns
a `SynthesisResult` whose `.message` is the `ENTITY SUMMARY …` block ready to
prepend to the reader context, or `None` when nothing is worth consolidating.
The block is bounded by `max_entities`, `max_statements` per entity, and an
overall `char_budget`.

### `focus_entities(question, entities) -> list[str]`

Return the subset of `entities` the `question` explicitly grounds — an entity
qualifies when at least one of its significant tokens (length ≥ 4, not a
stopword) appears in the question. Case-insensitive duplicates are collapsed and
first-appearance order is preserved.

### `SynthFact` (Protocol)

Structural shape of an atomic fact: `text`, `session_idx`, `role`, `fact_type`,
`valid_from`, `entities`, `date_mentions`, `session_date`. Declared structurally
(runtime-checkable) so the module never imports the retriever and cannot create
an import cycle.

### `StatementLine` / `EntityDigest` / `SynthesisResult`

Dataclasses for one consolidated statement (date, session, text), all statements
about one entity, and the whole rendered result. `SynthesisResult` is falsy when
it carries no entities.

## Rust fast path

`cross_session_synthesis` attempts to import `remanentia_cross_session_synthesis`
at module load. When present, `focus_entities` delegates the tokenisation /
entity-matching primitive to the Rust binding; the pure-Python path is the
documented fallback floor and runs on `ImportError`.

The repository does not currently track a Rust source crate for this extension
(the same state as `remanentia_aggregate_precompute`, whose compiled artefact is
installed but whose source is not in the tree). The Rust acceleration is
therefore a stable extension point, not yet a built artefact; the Python path is
the active implementation and the benchmark-validated one.

## Out of scope

- **Entity discovery beyond the question.** Entities the question does not name
  (e.g. "which restaurants have I visited", where the names are not in the
  query) are not consolidated; that is the category-enumeration case handled, in
  part, by `aggregate_precompute`'s distinct-count path.
- **Arithmetic.** Sums and counts stay with `aggregate_precompute`.
- **Temporal timelines.** Routed to the dated-timeline path in the bench.
- **New retrieval.** The module never fetches; it only reorganises facts the
  retriever already returned.

## Invariants

- **Deterministic.** Same question + same facts → same block, byte for byte.
- **No extra API call.** Works over retrieved facts only; no per-question cost.
- **Never fabricates.** It only reorders and deduplicates existing fact text; it
  never synthesises new claims.
- **Bounded.** Output is capped by entity count, statements per entity, and a
  character budget, with the most-evidenced entities first.

## Usage pattern

```python
from cross_session_synthesis import synthesise

synth = synthesise(question, [r.fact for r in results], qtype=qtype)
header = f"{synth.message}\n\n" if synth else ""
prompt = f"{header}{retrieved_facts}\n\n{full_context}\n\n{question}"
```

## See also

- [`aggregate_precompute`](aggregate_precompute.md) — sister module; the
  deterministic SUM/COUNT precompute, same conservative + Rust-first stance.
- `bench_longmemeval.py` — primary consumer (`_arcane_answer`, env knob
  `REMANENTIA_SYNTHESIS_DISABLE=1` for ablation runs).
- `fact_decomposer.AtomicFact` — the concrete fact type that satisfies
  `SynthFact`.
