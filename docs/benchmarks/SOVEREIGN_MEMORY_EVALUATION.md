# Sovereign Memory Evaluation

Long-term-memory leaderboards report one number: final-answer accuracy, usually on
an oracle retrieval setting with a cloud reader and a cloud LLM-as-judge. That
number is necessary but not sufficient. A memory system that is meant to run under
governance — locally, auditable, honest about what it does not know — has
properties the leaderboards do not measure at all. This document defines the axes
Remanentia measures in addition to accuracy, and how each is measured, so the
claims are reproducible rather than rhetorical.

Nothing here replaces accuracy. It surrounds it.

## Methodology guardrails

Two runs' accuracy numbers are comparable only when three things match, so every
result records them (`world_class_scorecard.RunConfig`):

1. **Setting — oracle vs realistic full-S, never conflated.** The oracle setting
   (the gold evidence sessions are handed to the reader) inflates by roughly 30
   points over full-haystack retrieval. Remanentia reports full-S as the headline
   and marks oracle explicitly when shown.
2. **Reader** — the answer-generating model. A cloud `gpt-4o-mini` answer and a
   local `gemma3:4b` answer are different systems, not one system's two runs.
3. **Judge** — the LLM-as-judge that marks correctness. Cross-vendor headline
   numbers use different judges and are treated as claims, not facts.

Report variance, never a single run.

## The axes

Each axis is a metric module, pure and deterministic (no model calls), scoring
records the benchmark already produces. `scorecard_report.build_run_report` folds
them into one comparable record; an axis whose input is absent reports
`not measured` rather than a fabricated value.

### 1. Sovereign no-egress accuracy
*What:* full-S accuracy with **zero** cloud LLM in the loop — reader, and where
possible retrieval, running locally — plus the honest accuracy cost of staying
local. *How:* `no_egress_audit.audit_endpoints` inspects the reader endpoints and
records `pure_local` and `cloud_calls`; the scorecard pairs that with the accuracy.
*State:* first measurement — sovereign `gemma3:4b` on an AMD ROCm GPU scored
**35.4 %** full-S, **−21 pp** against the ~56.6 % cloud `gpt-4o-mini` baseline on
the same retrieval. The gap is reader synthesis, not retrieval; a stronger local
reader is the open lever.

### 2. Write-discipline → accuracy
*What:* structured-ingestion quality and its ceiling on everything downstream — a
memory answer cannot beat what the write side recorded. *How:* `write_discipline`
gates ingestion against the canonical schema; `discipline_impact` relates schema
compliance to retrievability. *State:* a first audit found ~15 % of fleet stimuli
satisfied the schema (dominant failure: missing timestamp, fatal for a bi-temporal
store); the gate now exists. The compliance→accuracy curve is an open research
question the harness is built to quantify.

### 3. Calibrated abstention
*What:* knowing what you don't know — answering where there is support and
abstaining where there is not — instead of maximising recall. *How:*
`coverage_accuracy.risk_coverage` sweeps a confidence threshold to produce the
risk–coverage curve, the area under it (AURC; lower is better), and the maximum
coverage attainable at a target accuracy; `render_curve_jsonl` emits the plottable
curve. *State:* machinery complete; awaiting a confidence-bearing run to publish a
curve over real data.

### 4. Fleet-fed multi-producer recall
*What:* recall when many independent producers write into one principal-scoped
memory, not a single authored corpus. *How:* `bus_recall` scores retrieval over
multi-producer, principal-scoped records. *State:* instrument present; a
multi-producer evaluation set is the pending input.

### 5. Lineage-of-belief auditability
*What:* every answer traces to queryable, lineage-complete provenance —
*Provenance Visibility = Queryable ∧ LineageComplete*. *How:*
`provenance_export` projects the knowledge-store belief graph (a note's
`derived_from` link → parent toward origin; a `stated` note → originating write)
onto the provenance store that `lineage_completeness` scores; an answer is
provenance-visible iff it cites at least one id and every cited id resolves and its
chain reaches an originating write. An inferred belief with no recorded derivation
is correctly scored incomplete — the auditability gap the axis exists to expose.
*State:* export and scorer complete and round-trip-tested; awaits a run whose
answers cite knowledge-store ids to score end-to-end.

## Composition and honest status

`world_class_scorecard` / `scorecard_report` combine accuracy with the axes above
into one JSON record that pins setting, reader, and judge. As of this writing the
harness reproduces the two accuracy anchors from real result files — cloud
full-S ~56.6 % (`gpt-4o-mini`) and sovereign full-S 35.4 % (`gemma3:4b`, no
egress) — and the abstention, lineage, and write-discipline axes are implemented
and tested but report `not measured` until a confidence- and citation-bearing run
completes. Those runs are currently gated by hardware (RAM/GPU contention), not by
missing tooling. The targets are a cloud-comparable full-S ≥ 80 % and a pure-local
70–75 %, with the local-first cost stated, not assumed.
