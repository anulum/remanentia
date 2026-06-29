# scorecard_report

The world-class evaluation harness wired to **real** benchmark output (roadmap W1
finish): turn a LongMemEval results JSONL into one comparable run report.

## Why this module exists

A world-class claim needs a number that is comparable — pinned to its setting
(oracle vs full-S), reader, and judge — and that reports the sovereign no-egress
axis. This module reads the rows the judge already writes (`judge_label`,
`judge_model`, `question_type`) and produces that report, using `no_egress_audit`
for the local-vs-cloud verdict from the reader endpoint.

Honesty over a full grid: the current bench emits neither per-question confidence
nor per-answer provenance, so calibrated abstention (`coverage_accuracy`) and
lineage-of-belief (`lineage_completeness`) cannot be scored from this file — the
report flags them `not measured` rather than fabricate an uncalibrated curve.
Populating them is a bench-instrumentation follow-up (emit confidence + cited
provenance ids per answer), after which those modules plug straight in.

## Public surface

```python
from scorecard_report import parse_results, ResultSummary, build_run_report, RunReport
```

- `parse_results(path) -> ResultSummary(total, correct, accuracy, judge_models)`
  — boolean `judge_label` is correctness; unjudged/blank rows are skipped;
  `judge_model` collected for judge-matched comparison.
- `build_run_report(results_path, *, setting, reader, reader_endpoints) ->
  RunReport` — folds accuracy + judge + the no-egress verdict; flags
  abstention/lineage as not-yet-instrumented.

## CLI

```
remanentia-scorecard RESULTS.jsonl [--setting oracle|full_s] [--reader MODEL] [--reader-endpoint URL ...]
```

Prints the comparable run report as JSON.

## See also

- `world_class_scorecard` — the in-memory scorecard this complements with the
  abstention/lineage axes (once the bench emits confidence + provenance).
- `no_egress_audit` — the sovereignty verdict folded in here.
- `plan_2026-06-29_sota_world_class_roadmap.md` — W1 context.
