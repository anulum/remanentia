# scorecard_report

The world-class evaluation harness wired to **real** benchmark output (roadmap W1
finish): turn a LongMemEval results JSONL into one comparable run report.

## Why this module exists

A world-class claim needs a number that is comparable — pinned to its setting
(oracle vs full-S), reader, and judge — and that reports the sovereign no-egress
axis. This module reads the rows the judge already writes (`judge_label`,
`judge_model`, `question_type`) and produces that report, using `no_egress_audit`
for the local-vs-cloud verdict from the reader endpoint.

The two new-category axes activate the moment the bench emits their data and stay
honestly dark until then — no fabricated curve. When **every** judged row carries
a numeric `confidence`, the calibrated-abstention axis is scored via
`coverage_accuracy` (accuracy at full coverage, AURC, coverage retained at a
target accuracy). When **every** judged row carries a `cited_ids` list, the
lineage-of-belief axis is scored via `lineage_completeness` (the fraction of
answers that rest on queryable provenance). A run missing a field reports that
axis `not measured` rather than guess. The remaining producer-side work is the
bench emitting `confidence` + `cited_ids` per question.

## Public surface

```python
from scorecard_report import parse_results, ResultSummary, build_run_report, RunReport
```

- `parse_results(path) -> ResultSummary(total, correct, accuracy, judge_models)`
  — boolean `judge_label` is correctness; unjudged/blank rows are skipped;
  `judge_model` collected for judge-matched comparison.
- `build_run_report(results_path, *, setting, reader, reader_endpoints,
  accuracy_target=0.90) -> RunReport` — folds accuracy + judge + the no-egress
  verdict, and (when the data is present) the calibrated-abstention axis (`aurc`,
  `coverage_at_target`) and the lineage axis (`lineage_completeness`), each with
  its `*_measured` flag.

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
