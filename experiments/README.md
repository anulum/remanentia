# Experiments

Historical A/B runs from the retrieval-tuning sprints (run_exp1 →
run_exp8). Each file is a self-contained script — different variants
of the retrieval stack, run against an earlier cut of the LongMemEval
oracle, preserved here for replayability and for the commentary
embedded in their docstrings.

These scripts are **not** part of the supported production surface:

- They import internals that have since been refactored (the scripts
  will need small adjustments to run against current ``master``).
- They predate the ``seed_utils`` / ``device_utils`` / ``file_utils``
  refactors and do not honour ``REMANENTIA_SEED`` or
  ``safe_device``.
- They pin stale configurations (``--arcane`` pre-R8, pre-TReMu
  prompt rules, pre-aggregation pre-compute).

For new A/B work, start from ``bench_longmemeval.py`` and add flags
there; use these scripts only as historical references.
