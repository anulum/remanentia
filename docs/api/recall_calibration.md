# recall_calibration

Calibrates recall abstention thresholds from correctness-labelled recall ledger
events. The gate estimates accepted-query correctness from neighbouring scored
examples, chooses the highest-coverage threshold that satisfies the configured
error budget, and reports held-out coverage, accuracy, and calibration error.

Run the operator report against the default recall ledger:

```bash
remanentia-recall-calibration --target-error-rate 0.1 --min-labelled 30
```

Emit machine-readable calibration evidence for automation:

```bash
remanentia-recall-calibration --ledger recall.jsonl --json
```

::: recall_calibration
