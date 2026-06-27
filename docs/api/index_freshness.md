# index_freshness

`index_freshness.py` checks whether the retrieval index is fresh relative to
its source firehose. It probes the configured stimuli, finding, digest, and
vector-index stages, then reports the drift so watchdog timers and status
commands can detect stalled consolidation.

```bash
python -m index_freshness --json
python -m index_freshness --report snn_state/index_freshness.json
```

::: index_freshness
