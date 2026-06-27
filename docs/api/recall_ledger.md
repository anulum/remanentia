# recall_ledger

`recall_ledger.py` stores recall queries and later outcome labels as append-only
JSONL records. The ledger is the local calibration substrate for recall
abstention, usage tracking, and query-weighted retrieval monitoring.

```python
from recall_ledger import RecallLedger

ledger = RecallLedger("runtime/recall_ledger.jsonl")
event_id = ledger.record_query("what did the benchmark report change?", returned_ids=["b:1"])
ledger.record_outcome(event_id, was_used=True)
```

::: recall_ledger
