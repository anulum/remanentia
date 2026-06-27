# recall_outcome_tracker

`recall_outcome_tracker.py` closes the recall-to-use loop by matching recently
returned memories against later written text from the same agent. It records
`was_used` labels in the recall ledger without requiring manual feedback for
every recall.

```python
from recall_outcome_tracker import RecallOutcomeTracker

tracker = RecallOutcomeTracker()
tracker.note_recall(event_id="q1", by="agent", returned_texts=["stable decision"], ledger=ledger)
used = tracker.note_text("The stable decision still applies.", by="agent", ledger=ledger)
```

::: recall_outcome_tracker
