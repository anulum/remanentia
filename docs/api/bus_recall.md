# bus_recall

`bus_recall.py` mirrors recall-query events from the synchronous recall path to
the optional fleet bus. The bridge owns one background event loop, keeps recall
latency independent from hub availability, and degrades to a no-op when the bus
dependency is absent or explicitly disabled.

```python
from bus_recall import default_emitter

emitter = default_emitter()
if emitter is not None:
    emitter.emit("what changed in the API docs", returned_claim_ids=["docs:api"])
```

::: bus_recall
