# benchmark_suites

`benchmark_suites.py` provides deterministic operational and historical query
suites for local retrieval checks. The current suite reads the live index and
latest performance report when those artefacts exist, while the historical suite
keeps fixed regression questions for older benchmark incidents.

Operational reports are expected to be JSON objects. Missing, invalid, or
non-object reports are treated as absent so dynamic query rows with unavailable
gold values are filtered instead of leaking malformed benchmark data into the
suite.

```python
from benchmark_suites import current_operational_queries, historical_regression_queries

queries = current_operational_queries()
```

::: benchmark_suites
