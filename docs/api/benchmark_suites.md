# benchmark_suites

`benchmark_suites.py` provides deterministic operational and historical query
suites for local retrieval checks. The current suite reads the live index and
latest performance report when those artefacts exist, while the historical suite
keeps fixed regression questions for older benchmark incidents.

```python
from benchmark_suites import current_operational_queries, historical_regression_queries

queries = current_operational_queries()
```

::: benchmark_suites
