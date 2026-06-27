# llm_backend

`llm_backend.py` defines the pluggable completion interface used by recall and
MCP synthesis paths. It provides local, hosted, automatic, and null backends so
callers can select a configured model without coupling retrieval code to one
provider SDK.

```python
from llm_backend import resolve_backend

backend = resolve_backend("auto")
answer = backend.complete("Summarise the retrieved facts.", max_tokens=120)
```

::: llm_backend
