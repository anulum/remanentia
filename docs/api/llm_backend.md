# llm_backend

`llm_backend.py` defines the pluggable completion interface used by recall and
MCP synthesis paths. It provides local, hosted, automatic, and null backends so
callers can select a configured model without coupling retrieval code to one
provider SDK.

The implementation treats every external boundary as optional and fallible:
missing TOML parsers fall back to defaults, malformed OpenAI-compatible
responses return `None`, and hosted SDK imports are resolved lazily so offline
retrieval remains usable.

```python
from llm_backend import resolve_backend

backend = resolve_backend("auto")
answer = backend.complete("Summarise the retrieved facts.", max_tokens=120)
```

## Backends

| Name | Implementation | Behavior |
|---|---|---|
| `auto` | `AutoBackend` | Try local `/models`, then hosted Anthropic if `ANTHROPIC_API_KEY` is set, then `NullBackend`. |
| `local` | `LocalLLMBackend` | POST to an OpenAI-compatible `/chat/completions` endpoint. |
| `anthropic` | `AnthropicBackend` | Use the optional `anthropic` SDK with lazy client creation. |
| `none` | `NullBackend` | Return `None` for explicit no-LLM operation. |

Focused validation for this surface is in `tests/test_llm_backend.py`; it
exercises the local HTTP request path, malformed response handling, hosted SDK
factory handling, auto-selection, config loading, and answer-extractor/MCP
wiring at 100% isolated line coverage.

::: llm_backend
