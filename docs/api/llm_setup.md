# llm_setup

`llm_setup.py` contains local-LLM setup helpers for hardware detection, model
recommendation, and configuration-file generation. The CLI commands use these
helpers when preparing a local completion backend for recall synthesis.

```python
from llm_setup import recommend_model, write_config

model = recommend_model(vram_gb=8.0, ram_gb=32.0)
write_config(backend="local", local_model=model.name if model else "gemma3:4b")
```

::: llm_setup
