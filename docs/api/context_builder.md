# context_builder

`context_builder.py` organises retrieved atomic facts into prompt layers for
LLM synthesis: top-of-mind, work context, stable facts, and background
knowledge. Callers can pass session dates and a reference date to keep the
prompt ordering tied to the same temporal frame as retrieval.

```python
from context_builder import build_hierarchical_context

context = build_hierarchical_context(facts, reference_date="2026-06-27")
prompt_section = context.to_prompt_string()
```

::: context_builder
