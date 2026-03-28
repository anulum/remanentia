# fact_decomposer

Decomposes conversation turns into atomic facts with temporal validity windows.

Each fact has: text, session index, fact type (state/event/preference/plan), valid_from/valid_until dates, entities, and supersession tracking.

::: fact_decomposer.FactIndex
    options:
      show_source: true
      members_order: source

::: fact_decomposer.AtomicFact

::: fact_decomposer.decompose_sessions
