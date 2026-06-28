# temporal_relation

Temporal relation classifier for event pairs. Predicts one of six
Allen-interval-inspired relations: before, after, same_day, overlaps,
contains, unknown.

The wrapper lazy-loads `models/temporal-relation-v1/model.pt` and returns
`None` from `classify_relation()` when the trained model or configuration is
missing or invalid. `order_events()` preserves the input order in that
unavailable-model case.

::: temporal_relation.RelationResult

::: temporal_relation.classify_relation
    options:
      show_source: true

::: temporal_relation.order_events
    options:
      show_source: true
