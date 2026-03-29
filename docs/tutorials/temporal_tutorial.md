# Tutorial: Temporal Queries and Date Reasoning

Remanentia extracts dates from documents and builds a temporal event graph. This enables queries like "what happened before X", "how many days between events", and "what was the most recent decision".

## Building a Temporal Graph

```python
from temporal_graph import TemporalGraph, TemporalEvent

tg = TemporalGraph()

# Extract events from document text
events = tg.extract_events(
    "The project started on 2026-01-15. The v1.0 release happened on 2026-03-20.",
    "project.md",
)
tg.add_events(events)

print(f"Events: {tg.stats['events']}")
print(f"Unique dates: {tg.stats['unique_dates']}")
print(f"Date range: {tg.stats['date_range']}")
```

### Supported Date Formats

The date parser handles:

| Format | Example | Extracted |
|--------|---------|-----------|
| ISO | `2026-03-15` | `2026-03-15` |
| Written (full) | `March 15, 2026` | `2026-03-15` |
| Written (abbrev) | `Jan 5, 2026` | `2026-01-05` |
| Written (no year) | `March 20` | Uses default year |
| M/D/YYYY | `3/15/2024` | `2024-03-15` |
| M/D/YY | `3/15/24` | `2024-03-15` |

## Querying the Graph

```python
# Keyword-based temporal search
results = tg.query_temporal("when was the release?", top_k=3)
for ev in results:
    print(f"  {ev.date}: {ev.text} (source={ev.source})")
```

### Query Types

The temporal query engine recognises several patterns:

| Pattern | Example | Behaviour |
|---------|---------|-----------|
| After date | "what happened after 2026-03-15" | Filters events >= date |
| Before date | "what happened before 2026-03-15" | Filters events <= date |
| Date range | "events between 2026-01-01 and 2026-03-31" | Filters events in range |
| Most recent | "latest decision" | Returns newest event |
| First/earliest | "first experiment" | Returns oldest event |
| Keyword | "when was the release" | BM25-scored by text overlap |

## Date Arithmetic

`temporal_code_execute()` computes answers from event dates:

```python
from temporal_graph import temporal_code_execute, TemporalEvent

events = [
    TemporalEvent(date="2026-01-10", text="Project started", source="a.md"),
    TemporalEvent(date="2026-03-20", text="v1.0 released", source="b.md"),
]

# How long between events?
answer = temporal_code_execute("how many days between events", events)
# Returns: "69 days (from 2026-01-10 to 2026-03-20)"

# Before/after comparison
answer = temporal_code_execute("did start happen before or after release", events)
# Returns: "Project started happened before v1.0 released (69 days earlier)"

# Most recent
answer = temporal_code_execute("what was the most recent event", events)
# Returns: "v1.0 released (2026-03-20)"

# How long ago
answer = temporal_code_execute("how long since the release", events)
# Returns: "N days since v1.0 released (2026-03-20)"
```

## Events on a Specific Date

```python
tg.add_events([
    TemporalEvent(date="2026-03-15", text="Decision A", source="a.md"),
    TemporalEvent(date="2026-03-15", text="Decision B", source="b.md"),
    TemporalEvent(date="2026-03-20", text="Release", source="c.md"),
])

# Get all events on a date
events = tg.events_on_date("2026-03-15")
# Returns 2 events

# Get events in a range
events = tg.events_between("2026-03-12", "2026-03-18")
# Returns events on 2026-03-15
```

## Persistence

```python
from pathlib import Path

tg.save(Path("temporal.jsonl"))

tg2 = TemporalGraph()
tg2.load(Path("temporal.jsonl"))
```

## Integration with Search

When you use `MemoryIndex.search()`, temporal augmentation runs automatically for temporal-type queries. The search pipeline:

1. Classifies the query as "temporal" if it contains date expressions or temporal keywords
2. Finds matching paragraphs via BM25
3. Extracts dates from matched paragraphs
4. Builds a temporary temporal graph
5. Runs `temporal_code_execute()` for date arithmetic
6. Injects the computed answer into the top search result

This is why a query like "how many days between the start and the release" can return a numeric answer even though no document explicitly states the number of days.

## Current Limitations

Temporal-reasoning is the weakest category in the LongMemEval benchmark at 45.9% (61/133 questions). The main gaps:

- Questions about event ordering across sessions ("did X happen before Y started Z")
- Questions requiring knowledge-update tracking ("what is their CURRENT job" when the job changed across sessions)
- Counterfactual temporal questions ("would they have been available on March 10")

The `fact_decomposer.py` and `arcane_retriever.py` modules are designed to address these gaps via atomic fact validity windows and multi-channel temporal retrieval.

## Next Steps

- [API Reference: temporal_graph](../api/temporal_graph.md)
- [API Reference: fact_decomposer](../api/fact_decomposer.md)
- [Benchmarks: LongMemEval](../benchmarks/LongMemEval.md) — temporal category analysis
