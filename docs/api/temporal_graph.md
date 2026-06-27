# temporal_graph

Temporal event graph with relative date resolution and TReMu code execution.
The Python implementation is the reference path; when the optional
`remanentia_temporal` Rust extension is installed, date parsing, phase
resonance, temporal-edge building, and query scoring use the native helpers
behind the same public API. Graph persistence is JSONL via
`TemporalGraph.save()` / `TemporalGraph.load()`, and LongMemEval uses the same
surface for TReMu precompute before reader synthesis.

::: temporal_graph.TemporalGraph
    options:
      show_source: true
      members_order: source

::: temporal_graph.TemporalEvent

::: temporal_graph.TemporalEdge

::: temporal_graph.temporal_code_execute

::: temporal_graph.parse_dates
