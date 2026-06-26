# Feed Ingest

Consumes explicit `Finding:` and `Decision:` rows from SYNAPSE `feed.ndjson`
and runs them through the same parser, admission gate, and Markdown finding
sink as hub-backed findings. Feed identity metadata is normalised through
`feed_normalization` before persistence, so admitted findings carry controlled
project, actor, sequence, and timestamp provenance.

::: feed_ingest
