# Feed Normalization

Normalises SYNAPSE feed metadata before `feed_ingest` turns explicit findings
and decisions into retrievable memory. It collapses mixed sender identities,
project names, actor names, feed sequences, and timestamp shapes into stable
provenance fields used by the finding admission path.

::: feed_normalization
