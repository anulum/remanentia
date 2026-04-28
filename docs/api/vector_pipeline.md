<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
-->

# vector_pipeline

Bridge between the existing `MemoryIndex` paragraph corpus and the
persistent vector index.

## Purpose

`vector_pipeline.py` makes the vector store part of the Remanentia
retrieval system without changing the BM25 index. It converts each
`MemoryIndex` paragraph into a stable `VectorChunk`, writes the vector
store, and searches the persisted vector index.

This keeps the current deterministic retrieval path intact while adding a
large dense-retrieval side channel.

## Pipeline

```text
MemoryIndex.build()
        │
        ▼
chunks_from_memory_index()
        │
        ▼
PersistentVectorIndex.build()
        │
        ▼
search_memory_vector_index()
```

## Operator Commands

The vector pipeline can be operated without importing Python internals:

```bash
python -m vector_pipeline status
python -m vector_pipeline estimate 200000 --dimension 768
python -m vector_pipeline build --batch-size 64
python -m vector_pipeline refresh --batch-size 64
python -m vector_pipeline watch --interval-s 900
python -m vector_pipeline search "hybrid retrieval decision" --top 5
python -m vector_pipeline search "hybrid retrieval decision" --public \
  --public-source paper --public-path-prefix paper --redaction-file local_terms.txt
```

`build`, `refresh`, `watch`, and `search` read endpoint configuration
from environment variables using the `REMANENTIA_EMBEDDING` prefix by
default:

| Variable | Purpose |
|---|---|
| `REMANENTIA_EMBEDDING_BASE_URL` | HTTP or HTTPS embedding endpoint base URL |
| `REMANENTIA_EMBEDDING_MODEL` | Embedding model identifier understood by the endpoint |
| `REMANENTIA_EMBEDDING_API_KEY` | Optional bearer token |
| `REMANENTIA_EMBEDDING_TIMEOUT_S` | Optional timeout in seconds |

Runtime endpoint values belong in shell environment, ignored deployment
configuration, or secret storage. They must not be written into public
tracked files.

By default, `build` excludes ignored coordination folders, internal docs,
runtime state, backups, archives, model directories, and similar private
paths. Use `--include-private` only for a local-only operational index
that will not back a public API or public demo.

`refresh` computes a deterministic corpus fingerprint and rebuilds only
when the current `MemoryIndex` paragraph corpus changed. The manifest
stores `corpus_fingerprint`, `corpus_chunk_count`, and
`refreshed_at_unix` after the first refresh. Use `--force` to rebuild
even when the fingerprint is unchanged.

`watch` is the scheduled worker form of `refresh`. It runs the same
public-safe corpus selection by default, sleeps between cycles, and writes
a heartbeat JSON file to `snn_state/vector_refresh_worker.json`. The
heartbeat records the worker PID, cycle number, status, timestamp, and
last refresh decision so operators can distinguish a live worker from a
stale legacy daemon.

Public-facing result output is a separate opt-in view. `--public` returns
only results that match explicit source and path-prefix allowlists, emits
only selected metadata keys, truncates long text, and applies configured
redaction terms. If no allowlist is provided, public output returns no
raw results.

Redaction files are newline-delimited local policy files. Keep operational
term lists in ignored internal configuration or deployment secrets, not in
public tracked files.

## Metadata Contract

Each vector chunk stores:

- stable chunk ID
- source label
- document name
- document type
- date
- paragraph index
- source path
- raw text
- SHA-256 content hash

The hash makes incremental rebuilds and stale-vector detection possible
without relying on filenames alone.

## API Reference

::: vector_pipeline
