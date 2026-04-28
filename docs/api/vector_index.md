<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
-->

# vector_index

Persistent dense-vector storage for large Remanentia retrieval indexes.

## Purpose

`vector_index.py` stores one normalised dense vector per retrievable text
chunk and keeps the chunk text, source, hash, and metadata in SQLite. It
is designed for long-running private memory systems where the corpus grows
steadily and index rebuilds must remain inspectable.

## Storage Model

The index directory contains:

| File | Role |
|---|---|
| `vectors.npz` | Compressed float32 vector matrix |
| `chunks.sqlite` | Chunk IDs, source labels, text, content hashes, metadata |
| `manifest.json` | Count, dimension, byte totals, build timestamp |

The raw vector budget is:

```text
count × dimension × bytes_per_value
```

For 768-dimensional float32 vectors:

| Count | Raw vector data | Practical local budget |
|---:|---:|---:|
| 2,048 | ~6 MiB | ~6 MiB measured on the local probe |
| 200,000 | ~586 MiB | ~1-3 GiB with metadata and search overhead |
| 1,000,000 | ~2.86 GiB | ~5-15 GiB with metadata and search overhead |

Growth is linear: each new chunk adds one vector plus one metadata row.

## Retrieval Contract

`PersistentVectorIndex` performs exact cosine search over normalised
vectors. Exact search is the first production path because it is
deterministic and easy to audit. Approximate search can be added later
behind the same public result contract when the corpus requires it.

```python
from pathlib import Path

from vector_index import HttpEmbeddingClient, PersistentVectorIndex, VectorChunk

provider = HttpEmbeddingClient.from_env()
index = PersistentVectorIndex(Path("snn_state/vector_index"))

chunks = [
    VectorChunk(
        chunk_id="trace:001",
        text="Decision: use hybrid retrieval for memory recall.",
        source="reasoning_traces",
        metadata={"project": "remanentia"},
    )
]

index.build(chunks, provider, batch_size=64)
results = index.search("hybrid retrieval decision", provider, top_k=5)
```

## Industrialisation Rules

- Chunk IDs must be stable across rebuilds.
- Content hashes decide whether a chunk changed.
- Source, project, session, date, and document type belong in metadata.
- Raw text remains in SQLite for auditability.
- Public configuration uses generic endpoint/model variables only.
- Private endpoint mappings stay in ignored coordination files.

## API Reference

::: vector_index
