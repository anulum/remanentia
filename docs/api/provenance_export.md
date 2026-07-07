# provenance_export

Project the knowledge-store belief graph onto the provenance store the
lineage-of-belief scorer reads. Part of the world-class evaluation harness
(roadmap W5) — the producer side of the auditability axis.

## Why this module exists

`lineage_completeness` and `scorecard_report` score *why a fact is believed* against
a provenance store, but nothing produced one from a real knowledge store, so the
lineage axis reported "not measured". This module is that producer: it turns a
persisted `knowledge_store` into the provenance-node JSONL the scorer consumes,
keyed by the note's **own id** — the single canonical belief id — so a run whose
cited ids are note ids resolves against it and lights the lineage axis.

## Public surface

```python
from provenance_export import (
    provenance_node_from_note, build_provenance_store,
    render_provenance_jsonl, export_knowledge_store, main,
)
```

- `provenance_node_from_note(note)` — project one serialised knowledge note to a
  `ProvenanceNode` (`parent` = first `derived_from` link; `origin` =
  `source_quality == "stated"`).
- `build_provenance_store(notes)` — id → provenance-node store (last write wins).
- `render_provenance_jsonl(store)` — the id/origin/parent JSONL
  `scorecard_report.load_provenance_store` reads, sorted by id.
- `export_knowledge_store(notes_path, output_path)` — load a store and write its
  provenance JSONL; returns the node count.
- `main(argv)` — the `remanentia-provenance-export` CLI
  (`--notes`, `--output`).

## Invariants

- **Faithful, not invented.** Parent and origin come directly from the note's
  `derived_from` links and `source_quality`; an inferred note with no recorded
  derivation projects to a non-origin root and is correctly marked lineage-incomplete.
- **Deterministic and pure** (the projection functions make no model calls); the
  JSONL is sorted by id for a stable, diff-friendly artefact.

## See also

- `lineage_completeness` — the scorer this store feeds.
- `scorecard_report` — `--provenance-store` lights the lineage axis on a real run.
- `knowledge_store` — the belief graph this projects from.
