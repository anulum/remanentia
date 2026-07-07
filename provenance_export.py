# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — provenance store export

"""Project the knowledge-store belief graph onto a queryable provenance store.

:mod:`lineage_completeness` scores *why a fact is believed* — every cited id must
resolve to a record whose parent chain reaches an originating write. It needs a
provenance store to score against, but nothing produced one from real memory: the
scorer existed while the store stayed hypothetical, so the lineage axis of the
world-class scorecard always reported "not measured". This module produces it.

A :class:`knowledge_store.KnowledgeNote` already records its derivation — a typed
``derived_from`` link toward the note it was inferred from — and its
``source_quality`` (``stated`` = written directly by an external source;
``inferred`` / ``corrected`` = derived). That is exactly a provenance chain, so the
projection is faithful, not invented:

* ``parent`` = the note's first ``derived_from`` target (the step toward origin),
  or ``None`` when the note derives from nothing recorded.
* ``origin`` = ``source_quality == "stated"`` — an originating write event.

An *inferred* note with no recorded ``derived_from`` therefore projects to a
non-origin root: :func:`lineage_completeness.is_lineage_complete` marks it
incomplete, which is correct — it is a belief with no auditable source, exactly
the undisciplined write the metric exists to expose. The output is the
provenance-node JSONL that :func:`scorecard_report.load_provenance_store` reads,
closing the loop from stored belief to scored lineage. Pure and deterministic.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from lineage_completeness import ProvenanceNode

_DERIVED_FROM = "derived_from"


def _first_derivation(links: object) -> str | None:
    """Return the first ``derived_from`` link target, or ``None``.

    Links follow the knowledge-store shape ``{"target": id, "type": edge}``.
    Malformed entries (not a mapping, missing/blank target) are skipped rather
    than raising, so a partially-written note still exports.
    """
    if not isinstance(links, Sequence):
        return None
    for link in links:
        if not isinstance(link, Mapping):
            continue
        if link.get("type") != _DERIVED_FROM:
            continue
        target = link.get("target")
        if isinstance(target, str) and target:
            return target
    return None


def provenance_node_from_note(note: Mapping[str, object]) -> ProvenanceNode:
    """Project one serialised knowledge-store note onto a provenance node.

    Uses the note's ``derived_from`` link as the parent toward origin and marks a
    ``stated`` note as an originating write. Raises ``ValueError`` when the note
    has no usable string ``id`` — a provenance record without an identity cannot
    be resolved.
    """
    node_id = note.get("id")
    if not isinstance(node_id, str) or not node_id:
        raise ValueError(f"knowledge note needs a non-empty string id: {note!r}")
    origin = note.get("source_quality") == "stated"
    parent = _first_derivation(note.get("links"))
    return ProvenanceNode(id=node_id, origin=origin, parent=parent)


def build_provenance_store(
    notes: Sequence[Mapping[str, object]],
) -> dict[str, ProvenanceNode]:
    """Project a set of serialised notes into an id → provenance-node store.

    Later notes with a repeated id overwrite earlier ones (last write wins), so a
    re-exported store reflects the current state of each note.
    """
    store: dict[str, ProvenanceNode] = {}
    for note in notes:
        node = provenance_node_from_note(note)
        store[node.id] = node
    return store


def render_provenance_jsonl(store: Mapping[str, ProvenanceNode]) -> str:
    """Serialise a provenance store to the node JSONL the scorecard reads.

    One compact JSON object per line — ``id``, ``origin``, ``parent`` — ordered by
    id for a stable, diff-friendly artefact. ``load_provenance_store`` reads it
    back verbatim.
    """
    lines = []
    for node_id in sorted(store):
        node = store[node_id]
        lines.append(json.dumps({"id": node.id, "origin": node.origin, "parent": node.parent}))
    return "\n".join(lines)


def export_knowledge_store(notes_path: Path | None, output_path: Path) -> int:
    """Project a persisted knowledge store to a provenance-node JSONL file.

    Loads the knowledge store (from ``notes_path`` or its default location),
    projects every note to a provenance node keyed by the note's own id — the
    same id the store already assigns — and writes the JSONL that
    :func:`scorecard_report.load_provenance_store` reads. Returns the number of
    nodes written. This closes the lineage loop for a wheel consumer: the note
    id is the one canonical belief id, so a run whose cited ids are note ids
    resolves against this store and lights the lineage axis.
    """
    from knowledge_store import KnowledgeStore

    store = KnowledgeStore()
    store.load(notes_path=notes_path)
    notes = [note.to_dict() for note in store.notes.values()]
    provenance = build_provenance_store(notes)
    output_path.write_text(render_provenance_jsonl(provenance) + "\n", encoding="utf-8")
    return len(provenance)


def main(argv: list[str] | None = None) -> int:
    """Export a knowledge store's belief graph as a provenance-node JSONL."""
    parser = argparse.ArgumentParser(
        description="Export the knowledge-store belief graph to a provenance JSONL"
    )
    parser.add_argument(
        "--notes",
        type=Path,
        default=None,
        help="knowledge_notes.jsonl to read (defaults to the store's standard path)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="destination provenance-node JSONL",
    )
    args = parser.parse_args(argv)
    count = export_knowledge_store(args.notes, args.output)
    print(f"Wrote {count} provenance nodes: {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
