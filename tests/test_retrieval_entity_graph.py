# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Real-surface entity-graph retrieval tests

"""Exercise production graph loading and scoring through real JSONL files."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from retrieval_entity_graph import (
    EntityRecord,
    RelationRecord,
    entity_graph_score,
    entity_graph_score_python,
    load_entity_graph,
)


def _write_jsonl(path: Path, rows: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n\n",
        encoding="utf-8",
    )


def _graph() -> tuple[dict[str, EntityRecord], list[RelationRecord]]:
    return (
        {
            "scpn-control": {"id": "scpn-control", "type": "system"},
            "tokamak": {"id": "tokamak", "type": "domain"},
            "disruption": {"id": "disruption", "type": "event"},
            "memory": {"id": "memory", "type": "concept"},
        },
        [
            {"source": "scpn-control", "target": "tokamak", "weight": 2.0},
            {"source": "disruption", "target": "scpn-control", "weight": 1.5},
            {"source": "memory", "target": "archive", "weight": 1.0},
            {"source": "unknown", "target": "absent", "weight": 0.5},
        ],
    )


def test_graph_loader_reads_real_entity_and_relation_jsonl(tmp_path: Path) -> None:
    """Blank lines are ignored and numeric weights are normalized to floats."""
    graph_dir = tmp_path / "graph"
    _write_jsonl(
        graph_dir / "entities.jsonl",
        [
            {"id": "scpn-control", "name": "SCPN Control", "type": "system"},
            {"id": "tokamak", "name": "Tokamak", "type": "domain"},
        ],
    )
    _write_jsonl(
        graph_dir / "relations.jsonl",
        [
            {"source": "scpn-control", "target": "tokamak", "weight": 2},
            {"source": "tokamak", "target": "plasma"},
        ],
    )

    entities, relations = load_entity_graph(graph_dir)

    assert set(entities) == {"scpn-control", "tokamak"}
    assert relations == [
        {"source": "scpn-control", "target": "tokamak", "weight": 2.0},
        {"source": "tokamak", "target": "plasma", "weight": 1.0},
    ]
    assert load_entity_graph(tmp_path / "absent") == ({}, [])


@pytest.mark.parametrize(
    ("filename", "row", "message"),
    [
        ("entities.jsonl", ["not", "object"], "JSONL row must be an object"),
        ("entities.jsonl", {"name": "missing id"}, "requires a non-empty string id"),
        (
            "relations.jsonl",
            {"source": 7, "target": "tokamak"},
            "source and target must be strings",
        ),
        (
            "relations.jsonl",
            {"source": "tokamak", "target": 7},
            "source and target must be strings",
        ),
        (
            "relations.jsonl",
            {"source": "tokamak", "target": "plasma", "weight": "heavy"},
            "weight must be numeric",
        ),
    ],
)
def test_graph_loader_rejects_invalid_durable_rows(
    tmp_path: Path,
    filename: str,
    row: object,
    message: str,
) -> None:
    """Malformed durable records fail at the filesystem boundary with precise errors."""
    graph_dir = tmp_path / "graph"
    _write_jsonl(graph_dir / filename, [row])

    with pytest.raises(ValueError, match=message):
        load_entity_graph(graph_dir)


def test_portable_graph_score_covers_direct_reverse_shared_and_unrelated_edges() -> None:
    """The portable scorer uses weighted graph topology rather than hand-built results."""
    entities, relations = _graph()

    direct = entity_graph_score_python(
        "SCPN control mitigation", "tokamak-results.md", entities, relations
    )
    reverse = entity_graph_score_python(
        "tokamak mitigation", "scpn_control-results.md", entities, relations
    )
    shared = entity_graph_score_python("persistent memory", "memory-notes.md", entities, relations)
    unrelated = entity_graph_score_python("SCPN control", "memory-notes.md", entities, relations)

    assert direct == pytest.approx(1.0)
    assert reverse == pytest.approx(1.0)
    assert shared == pytest.approx(0.25)
    assert unrelated == 0.0


def test_graph_score_returns_zero_without_required_graph_mentions() -> None:
    """Empty graphs, absent query entities and absent trace entities cannot boost rank."""
    entities, relations = _graph()

    assert entity_graph_score_python("scpn control", "tokamak.md", {}, relations) == 0.0
    assert entity_graph_score_python("scpn control", "tokamak.md", entities, []) == 0.0
    assert entity_graph_score_python("bread recipe", "tokamak.md", entities, relations) == 0.0
    assert entity_graph_score_python("tokamak", "bread.md", entities, relations) == 0.0
    assert entity_graph_score("bread", "notes.md", entities, relations) == 0.0
    assert entity_graph_score("tokamak", "bread.md", entities, relations) == 0.0
    assert entity_graph_score("tokamak", "scpn.md", {}, relations) == 0.0


def test_runtime_dispatch_matches_portable_score_on_real_graph_data(tmp_path: Path) -> None:
    """Installed native dispatch and explicit portable scoring agree on loaded JSONL."""
    graph_dir = tmp_path / "graph"
    _write_jsonl(
        graph_dir / "entities.jsonl",
        [{"id": "scpn-control"}, {"id": "tokamak"}, {"id": "disruption"}],
    )
    _write_jsonl(
        graph_dir / "relations.jsonl",
        [
            {"source": "scpn-control", "target": "tokamak", "weight": 2.0},
            {"source": "disruption", "target": "scpn-control", "weight": 1.0},
        ],
    )
    entities, relations = load_entity_graph(graph_dir)

    portable = entity_graph_score_python(
        "scpn-control disruption", "tokamak-report.md", entities, relations
    )
    dispatched = entity_graph_score(
        "scpn-control disruption", "tokamak-report.md", entities, relations
    )

    assert dispatched == pytest.approx(portable)
    assert 0.0 < dispatched <= 1.0
