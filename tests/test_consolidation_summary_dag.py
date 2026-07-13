# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Native-independent summary DAG tests

"""Exercise the portable DAG builder and searcher on concrete trace metadata."""

from __future__ import annotations

from REMANENTIA.consolidation_summary_dag import (  # type: ignore[import]
    build_summary_dag_python,
    search_summary_dag,
)


def test_portable_builder_handles_empty_and_equal_date_traces() -> None:
    """Empty discovery and equal-date parent bounds remain deterministic."""
    assert build_summary_dag_python({}) == []
    traces = {
        "alpha.md": {
            "date": "2026-07-13",
            "project": "remanentia",
            "entities": ["bm25"],
            "key_lines": ["BM25 retrieval finding"],
            "text": "alpha",
        },
        "beta.md": {
            "date": "2026-07-13",
            "project": "remanentia",
            "entities": ["memory"],
            "key_lines": [],
            "text": "Persistent memory detail",
        },
    }

    dag = build_summary_dag_python(traces)
    parent = next(node for node in dag if node["level"] == 1)

    assert parent["date_range"] == ["2026-07-13", "2026-07-13"]
    assert parent["project"] == "remanentia"


def test_search_expands_matching_and_ignores_nonmatching_real_children() -> None:
    """Top-down search expands only child summaries relevant to the query."""
    dag = [
        {
            "node_id": "root",
            "level": 1,
            "summary": "retrieval overview",
            "children": ["hit", "miss"],
            "date_range": ["2026-07-13", "2026-07-13"],
            "entities": ["bm25"],
            "project": "remanentia",
        },
        {
            "node_id": "hit",
            "level": 0,
            "summary": "BM25 retrieval accuracy",
            "children": ["hit.md"],
            "date_range": ["2026-07-13", "2026-07-13"],
            "entities": ["bm25"],
            "project": "remanentia",
        },
        {
            "node_id": "miss",
            "level": 0,
            "summary": "unrelated cooking note",
            "children": ["miss.md"],
            "date_range": ["2026-07-13", "2026-07-13"],
            "entities": ["bread"],
            "project": "general",
        },
    ]

    results = search_summary_dag(dag, "BM25 retrieval")

    assert [node["node_id"] for node in results] == ["hit"]
