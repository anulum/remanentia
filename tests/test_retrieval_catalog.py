# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Real-filesystem tests for the retrieval catalog

"""Exercise production retrieval catalog behavior on real files and JSONL."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from REMANENTIA.retrieval_catalog import (
    append_history,
    chunk_trace_catalog,
    classify_trace_tier,
    find_related_traces,
    read_history,
    suggest_queries,
    summarize_traces,
    tier_boost,
)


def test_history_roundtrip_bounds_results_and_skips_invalid_rows(tmp_path: Path) -> None:
    """History persists real JSONL, keeps five ranks, and tolerates corrupt lines."""
    history = tmp_path / "state" / "retrieval_history.jsonl"
    assert read_history(history) == []

    results = [{"trace": f"trace-{rank}.md", "score": rank / 10} for rank in range(7)]
    append_history(history, "tokamak disruption", results, timestamp=42.5)
    with history.open("a", encoding="utf-8") as stream:
        stream.write("\nnot-json\n[]\n")
    append_history(history, "persistent memory", results[:1], timestamp=43.5)

    rows = read_history(history)
    assert [row["query"] for row in rows] == ["tokamak disruption", "persistent memory"]
    assert rows[0]["timestamp"] == 42.5
    assert len(rows[0]["results"]) == 5
    assert read_history(history, limit=1)[0]["query"] == "persistent memory"


def test_history_uses_a_real_current_timestamp(tmp_path: Path) -> None:
    """The default append path records a live wall-clock timestamp."""
    history = tmp_path / "history.jsonl"
    append_history(history, "live query", [])

    timestamp = read_history(history)[0]["timestamp"]
    assert isinstance(timestamp, float)
    assert timestamp > 0


def test_trace_summaries_parse_real_markdown_newest_first(tmp_path: Path) -> None:
    """Summary extraction skips headings/frontmatter and respects file modification time."""
    missing = tmp_path / "missing"
    assert summarize_traces(missing) == []

    older = tmp_path / "older.md"
    older.write_text("# Older\n\n**Source:** test\n- **Agent:** codex\n- Stable older fact.\n")
    newer = tmp_path / "newer.md"
    newer.write_text("# Newer\n---\n## Details\nNewest factual summary.\n")
    empty = tmp_path / "empty.md"
    empty.write_text("# Heading only\n")
    os.utime(older, (100, 100))
    os.utime(newer, (300, 300))
    os.utime(empty, (200, 200))

    assert summarize_traces(tmp_path) == [
        {"name": "newer.md", "summary": "Newest factual summary."},
        {"name": "empty.md", "summary": "(no summary)"},
        {"name": "older.md", "summary": "Stable older fact."},
    ]


@pytest.mark.parametrize(
    ("age_hours", "expected"),
    [(1, "hot"), (24, "hot"), (25, "warm"), (168, "warm"), (169, "cold")],
)
def test_trace_tiers_use_real_file_metadata(tmp_path: Path, age_hours: int, expected: str) -> None:
    """Tier boundaries are evaluated from the file's real mtime."""
    trace = tmp_path / "trace.md"
    trace.write_text("# Trace\ncontent\n")
    now = 1_000_000.0
    os.utime(trace, (now - age_hours * 3600, now - age_hours * 3600))

    assert classify_trace_tier(trace, timestamp=now) == expected


def test_tier_boosts_preserve_bounded_recency_weights() -> None:
    """Known persistence tiers and unknown future tiers receive declared weights."""
    assert tier_boost("hot") == 1.02
    assert tier_boost("warm") == 1.0
    assert tier_boost("cold") == 0.98
    assert tier_boost("archive") == 1.0


def test_related_traces_rank_real_markdown_by_shared_terms(tmp_path: Path) -> None:
    """Related lookup reads actual traces and ranks lexical overlap without substitutes."""
    (tmp_path / "plasma-control.md").write_text("tokamak plasma disruption mitigation")
    (tmp_path / "plasma-safety.md").write_text("tokamak plasma disruption safety")
    (tmp_path / "memory.md").write_text("persistent neural memory retrieval")

    related = find_related_traces(tmp_path, "plasma-control.md", top_k=1)

    assert related == [
        {"trace": "plasma-safety.md", "similarity": pytest.approx(0.5), "shared_terms": 3}
    ]
    assert find_related_traces(tmp_path, "../plasma-control.md") == []
    assert find_related_traces(tmp_path, "") == []
    assert find_related_traces(tmp_path, "not-markdown.txt") == []
    assert find_related_traces(tmp_path, "missing.md") == []
    (tmp_path / "directory.md").mkdir()
    assert find_related_traces(tmp_path, "directory.md") == []


def test_related_traces_handle_empty_and_disjoint_real_files(tmp_path: Path) -> None:
    """Empty targets and disjoint neighbors produce no fabricated relationship."""
    (tmp_path / "a.md").write_text("the and")
    (tmp_path / "food.md").write_text("bread flour yeast")
    assert find_related_traces(tmp_path, "a.md") == []

    (tmp_path / "target.md").write_text("plasma tokamak")
    assert find_related_traces(tmp_path, "target.md") == []


def test_query_suggestions_read_newest_distinct_valid_history_rows(tmp_path: Path) -> None:
    """Suggestions parse the real JSONL file and discard malformed or mistyped records."""
    history = tmp_path / "history.jsonl"
    assert suggest_queries(history, "to") == []
    history.write_text(
        "\n".join(
            [
                "",
                json.dumps({"query": "Tokamak control"}),
                "invalid-json",
                json.dumps(["not", "an", "object"]),
                json.dumps({"query": 17}),
                json.dumps({"query": "Tokamak safety"}),
                json.dumps({"query": "Tokamak control"}),
            ]
        )
        + "\n"
    )

    assert suggest_queries(history, "t") == []
    assert suggest_queries(history, "TO", limit=2) == ["Tokamak control", "Tokamak safety"]
    assert suggest_queries(history, "zz") == []


def test_chunk_catalog_groups_real_filenames_by_project_and_date(tmp_path: Path) -> None:
    """Chunking derives project/date groups from actual Markdown filenames."""
    assert chunk_trace_catalog(tmp_path / "missing") == []
    (tmp_path / "2026-07-13T1000_REMANENTIA_alpha.md").write_text("alpha")
    (tmp_path / "2026-07-13T1100_REMANENTIA_beta.md").write_text("beta")
    (tmp_path / "2026-07-12T0900_DIRECTOR_gamma.md").write_text("gamma")
    (tmp_path / "undated.md").write_text("general")

    chunks = chunk_trace_catalog(tmp_path)

    assert [chunk["name"] for chunk in chunks] == [
        "general undated",
        "REMANENTIA 2026-07-13",
        "DIRECTOR 2026-07-12",
    ]
    assert chunks[1]["count"] == 2
    assert chunks[1]["summary"] == "REMANENTIA: 2 traces (2026-07-13)"
    assert chunks[2]["summary"] == "DIRECTOR: 1 trace (2026-07-12)"
