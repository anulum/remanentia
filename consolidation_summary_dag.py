# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Hierarchical consolidation summary DAG

"""Build and search multi-level summaries over episodic traces."""

from __future__ import annotations

import re
from dataclasses import dataclass
from importlib import import_module
from typing import Any, TypeAlias, cast

from consolidation_trace_analysis import TraceData

DAGNodeDict: TypeAlias = dict[str, Any]

DAG_FANOUT = 4  # number of children per internal node


@dataclass
class DAGNode:
    """A node in the hierarchical summary DAG."""

    node_id: str
    level: int  # 0 = leaf (raw trace), 1+ = summary
    summary: str
    children: list[str]  # child node_ids or trace filenames
    date_range: tuple[str, str]  # (earliest, latest) ISO dates
    entities: list[str]
    project: str


def build_summary_dag(trace_data: TraceData) -> list[DAGNodeDict]:
    """Build a summary DAG with the native engine when it is installed."""
    if not trace_data:
        return []
    try:
        native_build = import_module("remanentia_consolidation").build_summary_dag
    except ImportError:  # pragma: no cover - platform-dependent dispatch
        return build_summary_dag_python(trace_data)
    tuples = [  # pragma: no cover - native dispatch
        (
            name,
            data.get("date", ""),
            data.get("project", "general"),
            data.get("entities", []),
            data.get("key_lines", []),
            data.get("text", ""),
        )
        for name, data in sorted(trace_data.items(), key=lambda item: item[1].get("date", ""))
    ]
    return cast(list[DAGNodeDict], native_build(tuples, DAG_FANOUT))  # pragma: no cover


def build_summary_dag_python(trace_data: TraceData) -> list[DAGNodeDict]:
    """Build a hierarchical summary DAG from trace data.

    Args:
        trace_data: dict mapping trace filename to metadata dict with
            keys: date, project, entities, key_lines, text

    Returns:
        List of DAGNode dicts (serialisable).
    """
    if not trace_data:
        return []

    # Level 0: leaf nodes from individual traces
    leaves: list[DAGNode] = []
    for name, data in sorted(trace_data.items(), key=lambda x: x[1].get("date", "")):
        summary_lines = data.get("key_lines", [])[:5]
        summary = " ".join(summary_lines) if summary_lines else data.get("text", "")[:200]
        date_str = data.get("date", "")[:10]
        leaves.append(
            DAGNode(
                node_id=f"L0_{name}",
                level=0,
                summary=summary,
                children=[name],
                date_range=(date_str, date_str),
                entities=data.get("entities", [])[:20],
                project=data.get("project", "general"),
            )
        )

    all_nodes = list(leaves)
    current_level_nodes = leaves
    level = 1

    # Build higher levels by grouping DAG_FANOUT nodes together
    while len(current_level_nodes) > 1:
        next_level: list[DAGNode] = []
        for i in range(0, len(current_level_nodes), DAG_FANOUT):
            group = current_level_nodes[i : i + DAG_FANOUT]
            if not group:  # pragma: no cover — range() never yields empty slice
                break

            # Merge summaries: take first sentence from each child
            merged_summary_parts = []
            all_entities: set[str] = set()
            earliest = "9999"
            latest = "0000"
            children_ids = []
            projects: list[str] = []

            for node in group:
                # Take first 100 chars of each child summary
                merged_summary_parts.append(node.summary[:100])
                all_entities.update(node.entities)
                if node.date_range[0] and node.date_range[0] < earliest:
                    earliest = node.date_range[0]
                if node.date_range[1] and node.date_range[1] > latest:
                    latest = node.date_range[1]
                children_ids.append(node.node_id)
                projects.append(node.project)

            # Most common project in group
            project = max(set(projects), key=projects.count) if projects else "general"

            merged_summary = " | ".join(merged_summary_parts)
            node_id = f"L{level}_{i // DAG_FANOUT}_{earliest}"

            parent = DAGNode(
                node_id=node_id,
                level=level,
                summary=merged_summary,
                children=children_ids,
                date_range=(
                    earliest if earliest != "9999" else "",
                    latest if latest != "0000" else "",
                ),
                entities=sorted(all_entities)[:30],
                project=project,
            )
            next_level.append(parent)

        all_nodes.extend(next_level)
        current_level_nodes = next_level
        level += 1

    return [_dag_node_to_dict(n) for n in all_nodes]


def _dag_node_to_dict(node: DAGNode) -> DAGNodeDict:
    """Convert a summary DAG node to its JSON-serialisable representation."""

    return {
        "node_id": node.node_id,
        "level": node.level,
        "summary": node.summary,
        "children": node.children,
        "date_range": list(node.date_range),
        "entities": node.entities,
        "project": node.project,
    }


def search_summary_dag(
    dag_nodes: list[DAGNodeDict],
    query: str,
    top_k: int = 10,
) -> list[DAGNodeDict]:
    """Search the summary DAG top-down for relevant nodes.

    Starts at the highest level, finds matching nodes, then
    drills into their children for more detail.
    """
    if not dag_nodes:
        return []

    query_tokens = set(re.findall(r"\w{3,}", query.lower()))
    if not query_tokens:
        return []

    # Group nodes by level
    by_level: dict[int, list[DAGNodeDict]] = {}
    node_map: dict[str, DAGNodeDict] = {}
    for n in dag_nodes:
        level = n["level"]
        by_level.setdefault(level, []).append(n)
        node_map[n["node_id"]] = n

    max_level = max(by_level.keys()) if by_level else 0

    # Score at highest level first
    def _score(node: DAGNodeDict) -> float:
        """Score a DAG node by lexical overlap with the query tokens."""

        text = (node["summary"] + " " + " ".join(node["entities"])).lower()
        text_tokens = set(re.findall(r"\w{3,}", text))
        overlap = len(query_tokens & text_tokens)
        return overlap

    # Top-down search: start from root, expand best matches
    candidates = by_level.get(max_level, [])
    scored = [(n, _score(n)) for n in candidates]
    scored.sort(key=lambda x: -x[1])

    results: list[DAGNodeDict] = []
    seen: set[str] = set()

    # Expand top matches down to leaves
    frontier = [n for n, s in scored[:top_k] if s > 0]
    while frontier:
        node = frontier.pop(0)
        if node["node_id"] in seen:  # pragma: no cover — dedup guard for overlapping DAG paths
            continue
        seen.add(node["node_id"])

        if node["level"] == 0:
            results.append(node)
        else:
            # Expand children, prioritise by score
            children = [node_map[cid] for cid in node["children"] if cid in node_map]
            children_scored = [(c, _score(c)) for c in children]
            children_scored.sort(key=lambda x: -x[1])
            for c, s in children_scored:
                if s > 0 and c["node_id"] not in seen:
                    frontier.append(c)

        if len(results) >= top_k:
            break

    return results[:top_k]
