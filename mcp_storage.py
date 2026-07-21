# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — MCP filesystem recall and graph queries

"""Real filesystem-backed storage operations used by the MCP adapter."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeAlias

Tokenizer: TypeAlias = Callable[[str], set[str]]
RecallIndex: TypeAlias = dict[str, tuple[set[str], str]]


def build_recall_index(base: Path, tokenizer: Tokenizer) -> RecallIndex:
    """Read trace and semantic Markdown into a lightweight token index."""
    index: RecallIndex = {}
    traces_dir = base / "reasoning_traces"
    semantic_dir = base / "memory" / "semantic"

    if traces_dir.exists():
        for path in traces_dir.glob("*.md"):
            text = path.read_text(encoding="utf-8")
            index[path.name] = (tokenizer(text), text[:500])

    if semantic_dir.exists():
        for path in semantic_dir.rglob("*.md"):
            text = path.read_text(encoding="utf-8")
            relative = path.relative_to(semantic_dir)
            index[f"[semantic] {relative}"] = (tokenizer(text), text[:500])
    return index


def recall_from_index(
    query: str,
    *,
    top_k: int,
    index: RecallIndex,
    tokenizer: Tokenizer,
) -> str:
    """Score a query against a previously built filesystem recall index."""
    query_tokens = tokenizer(query)
    if not query_tokens:
        return "Empty query."

    scored: list[tuple[str, float, str]] = []
    for name, (tokens, snippet) in index.items():
        overlap = len(query_tokens & tokens) / len(query_tokens)
        if overlap > 0:
            scored.append((name, overlap, snippet))
    scored.sort(key=lambda item: -item[1])

    if not scored:
        return f"No memories found for: {query}"
    return "\n\n".join(
        f"[{name} (score={score:.2f})]\n{snippet}" for name, score, snippet in scored[:top_k]
    )


def query_graph(graph_dir: Path, *, entity: str = "", top: int = 10) -> str:
    """Read and query the persisted entity relation graph."""
    relations_path = graph_dir / "relations.jsonl"
    if not relations_path.exists():
        return "No relations. Run consolidation first."

    relations: list[dict[str, Any]] = [
        json.loads(line)
        for line in relations_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if entity:
        matches = [
            relation
            for relation in relations
            if relation["source"] == entity or relation["target"] == entity
        ]
        matches.sort(key=lambda relation: -float(relation.get("weight", 0)))
        lines = [f"Connections for '{entity}':"]
        for relation in matches[:top]:
            other = relation["target"] if relation["source"] == entity else relation["source"]
            lines.append(
                f"  {other} (weight={relation['weight']}, "
                f"{len(relation.get('evidence', []))} traces)"
            )
        return "\n".join(lines)

    top_relations = sorted(
        relations,
        key=lambda relation: -float(relation.get("weight", 0)),
    )[:top]
    lines = [f"Top {len(top_relations)} entity relationships:"]
    lines.extend(
        f"  {relation['source']} <-> {relation['target']} weight={relation['weight']}"
        for relation in top_relations
    )
    return "\n".join(lines)
