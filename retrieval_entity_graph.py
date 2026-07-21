# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Entity-graph retrieval signal

"""Load the durable entity graph and score query-to-trace connections."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, TypedDict, cast


class EntityRecord(TypedDict, total=False):
    """Entity JSONL record; only ``id`` is required by retrieval."""

    id: str
    name: str
    type: str


class RelationRecord(TypedDict, total=False):
    """Weighted directed relation used as an undirected retrieval signal."""

    source: str
    target: str
    weight: float


EntityGraph = tuple[dict[str, EntityRecord], list[RelationRecord]]


def _jsonl_objects(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    objects: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{path}: JSONL row must be an object")
        objects.append(cast(dict[str, object], payload))
    return objects


def load_entity_graph(graph_dir: Path) -> EntityGraph:
    """Load entity and relation JSONL records from a graph directory."""
    entities: dict[str, EntityRecord] = {}
    for raw_entity in _jsonl_objects(graph_dir / "entities.jsonl"):
        entity_id = raw_entity.get("id")
        if not isinstance(entity_id, str) or not entity_id:
            raise ValueError("entity record requires a non-empty string id")
        entities[entity_id] = cast(EntityRecord, raw_entity)

    relations: list[RelationRecord] = []
    for raw_relation in _jsonl_objects(graph_dir / "relations.jsonl"):
        source = raw_relation.get("source", "")
        target = raw_relation.get("target", "")
        weight = raw_relation.get("weight", 1.0)
        if not isinstance(source, str) or not isinstance(target, str):
            raise ValueError("relation source and target must be strings")
        if not isinstance(weight, (int, float)):
            raise ValueError("relation weight must be numeric")
        relations.append({"source": source, "target": target, "weight": float(weight)})
    return entities, relations


def _mentioned_entities(text: str, entities: dict[str, EntityRecord]) -> list[str]:
    normalized = text.lower().replace("-", " ").replace("_", " ")
    return [
        entity_id
        for entity_id in entities
        if entity_id.lower().replace("-", " ").replace("_", " ") in normalized
    ]


def entity_graph_score_python(
    query: str,
    trace_name: str,
    entities: dict[str, EntityRecord],
    relations: list[RelationRecord],
) -> float:
    """Return the portable weighted connection score for a query and trace."""
    if not entities or not relations:
        return 0.0
    query_entities = _mentioned_entities(query, entities)
    if not query_entities:
        return 0.0
    trace_entities = _mentioned_entities(trace_name, entities)
    if not trace_entities:
        return 0.0

    score = 0.0
    for relation in relations:
        source = relation.get("source", "")
        target = relation.get("target", "")
        weight = float(relation.get("weight", 1.0))
        if (source in query_entities and target in trace_entities) or (
            target in query_entities and source in trace_entities
        ):
            score += weight
        elif (source in query_entities and source in trace_entities) or (
            target in query_entities and target in trace_entities
        ):
            score += weight * 0.5

    max_weight = max(float(relation.get("weight", 1.0)) for relation in relations)
    denominator = max(max_weight * len(query_entities), 1.0)
    return min(score / denominator, 1.0)


def _call_native_score(  # pragma: no cover - optional wheel verified separately
    native_score: Callable[[list[str], list[str], list[tuple[str, str, float]]], float],
    query_entities: list[str],
    trace_entities: list[str],
    relations: list[RelationRecord],
) -> float:
    native_relations = [
        (
            relation.get("source", ""),
            relation.get("target", ""),
            float(relation.get("weight", 1.0)),
        )
        for relation in relations
    ]
    return float(native_score(query_entities, trace_entities, native_relations))


def entity_graph_score(
    query: str,
    trace_name: str,
    entities: dict[str, EntityRecord],
    relations: list[RelationRecord],
) -> float:
    """Dispatch to the native scorer when installed, otherwise use Python."""
    if not entities or not relations:
        return 0.0
    query_entities = _mentioned_entities(query, entities)
    if not query_entities:
        return 0.0
    trace_entities = _mentioned_entities(trace_name, entities)
    if not trace_entities:
        return 0.0
    try:  # pragma: no cover - optional native wheel path verified separately
        from remanentia_retrieve import (  # type: ignore[import-not-found]
            entity_graph_score as native_score,
        )
    except ImportError:
        return entity_graph_score_python(query, trace_name, entities, relations)
    return _call_native_score(  # pragma: no cover - optional wheel verified separately
        native_score,
        query_entities,
        trace_entities,
        relations,
    )
