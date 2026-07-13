# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Entity-aware memory retrieval scoring

"""Match query entities and compute entity-aware retrieval boosts."""

from __future__ import annotations

import re
from typing import Any, cast

Entity = dict[str, Any]
EntityGraph = dict[str, Any]

_PERSON_CENTRIC_RE = re.compile(
    r"\b(relationship|hobby|hobbies|interest|interests|career|job|status|"
    r"personality|feel|feeling|prefer|favorite|partake|destress|self-care|"
    r"political|leaning|member|community)\b",
    re.IGNORECASE,
)

_POSSESSIVE_RE = re.compile(
    r"\b(his|her|their|'s)\s+(hobby|hobbies|interest|interests|career|"
    r"relationship|status|personality|feeling|preference|activity|activities)\b",
    re.IGNORECASE,
)


def build_relation_neighbors(
    entities: dict[str, Entity], relations: list[Entity]
) -> dict[str, list[tuple[str, str, str]]]:
    """Return bidirectional relation neighbors keyed by entity identifier."""
    neighbors: dict[str, list[tuple[str, str, str]]] = {}
    for relation in relations:
        relation_type = str(relation.get("type", "co_occurs"))
        source = str(relation.get("source", ""))
        target = str(relation.get("target", ""))
        if not source or not target:
            continue
        source_label = str(entities.get(source, {}).get("label", source)).lower()
        target_label = str(entities.get(target, {}).get("label", target)).lower()
        neighbors.setdefault(source, []).append((target, target_label, relation_type))
        neighbors.setdefault(target, []).append((source, source_label, relation_type))
    return neighbors


def query_entity_ids(query: str, graph: EntityGraph) -> set[str]:
    """Return identifiers of graph entities mentioned in a query."""
    query_lower = query.lower()
    query_tokens = set(re.findall(r"[a-z0-9][a-z0-9_-]{2,}", query_lower))
    matched: set[str] = set()
    entities = cast(dict[str, Entity], graph["entities"])
    for entity_id, entity in entities.items():
        label = str(entity.get("label", entity_id)).lower()
        if label in query_lower or label in query_tokens:
            matched.add(entity_id)
    return matched


def entity_boost_score(
    paragraph_text: str,
    query_entities: set[str],
    graph: EntityGraph,
) -> float:
    """Return a retrieval boost for query-related entities and typed edges."""
    if not query_entities:
        return 0.0
    paragraph_lower = paragraph_text.lower()
    boost = 0.0
    entities = cast(dict[str, Entity], graph["entities"])
    for entity_id in query_entities:
        label = str(entities.get(entity_id, {}).get("label", entity_id)).lower()
        if label in paragraph_lower:
            boost += 0.1

    relation_neighbors = cast(
        dict[str, list[tuple[str, str, str]]] | None,
        graph.get("relation_neighbors"),
    )
    if relation_neighbors is None:
        relation_neighbors = build_relation_neighbors(
            cast(dict[str, Entity], graph.get("entities", {})),
            cast(list[Entity], graph.get("relations", [])),
        )
        graph["relation_neighbors"] = relation_neighbors
    for entity_id in query_entities:
        for _, neighbor_label, relation_type in relation_neighbors.get(entity_id, []):
            if (
                relation_type != "co_occurs"
                and neighbor_label
                and neighbor_label in paragraph_lower
            ):
                boost += 0.15
    return boost


def extract_query_names(query: str) -> set[str]:
    """Extract likely person names from title-cased query words."""
    excluded = {
        "what",
        "when",
        "where",
        "who",
        "how",
        "why",
        "would",
        "could",
        "does",
        "did",
        "has",
        "have",
        "the",
        "which",
        "likely",
        "yes",
        "not",
    }
    return {
        match.group(1).lower()
        for match in re.finditer(r"\b([A-Z][a-z]{2,})\b", query)
        if match.group(1).lower() not in excluded
    }


def is_person_centric(query: str) -> bool:
    """Return whether a query asks about a person's traits or likely actions."""
    if _PERSON_CENTRIC_RE.search(query) or _POSSESSIVE_RE.search(query):
        return True
    query_lower = query.lower()
    return any(word in query_lower for word in ("would ", "could ", "likely "))
