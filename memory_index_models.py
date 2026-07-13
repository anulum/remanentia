# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Unified index records and priority results

"""Data records and compiled-fact priority handling for the memory index."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from compiled_memory import load_compiled_facts, search_compiled_facts

COMPILED_FACT_MIN_SCORE = 8.0


@dataclass
class Document:
    """One indexed source document and its materialized paragraphs."""

    name: str
    source: str
    path: str
    paragraphs: list[str] = field(default_factory=list)
    tokens: set[str] = field(default_factory=set)
    embedding: NDArray[np.generic] | None = None
    date: str = ""
    doc_type: str = ""


@dataclass
class Paragraph:
    """Paragraph text enriched with indexing metadata."""

    text: str
    para_type: str = ""
    prospective_queries: list[str] = field(default_factory=list)


@dataclass
class SearchResult:
    """Ranked retrieval result returned by the unified index."""

    name: str
    source: str
    score: float
    snippet: str
    paragraph_idx: int = 0
    answer: str = ""
    confidence: float = 0.0


class OperationalIndex(Protocol):
    """Index fields needed to determine compiled-memory readiness."""

    paragraph_index: list[tuple[int, int]]
    documents: list[Document]


def compiled_fact_results(
    query: str, top_k: int, facts_path: Path | None = None
) -> list[SearchResult]:
    """Load, score, filter, and format durable compiled facts."""
    matches = search_compiled_facts(query, load_compiled_facts(facts_path), top_k=top_k)
    return [
        SearchResult(
            name=f"{fact.fact_id}.fact",
            source="compiled",
            score=round(1000.0 + score, 4),
            snippet=fact.fact[:300],
            answer=fact.fact,
            confidence=1.0,
        )
        for fact, score in matches
        if score >= COMPILED_FACT_MIN_SCORE
    ]


def merge_priority_results(
    priority_results: list[SearchResult],
    ranked_results: list[SearchResult],
    top_k: int,
) -> list[SearchResult]:
    """Deduplicate priority and ranked results while preserving precedence."""
    merged: list[SearchResult] = []
    seen: set[tuple[str, str, str]] = set()
    for result in priority_results + ranked_results:
        key = (result.source, result.name, result.answer or result.snippet)
        if key in seen:
            continue
        seen.add(key)
        merged.append(result)
        if len(merged) >= top_k:
            break
    return merged


def has_operational_compiled_memory(index: OperationalIndex) -> bool:
    """Return whether the index has enough materialized or compiled memory."""
    return len(index.paragraph_index) > 1000 or any(
        document.source == "compiled" for document in index.documents
    )
