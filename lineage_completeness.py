# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — lineage-of-belief completeness

"""Measure why a fact is believed: lineage-of-belief completeness.

Final-answer accuracy cannot say *which evidence supported each claim* or *how a
belief was formed* — a recognised 2026 evaluation gap. The governance literature
formalises it as Provenance Visibility = Queryable ∧ LineageComplete: every
memory an answer rests on must resolve to a record, and that record's lineage
must trace back to an originating write event. No public memory leaderboard
scores this; it is part of the new category REMANENTIA defines (roadmap W1/W5).

This module scores it. Each answer cites the provenance ids it used; a
provenance store maps ids to nodes that link toward their origin. An answer is
*provenance-visible* iff it cites at least one id and every cited id is both
queryable (present in the store) and lineage-complete (its parent chain reaches
an originating write without dangling or cycling). The headline is the fraction
of answers that are provenance-visible — the auditability the category requires.

Pure and deterministic; no model calls.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class ProvenanceNode:
    """One provenance record and its link toward an originating write."""

    id: str
    origin: bool = False  # True = an originating write event (lineage root)
    parent: str | None = None  # next id toward the origin, or None


@dataclass(frozen=True)
class AnswerLineage:
    """The provenance ids one answer rested on."""

    answer_id: str
    cited_ids: tuple[str, ...]


@dataclass(frozen=True)
class LineageReport:
    """Provenance-visibility outcome over a set of answers."""

    total: int
    visible: int
    completeness: float  # fraction of answers that are provenance-visible
    incomplete_answers: tuple[str, ...]


def is_lineage_complete(node_id: str, store: Mapping[str, ProvenanceNode]) -> bool:
    """Whether *node_id*'s parent chain reaches an origin without dangling/cycle.

    Returns ``False`` on a missing (non-queryable) node, a cycle, or a chain that
    ends before any originating write.
    """
    seen: set[str] = set()
    current: str | None = node_id
    while current is not None:
        if current in seen:
            return False  # cycle — not a sound lineage
        seen.add(current)
        node = store.get(current)
        if node is None:
            return False  # dangling — not queryable
        if node.origin:
            return True
        current = node.parent
    return False  # chain ended without reaching an originating write


def answer_provenance_visible(answer: AnswerLineage, store: Mapping[str, ProvenanceNode]) -> bool:
    """Whether an answer cites provenance that is fully queryable + lineage-complete."""
    if not answer.cited_ids:
        return False
    return all(is_lineage_complete(cid, store) for cid in answer.cited_ids)


def lineage_completeness(
    answers: Sequence[AnswerLineage],
    store: Mapping[str, ProvenanceNode],
) -> LineageReport:
    """Return the provenance-visibility report over *answers* against *store*."""
    incomplete: list[str] = []
    visible = 0
    for answer in answers:
        if answer_provenance_visible(answer, store):
            visible += 1
        else:
            incomplete.append(answer.answer_id)
    total = len(answers)
    completeness = 1.0 if total == 0 else visible / total
    return LineageReport(
        total=total,
        visible=visible,
        completeness=completeness,
        incomplete_answers=tuple(incomplete),
    )
