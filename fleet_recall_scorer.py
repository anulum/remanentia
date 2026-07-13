# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — fleet-fed recall scorer (axis 3)

"""Score the fleet-fed recall axis from the recall query-stream ledger.

The category claim "fleet-fed" says the memory is exercised and calibrated by
the real questions the whole fleet asks — not by a benchmark's question set and
not by the owning agent alone. The query stream already exists twice
(:mod:`recall_ledger` is the durable local sink, :mod:`bus_recall` mirrors the
same events to the hub), but nothing turned it into a scored axis: the stream
was recorded, never measured. This module is that scorer; the axis folds into
:mod:`world_class_scorecard` and :mod:`scorecard_report` beside abstention,
no-egress and lineage.

Three honesty rules, matching the harness's null-handling:

1. **Label coverage is reported, never assumed.** ``was_used`` and
   ``was_correct`` arrive later than the query record and may never arrive;
   the report carries how many answered queries actually hold each label, so a
   rate computed over three labelled records cannot masquerade as a measured
   axis over a thousand.
2. **Usage is not correctness.** ``usage_rate`` is the retrieval-precision
   proxy derived from loop closure (:mod:`recall_outcome_tracker`);
   ``fleet_accuracy`` scores only the verifier-supplied ``was_correct`` label.
   The two are reported separately and never mixed.
3. **Fleet-fed is objective.** A stream fed by one identity is self-fed;
   ``fleet_fed`` is simply "≥ 2 distinct ``by`` identities", so a single-agent
   run cannot claim the axis by volume alone.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from recall_ledger import RecallLedger, RecallQuery

__all__ = ["FleetRecallReport", "report_from_ledger", "score_fleet_recall"]


@dataclass(frozen=True)
class FleetRecallReport:
    """Scored view of one recall query-stream window.

    Parameters
    ----------
    queries
        Query records considered.
    agents
        Distinct querying identities (``by``) in the window.
    fleet_fed
        Whether the stream was fed by at least two distinct identities.
    answered
        Queries that returned at least one memory (``found``).
    answered_rate
        ``answered / queries``; 0.0 when no queries.
    usage_labelled
        Answered queries carrying a ``was_used`` outcome label.
    usage_rate
        Fraction of *labelled* answered queries with ``was_used=True``;
        0.0 when nothing is labelled. A usage proxy, not correctness.
    correctness_labelled
        Answered queries carrying a ``was_correct`` verifier label.
    fleet_accuracy
        Fraction of *correctness-labelled* answered queries with
        ``was_correct=True``; 0.0 when nothing is labelled. The scored
        accuracy of the axis.
    contradictions
        Records that both abstained and returned memories — a producer
        integrity fault worth surfacing, never silently dropped.
    measured
        Whether any query records existed; ``False`` keeps the axis
        honestly dark instead of scoring an empty stream as perfect.
    """

    queries: int
    agents: int
    fleet_fed: bool
    answered: int
    answered_rate: float
    usage_labelled: int
    usage_rate: float
    correctness_labelled: int
    fleet_accuracy: float
    contradictions: int
    measured: bool

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable view of the report."""
        return {
            "queries": self.queries,
            "agents": self.agents,
            "fleet_fed": self.fleet_fed,
            "answered": self.answered,
            "answered_rate": round(self.answered_rate, 4),
            "usage_labelled": self.usage_labelled,
            "usage_rate": round(self.usage_rate, 4),
            "correctness_labelled": self.correctness_labelled,
            "fleet_accuracy": round(self.fleet_accuracy, 4),
            "contradictions": self.contradictions,
            "measured": self.measured,
        }


def score_fleet_recall(records: Iterable[RecallQuery]) -> FleetRecallReport:
    """Score one window of recall query records into the fleet-fed axis.

    Parameters
    ----------
    records
        Recall query records with outcomes merged in, as
        :meth:`recall_ledger.RecallLedger.queries` yields them.

    Returns
    -------
    FleetRecallReport
        Deterministic, JSON-serialisable axis report. Rates over an empty
        denominator are 0.0 and the corresponding count fields make the
        emptiness visible.
    """
    queries = 0
    agents: set[str] = set()
    answered = 0
    usage_labelled = 0
    used = 0
    correctness_labelled = 0
    correct = 0
    contradictions = 0

    for record in records:
        queries += 1
        agents.add(record.by)
        if record.abstained is True and record.found:
            contradictions += 1
        if not record.found:
            continue
        answered += 1
        if record.was_used is not None:
            usage_labelled += 1
            if record.was_used:
                used += 1
        if record.was_correct is not None:
            correctness_labelled += 1
            if record.was_correct:
                correct += 1

    return FleetRecallReport(
        queries=queries,
        agents=len(agents),
        fleet_fed=len(agents) >= 2,
        answered=answered,
        answered_rate=answered / queries if queries else 0.0,
        usage_labelled=usage_labelled,
        usage_rate=used / usage_labelled if usage_labelled else 0.0,
        correctness_labelled=correctness_labelled,
        fleet_accuracy=correct / correctness_labelled if correctness_labelled else 0.0,
        contradictions=contradictions,
        measured=queries > 0,
    )


def report_from_ledger(path: str | Path) -> FleetRecallReport:
    """Score the fleet-fed axis straight from a recall ledger file.

    Parameters
    ----------
    path
        The ledger JSONL written by :class:`recall_ledger.RecallLedger` —
        the same file the production recall path appends to. A missing file
        scores as an unmeasured axis, not an error.

    Returns
    -------
    FleetRecallReport
        The scored axis for every query record in the ledger.
    """
    return score_fleet_recall(RecallLedger(Path(path)).queries())
