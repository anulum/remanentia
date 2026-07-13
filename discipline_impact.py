# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — write-discipline impact on retrieval

"""Quantify the write-discipline ceiling on retrieval (roadmap W4).

The new-category claim REMANENTIA makes is that write-side discipline *caps*
retrieval quality — that a memory written without its canonical fields can never
be retrieved as well as one written with them, no matter how good the retriever.
The fleet audit found the dominant discipline failure is a missing timestamp
(only ~15% of stimuli conform). This module measures the cost of exactly that
failure: it compares retrieval recall computed with the canonical timestamps
against recall computed with the timestamps stripped (the degraded write), and
reports the per-question-type drop — the discipline ceiling.

The retrieval runs themselves live in the (CPU-bound) recall harness; this module
holds the pure comparison so the ceiling is computed and tested deterministically.
Each aggregate is the `tools.retrieval_recall.aggregate_recall` shape:
``{qtype: {"mean@N": float, ...}}``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

RecallAggregate = Mapping[str, Mapping[str, float]]


@dataclass(frozen=True)
class RecallDelta:
    """Per-question-type recall under canonical vs timestamp-degraded writes."""

    qtype: str
    canonical: dict[int, float]
    degraded: dict[int, float]
    delta: dict[int, float]  # canonical - degraded; > 0 means discipline helps

    def delta_at(self, n: int) -> float:
        """Recall lost to the degraded write at cutoff *n* (positive = lost)."""
        return self.delta[n]


def discipline_impact(
    canonical: RecallAggregate,
    degraded: RecallAggregate,
    ns: Sequence[int] = (1, 3, 5, 10, 20),
) -> list[RecallDelta]:
    """Compare two recall aggregates into per-qtype recall deltas.

    Only question types present in *both* aggregates are compared. ``"overall"``
    is sorted first, then the rest alphabetically. Raises ``KeyError`` if a
    requested ``mean@n`` is absent from a shared qtype (a malformed aggregate).
    """
    shared = set(canonical) & set(degraded)
    deltas: list[RecallDelta] = []
    for qtype in shared:
        c = {n: canonical[qtype][f"mean@{n}"] for n in ns}
        d = {n: degraded[qtype][f"mean@{n}"] for n in ns}
        deltas.append(
            RecallDelta(
                qtype=qtype,
                canonical=c,
                degraded=d,
                delta={n: c[n] - d[n] for n in ns},
            )
        )
    deltas.sort(key=lambda r: (r.qtype != "overall", r.qtype))
    return deltas


def worst_hit(impacts: Sequence[RecallDelta], n: int = 10) -> RecallDelta | None:
    """Return the non-overall qtype whose recall@n falls most under degradation."""
    candidates = [r for r in impacts if r.qtype != "overall"]
    if not candidates:
        return None
    return max(candidates, key=lambda r: r.delta_at(n))


def impact_payload(
    impacts: Sequence[RecallDelta],
    *,
    worst_at: int = 10,
    metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Serialise *impacts* into the committable discipline-ceiling artefact.

    The payload carries every per-qtype canonical/degraded/delta curve (keyed
    by string cutoffs, JSON-safe), the worst-hit qtype at *worst_at*, and any
    run *metadata* the harness supplies (dataset, question count, timestamps).
    Deltas are rounded to 4 places; an empty impact list yields a payload with
    ``worst_hit: None`` rather than a fabricated entry.
    """
    worst = worst_hit(impacts, n=worst_at)
    return {
        "schema_version": 1,
        "benchmark": "longmemeval_write_discipline",
        "metadata": dict(metadata) if metadata else {},
        "impacts": [
            {
                "qtype": imp.qtype,
                "canonical": {str(n): round(v, 4) for n, v in imp.canonical.items()},
                "degraded": {str(n): round(v, 4) for n, v in imp.degraded.items()},
                "delta": {str(n): round(v, 4) for n, v in imp.delta.items()},
            }
            for imp in impacts
        ],
        "worst_hit": {
            "qtype": worst.qtype,
            "at": worst_at,
            "delta": round(worst.delta_at(worst_at), 4),
        }
        if worst is not None
        else None,
    }
