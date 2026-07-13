# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — judge-agreement statistics (sovereign judge validation)

"""Measure how well a candidate judge reproduces a reference judge's labels.

The benchmark's correctness labels historically come from a hosted
LLM-as-judge (LongMemEval's evaluation protocol; every committed anchor was
scored by the same hosted judge, so scores stay comparable). A *sovereign*
memory system should not need a cloud call to know its own score — but a
local judge may only replace the hosted one on **measured** agreement, never
on assumption. This module holds the pure statistics for that measurement:
each item pairs the reference judge's boolean label with the candidate
judge's label (``None`` when the candidate failed to answer — counted,
never fabricated), and :func:`agreement_stats` reports raw agreement,
per-class agreement and Cohen's kappa so a chance-level candidate cannot
masquerade as a good one.

The judging loop that produces the pairs is the CPU/GPU-bound harness
(``tools/local_judge_agreement.py``); this module stays deterministic and
fully unit-tested.
"""

from __future__ import annotations

from collections.abc import Sequence

#: (reference_label, candidate_label) — candidate ``None`` = no usable answer.
JudgePair = tuple[bool, "bool | None"]


def agreement_stats(pairs: Sequence[JudgePair]) -> dict[str, object]:
    """Fold judge pairs into an honest agreement record.

    Returns raw ``agreement`` over the answered pairs, per-class agreement
    (``positive_agreement`` = candidate yes-rate on reference-yes items,
    ``negative_agreement`` analogously), and ``cohen_kappa``
    (chance-corrected; ``None`` when undefined, e.g. a single-class
    reference). ``measured`` is ``False`` when no pair was answered — an
    empty measurement is reported, not invented.
    """
    total = len(pairs)
    answered = [(ref, cand) for ref, cand in pairs if cand is not None]
    n = len(answered)
    if n == 0:
        return {
            "pairs": total,
            "answered": 0,
            "unanswered": total,
            "agreement": 0.0,
            "positive_agreement": None,
            "negative_agreement": None,
            "cohen_kappa": None,
            "measured": False,
        }

    both_yes = sum(1 for ref, cand in answered if ref and cand)
    both_no = sum(1 for ref, cand in answered if not ref and not cand)
    ref_yes = sum(1 for ref, _ in answered if ref)
    cand_yes = sum(1 for _, cand in answered if cand)
    agree = both_yes + both_no

    p_observed = agree / n
    p_chance = (ref_yes * cand_yes + (n - ref_yes) * (n - cand_yes)) / (n * n)
    kappa: float | None
    if p_chance == 1.0:
        # Both judges are single-class; chance correction is undefined.
        kappa = None
    else:
        kappa = (p_observed - p_chance) / (1.0 - p_chance)

    return {
        "pairs": total,
        "answered": n,
        "unanswered": total - n,
        "agreement": round(p_observed, 4),
        "positive_agreement": round(both_yes / ref_yes, 4) if ref_yes else None,
        "negative_agreement": round(both_no / (n - ref_yes), 4) if n - ref_yes else None,
        "cohen_kappa": round(kappa, 4) if kappa is not None else None,
        "measured": True,
    }


def agreement_payload(
    pairs: Sequence[JudgePair],
    *,
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    """Serialise the agreement measurement into a committable artefact."""
    return {
        "schema_version": 1,
        "benchmark": "judge_agreement",
        "metadata": dict(metadata) if metadata else {},
        "stats": agreement_stats(pairs),
    }
