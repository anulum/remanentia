# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — recall gate (read-side honesty surface: render lattice + INV-4)

"""Decide how a recalled finding — and a synthesis over several — may render.

This is the read side of the fleet-memory honesty contract: the admission gate
(SYNAPSE ``admit()``, consumed by :mod:`finding_ingest`) decides what *enters*
the store; this gate decides how what was stored may be *presented* on recall, so
a synthesised answer never renders above its weakest input (the INV-4 invariant).

The contract is the SCPN-STUDIO keeper's recall-gate spec, derived verbatim from
the ``scpn-studio-platform`` 0.8.0 source (``evidence/conformance.py present()``,
``evidence/federation.py _propagate_provenance``). It is pure — no platform
import is needed — and re-implemented here for the read side; importing
``scpn_studio_platform.evidence`` (pin ``>=0.8,<0.9``) for parity is optional.

Each finding folds to one of three presentation modes — ``validated`` (the only
assertive state, established against a trusted reference), ``boundary`` (a hedged
claim, true only within a stated scope, never upgraded), or ``refuted`` (a
tested-false / falsified claim, first-class but never validated). A synthesis
renders at the **minimum** of its inputs' modes: all-validated renders validated,
any boundary floors to boundary, and a single refuted input floors the whole
synthesis to refuted — a tested-false input surfaces its refutation, it is never
laundered into a softer hedge (the keeper's ratified choice, confirmed read-side).

Unknown or undeclared members on a gating axis (evidence kind, claim status,
admission) withhold validation rather than crash — forward-tolerance: a consumer
that does not recognise a value renders it at the boundary, never above it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from claim_axes import (
    ADMISSIONS,
    ADMITTED,
    CLAIM_STATUSES,
    EVIDENCE_KINDS,
    FALSIFIED,
    FRESHNESSES,
    PRODUCER_ASSERTED,
    REFERENCE_VALIDATED,
    REFUTED_STATUS,
    VERIFIED_AT_SOURCE,
    normalize_axis,
)

# ── The render lattice ───────────────────────────────────────────────────────

VALIDATED = "validated"
BOUNDARY = "boundary"
REFUTED = "refuted"

#: Strongest → weakest. ``min`` over these ranks is the synthesis floor.
_RANK = {VALIDATED: 2, BOUNDARY: 1, REFUTED: 0}

# ── Gating-axis members (wire strings; anything else is "unknown") ────────────

def _norm(value: object) -> str | None:
    """Canonicalise a wire value: lower-case, ``_``→``-``; ``None`` stays ``None``."""
    return normalize_axis(value)


def freshness_permits(freshness: str | None) -> bool:
    """Whether *freshness* allows a finding to render validated.

    True iff the source was re-checked this session (``verified-at-source``) or
    freshness is undeclared (``None`` — the axis is additive: an absent freshness
    does not by itself withhold validation). ``traceable-unchecked``,
    ``untraceable``, and any unknown value withhold it. Freshness can only ever
    *withhold* validation; it never turns a non-validated verdict into validated.
    """
    fresh = _norm(freshness)
    if fresh is None:
        return True
    if fresh not in FRESHNESSES:
        return False
    return fresh == VERIFIED_AT_SOURCE


def present(
    evidence_kind: str | None,
    claim_status: str | None,
    admission: str | None,
    freshness: str | None,
) -> str:
    """Fold one finding's axes to a single presentation mode.

    The order is load-bearing and faithful to the platform source: an unknown
    gating axis floors first, then a negative finding is refuted, then an
    unverified self-claim is held at the boundary, and only a
    reference-validated + admitted + fresh finding reaches validated.
    """
    kind = _norm(evidence_kind)
    status = _norm(claim_status)
    admit = _norm(admission)

    # 1. Any unknown/undeclared member on a gating axis withholds validation.
    #    This check is strictly first, faithful to the platform source: a
    #    finding with an unknown sibling axis floors to boundary even if another
    #    axis would read falsified/refuted. (In valid data a refuted finding
    #    always carries a known evidence_kind, so this edge does not arise; the
    #    parity-preserving literal order is kept rather than a local safety tweak.)
    if kind not in EVIDENCE_KINDS or status not in CLAIM_STATUSES or admit not in ADMISSIONS:
        return BOUNDARY
    # 2. A negative finding is refuted, even against a reference-validated boundary.
    if kind == FALSIFIED or status == REFUTED_STATUS:
        return REFUTED
    # 3. An unverified self-claim can never be laundered to validated.
    if kind == PRODUCER_ASSERTED:
        return BOUNDARY
    # 4. The one path to validated.
    if status == REFERENCE_VALIDATED and admit == ADMITTED and freshness_permits(freshness):
        return VALIDATED
    # 5. Everything else holds at the boundary.
    return BOUNDARY


def renders_validated(
    evidence_kind: str | None,
    claim_status: str | None,
    admission: str | None,
    freshness: str | None,
) -> bool:
    """Whether one finding renders as the assertive ``validated`` mode."""
    return present(evidence_kind, claim_status, admission, freshness) == VALIDATED


def render_synthesis(input_modes: Iterable[str]) -> str:
    """The mode of a recall synthesised over *input_modes* — the INV-4 floor.

    Never above the weakest input: all validated → validated; any boundary (no
    refuted) → boundary (the platform's ``_propagate_provenance`` exactly); any
    refuted → refuted (a synthesis resting on a tested-false input surfaces the
    refutation). A synthesis over no inputs asserts nothing, so it renders at the
    boundary — never validated.
    """
    modes = list(input_modes)
    if not modes:
        return BOUNDARY
    return min(modes, key=lambda m: _RANK.get(m, _RANK[BOUNDARY]))


def renders_validated_synthesis(input_modes: Iterable[str]) -> bool:
    """Whether a synthesis renders validated — iff it has inputs and all validated."""
    modes = list(input_modes)
    return bool(modes) and all(mode == VALIDATED for mode in modes)


def mode_for_record(record: Mapping[str, object]) -> str:
    """Presentation mode for one stored finding (a :mod:`finding_ingest` record).

    Bridges the ingest frontmatter to :func:`present`: the stored ``verdict``
    (``accept``/``floor`` from the admission gate) maps to the admission axis —
    only a cleanly accepted finding is ``admitted``; a floored one is held at the
    boundary like any non-admitted finding.
    """
    verdict = _norm(record.get("verdict"))
    admission = ADMITTED if verdict == "accept" else "floored"
    return present(
        _norm(record.get("evidence_kind")),
        _norm(record.get("claim_status")),
        admission,
        _norm(record.get("freshness")),
    )
