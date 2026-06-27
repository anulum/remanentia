# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — shared finding claim-axis vocabulary

"""Shared vocabulary for Remanentia finding evidence and claim-status axes.

The finding ingest path and the recall render gate both consume the same wire
axes. Keeping the vocabulary here prevents the admission boundary from drifting
away from the read-side presentation lattice as the claim model evolves.
"""

from __future__ import annotations

REFERENCE_VALIDATED = "reference-validated"
"""Claim status for findings validated against a trusted reference."""

REFUTED_STATUS = "refuted"
"""Claim status for promoted negative findings."""

CLAIM_STATUSES = frozenset(
    {
        REFERENCE_VALIDATED,
        "bounded-model",
        "bounded-support",
        "validation-gap",
        "external-dependency-blocked",
        "roadmap",
        "toolchain-gated",
        REFUTED_STATUS,
    }
)
"""Known claim-status wire values."""

FALSIFIED = "falsified"
"""Evidence kind for tested-false claims."""

PRODUCER_ASSERTED = "producer-asserted"
"""Evidence kind for claims asserted by a producer but not independently checked."""

EVIDENCE_KINDS = frozenset(
    {
        "measured",
        "curated",
        "formally-proven",
        "hardware-validated",
        "noise-limited",
        PRODUCER_ASSERTED,
        FALSIFIED,
    }
)
"""Known evidence-kind wire values."""

ADMITTED = "admitted"
"""Admission-axis value for findings accepted above the boundary."""

ADMISSIONS = frozenset({ADMITTED, "floored", "rejected"})
"""Known admission-axis wire values."""

VERIFIED_AT_SOURCE = "verified-at-source"
"""Freshness value for findings re-checked at source in the current session."""

FRESHNESSES = frozenset({VERIFIED_AT_SOURCE, "traceable-unchecked", "untraceable"})
"""Known freshness-axis wire values."""


def normalize_axis(value: object) -> str | None:
    """Return the canonical wire form for one claim-axis value.

    Parameters
    ----------
    value
        Raw axis value from a finding record.

    Returns
    -------
    str | None
        Lowercase hyphenated wire value. ``None`` remains ``None`` because
        freshness may be additive; non-string values become ``""`` so callers
        can treat them as unknown without crashing.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value.replace("_", "-").lower()
    return ""


def is_falsified_reference_validated(
    evidence_kind: object,
    claim_status: object,
) -> bool:
    """Return whether a finding carries the forbidden refuted-axis pair.

    Parameters
    ----------
    evidence_kind
        Raw evidence-kind axis value.
    claim_status
        Raw claim-status axis value.

    Returns
    -------
    bool
        ``True`` when falsified evidence is still labelled as
        reference-validated and must be rejected before persistence.
    """
    return (
        normalize_axis(evidence_kind) == FALSIFIED
        and normalize_axis(claim_status) == REFERENCE_VALIDATED
    )
