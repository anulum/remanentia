# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — tests for shared claim-axis vocabulary

"""Tests for :mod:`claim_axes`."""

from __future__ import annotations

from claim_axes import (
    ADMITTED,
    CLAIM_STATUSES,
    EVIDENCE_KINDS,
    FALSIFIED,
    REFERENCE_VALIDATED,
    REFUTED_STATUS,
    VERIFIED_AT_SOURCE,
    is_falsified_reference_validated,
    normalize_axis,
)


def test_normalize_axis_accepts_wire_variants() -> None:
    assert normalize_axis("Reference_Validated") == REFERENCE_VALIDATED
    assert normalize_axis("VERIFIED_AT_SOURCE") == VERIFIED_AT_SOURCE
    assert normalize_axis(None) is None
    assert normalize_axis(42) == ""


def test_shared_vocabulary_contains_render_and_ingest_axes() -> None:
    assert REFERENCE_VALIDATED in CLAIM_STATUSES
    assert REFUTED_STATUS in CLAIM_STATUSES
    assert FALSIFIED in EVIDENCE_KINDS
    assert ADMITTED == "admitted"


def test_falsified_reference_validated_pair_is_forbidden() -> None:
    assert is_falsified_reference_validated("falsified", "reference_validated") is True
    assert is_falsified_reference_validated("FALSIFIED", "REFERENCE-VALIDATED") is True
    assert is_falsified_reference_validated("falsified", "refuted") is False
    assert is_falsified_reference_validated("measured", "reference_validated") is False
    assert is_falsified_reference_validated(42, "reference_validated") is False
