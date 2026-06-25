# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — tests for the read-side recall gate

"""Tests for :mod:`recall_gate`, pinned to the SCPN-STUDIO keeper's recall-gate
spec (render lattice §1, per-finding fold §2, the §3 status table, INV-4 synthesis
floor §4, evidence-kind gating §5). The §3 table and the fold rules give exact
expected outputs, so the suite asserts the contract row by row rather than a
hand-wavy sample.
"""

from __future__ import annotations

import pytest

from recall_gate import (
    BOUNDARY,
    REFUTED,
    VALIDATED,
    freshness_permits,
    mode_for_record,
    present,
    render_synthesis,
    renders_validated,
    renders_validated_synthesis,
)


def _validated_axes(**over):
    """A finding on the one path to validated; override to walk off it."""
    axes = {
        "evidence_kind": "measured",
        "claim_status": "reference-validated",
        "admission": "admitted",
        "freshness": "verified-at-source",
    }
    axes.update(over)
    return axes


class TestPresentValidatedPath:
    def test_reference_validated_admitted_fresh_is_validated(self):
        assert present(**_validated_axes()) == VALIDATED

    def test_undeclared_freshness_still_validates(self):
        # §2: freshness is additive — None does not withhold validation.
        assert present(**_validated_axes(freshness=None)) == VALIDATED

    def test_traceable_unchecked_withholds_validation(self):
        assert present(**_validated_axes(freshness="traceable-unchecked")) == BOUNDARY

    def test_untraceable_withholds_validation(self):
        assert present(**_validated_axes(freshness="untraceable")) == BOUNDARY

    def test_not_admitted_withholds_validation(self):
        assert present(**_validated_axes(admission="floored")) == BOUNDARY


class TestPresentNegativeAndSelfClaim:
    def test_refuted_status_is_refuted_even_when_reference_validated(self):
        # §2: a negative finding is refuted even against a reference-validated boundary.
        assert present("measured", "refuted", "admitted", "verified-at-source") == REFUTED

    def test_falsified_kind_is_refuted(self):
        assert present("falsified", "reference-validated", "admitted", "verified-at-source") == (
            REFUTED
        )

    def test_producer_asserted_is_boundary_always(self):
        # §5: an unverified self-claim is never validated, independent of freshness.
        assert present("producer-asserted", "reference-validated", "admitted", None) == BOUNDARY
        assert (
            present("producer-asserted", "reference-validated", "admitted", "verified-at-source")
            == BOUNDARY
        )


class TestPresentUnknownAxes:
    def test_unknown_evidence_kind_is_boundary(self):
        assert present("speculative", "reference-validated", "admitted", "verified-at-source") == (
            BOUNDARY
        )

    def test_none_claim_status_is_boundary(self):
        # A decision/outcome carries no claim_status — renders at the boundary.
        assert present("measured", None, "admitted", "verified-at-source") == BOUNDARY

    def test_unknown_admission_is_boundary(self):
        assert present("measured", "reference-validated", "pending", "verified-at-source") == (
            BOUNDARY
        )

    def test_falsified_with_unknown_status_floors_to_boundary(self):
        # Faithful to the platform: the unknown-gating-axis check is strictly
        # first, so an unknown sibling axis floors even a falsified finding to
        # boundary. (Does not arise in valid data — a refuted/falsified finding
        # always carries a known claim_status.)
        assert present("falsified", None, "admitted", None) == BOUNDARY

    def test_refuted_status_with_unknown_kind_floors_to_boundary(self):
        assert present(None, "refuted", "admitted", None) == BOUNDARY


class TestStatusTable:
    """§3 — every claim_status alone (measured kind, admitted, verified) → mode."""

    @pytest.mark.parametrize(
        "status,expected",
        [
            ("reference-validated", VALIDATED),
            ("bounded-model", BOUNDARY),
            ("bounded-support", BOUNDARY),
            ("validation-gap", BOUNDARY),
            ("external-dependency-blocked", BOUNDARY),
            ("roadmap", BOUNDARY),
            ("toolchain-gated", BOUNDARY),
            ("refuted", REFUTED),
        ],
    )
    def test_status_to_mode(self, status, expected):
        assert present("measured", status, "admitted", "verified-at-source") == expected


class TestNormalisation:
    def test_underscore_and_case_are_canonicalised(self):
        # The wire may carry REFERENCE_VALIDATED / Verified_At_Source variants.
        assert present("MEASURED", "Reference_Validated", "ADMITTED", "VERIFIED_AT_SOURCE") == (
            VALIDATED
        )


class TestFreshnessPermits:
    @pytest.mark.parametrize(
        "freshness,expected",
        [
            ("verified-at-source", True),
            (None, True),
            ("traceable-unchecked", False),
            ("untraceable", False),
            ("nonsense", False),
        ],
    )
    def test_permits(self, freshness, expected):
        assert freshness_permits(freshness) is expected


class TestRendersValidated:
    def test_true_on_validated_path(self):
        assert renders_validated(**_validated_axes()) is True

    def test_false_off_path(self):
        assert renders_validated(**_validated_axes(claim_status="bounded-support")) is False


class TestSynthesisFloor:
    """§4 — render = min over inputs; never above the weakest."""

    def test_all_validated_renders_validated(self):
        assert render_synthesis([VALIDATED, VALIDATED, VALIDATED]) == VALIDATED

    def test_any_boundary_floors_to_boundary(self):
        assert render_synthesis([VALIDATED, BOUNDARY, VALIDATED]) == BOUNDARY

    def test_any_refuted_floors_to_refuted(self):
        # The keeper-ratified open choice: a refuted input floors the whole synthesis.
        assert render_synthesis([VALIDATED, BOUNDARY, REFUTED]) == REFUTED

    def test_refuted_beats_boundary_for_the_floor(self):
        assert render_synthesis([BOUNDARY, REFUTED]) == REFUTED

    def test_single_input_passes_through(self):
        assert render_synthesis([BOUNDARY]) == BOUNDARY

    def test_empty_synthesis_is_boundary_never_validated(self):
        assert render_synthesis([]) == BOUNDARY

    def test_unknown_mode_treated_as_boundary_rank(self):
        # A mode string the lattice does not know ranks as boundary, not validated.
        assert render_synthesis([VALIDATED, "mystery"]) in (BOUNDARY, "mystery")


class TestRendersValidatedSynthesis:
    def test_all_validated_is_true(self):
        assert renders_validated_synthesis([VALIDATED, VALIDATED]) is True

    def test_one_boundary_is_false(self):
        assert renders_validated_synthesis([VALIDATED, BOUNDARY]) is False

    def test_empty_is_false(self):
        assert renders_validated_synthesis([]) is False


class TestModeForRecord:
    def test_accepted_reference_validated_fresh_is_validated(self):
        record = {
            "verdict": "accept",
            "evidence_kind": "measured",
            "claim_status": "reference-validated",
            "freshness": "verified-at-source",
        }
        assert mode_for_record(record) == VALIDATED

    def test_floored_verdict_is_boundary(self):
        record = {
            "verdict": "floor",
            "evidence_kind": "measured",
            "claim_status": "reference-validated",
            "freshness": "verified-at-source",
        }
        assert mode_for_record(record) == BOUNDARY

    def test_refuted_record_is_refuted(self):
        record = {"verdict": "accept", "evidence_kind": "measured", "claim_status": "refuted"}
        assert mode_for_record(record) == REFUTED

    def test_missing_axes_default_to_boundary(self):
        assert mode_for_record({"verdict": "accept"}) == BOUNDARY
