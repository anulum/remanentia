# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Tests for judge_agreement

from __future__ import annotations

import json

import pytest

from judge_agreement import agreement_payload, agreement_stats


class TestAgreementStats:
    def test_perfect_agreement(self) -> None:
        pairs = [(True, True), (False, False), (True, True), (False, False)]
        stats = agreement_stats(pairs)
        assert stats["measured"] is True
        assert stats["answered"] == 4
        assert stats["agreement"] == 1.0
        assert stats["positive_agreement"] == 1.0
        assert stats["negative_agreement"] == 1.0
        assert stats["cohen_kappa"] == 1.0

    def test_partial_agreement_with_kappa_below_raw(self) -> None:
        # 3/4 raw agreement; kappa must correct for chance and land below it.
        pairs = [(True, True), (True, True), (False, False), (False, True)]
        stats = agreement_stats(pairs)
        assert stats["agreement"] == 0.75
        assert stats["positive_agreement"] == 1.0
        assert stats["negative_agreement"] == 0.5
        kappa = stats["cohen_kappa"]
        assert kappa is not None and 0.0 < kappa < 0.75
        assert kappa == pytest.approx(0.5, abs=0.01)

    def test_chance_level_candidate_has_zero_kappa(self) -> None:
        # Candidate says yes to everything: raw agreement = base rate, kappa = 0.
        pairs = [(True, True), (True, True), (False, True), (False, True)]
        stats = agreement_stats(pairs)
        assert stats["agreement"] == 0.5
        assert stats["cohen_kappa"] == 0.0

    def test_unanswered_pairs_counted_never_fabricated(self) -> None:
        pairs = [(True, True), (True, None), (False, None), (False, False)]
        stats = agreement_stats(pairs)
        assert stats["pairs"] == 4
        assert stats["answered"] == 2
        assert stats["unanswered"] == 2
        assert stats["agreement"] == 1.0  # over answered pairs only

    def test_empty_and_all_unanswered_report_not_measured(self) -> None:
        for pairs in ([], [(True, None), (False, None)]):
            stats = agreement_stats(pairs)
            assert stats["measured"] is False
            assert stats["agreement"] == 0.0
            assert stats["cohen_kappa"] is None
            assert stats["positive_agreement"] is None

    def test_single_class_reference_has_null_class_and_kappa_fields(self) -> None:
        # Reference all-yes: negative agreement undefined; if the candidate also
        # answers all-yes, chance correction is undefined too.
        stats = agreement_stats([(True, True), (True, True)])
        assert stats["agreement"] == 1.0
        assert stats["negative_agreement"] is None
        assert stats["cohen_kappa"] is None

    def test_single_class_reference_with_mixed_candidate_keeps_kappa(self) -> None:
        # Reference all-yes but candidate mixed: chance < 1, kappa defined (0 here).
        stats = agreement_stats([(True, True), (True, False)])
        assert stats["agreement"] == 0.5
        assert stats["cohen_kappa"] == 0.0


class TestAgreementPayload:
    def test_payload_wraps_stats_with_metadata(self) -> None:
        payload = agreement_payload(
            [(True, True), (False, False)],
            metadata={"reference_judge": "gpt-4o-mini", "candidate_judge": "gemma3:12b-ctx8k"},
        )
        assert payload["schema_version"] == 1
        assert payload["benchmark"] == "judge_agreement"
        assert payload["metadata"]["reference_judge"] == "gpt-4o-mini"
        assert payload["stats"]["agreement"] == 1.0
        json.dumps(payload)  # JSON-safe without a custom encoder

    def test_payload_without_metadata_is_empty_object(self) -> None:
        payload = agreement_payload([])
        assert payload["metadata"] == {}
        assert payload["stats"]["measured"] is False
