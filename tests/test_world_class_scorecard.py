# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Tests for world_class_scorecard

from __future__ import annotations

from coverage_accuracy import Outcome
from lineage_completeness import LineageReport
from no_egress_audit import audit_endpoints
from world_class_scorecard import RunConfig, build_scorecard


def _outcomes() -> list[Outcome]:
    return [
        Outcome(correct=True, confidence=0.9),
        Outcome(correct=True, confidence=0.8),
        Outcome(correct=False, confidence=0.2),
    ]


class TestRunConfig:
    def test_comparable_when_all_match(self) -> None:
        a = RunConfig("full_s", "gpt-4o-mini", "gpt-4o")
        b = RunConfig("full_s", "gpt-4o-mini", "gpt-4o")
        assert a.comparable_to(b) is True

    def test_not_comparable_on_setting_reader_or_judge(self) -> None:
        base = RunConfig("full_s", "gpt-4o-mini", "gpt-4o")
        assert base.comparable_to(RunConfig("oracle", "gpt-4o-mini", "gpt-4o")) is False
        assert base.comparable_to(RunConfig("full_s", "qwen", "gpt-4o")) is False
        assert base.comparable_to(RunConfig("full_s", "gpt-4o-mini", "deepseek")) is False


class TestBuildScorecard:
    def test_cloud_run(self) -> None:
        config = RunConfig("full_s", "gpt-4o-mini", "gpt-4o")
        egress = audit_endpoints(["https://api.openai.com/v1"])
        lineage = LineageReport(total=3, visible=3, completeness=1.0, incomplete_answers=())
        card = build_scorecard(config, _outcomes(), egress, lineage)
        assert card.questions == 3
        assert round(card.accuracy, 3) == 0.667
        assert card.pure_local is False
        assert card.cloud_calls == 1
        assert card.lineage_completeness == 1.0
        # two high-confidence correct answers form a prefix >= 0.90 accuracy (cov 2/3)
        assert round(card.coverage_at_target, 3) == 0.667

    def test_pure_local_run_and_dict(self) -> None:
        config = RunConfig("full_s", "qwen3:8b", "gpt-4o")
        egress = audit_endpoints(["http://localhost:11434/v1", "http://localhost:11434/v1"])
        lineage = LineageReport(total=2, visible=1, completeness=0.5, incomplete_answers=("x",))
        # two correct high-confidence answers -> coverage_at_target(0.9) == 1.0
        outcomes = [Outcome(True, 0.9), Outcome(True, 0.5)]
        card = build_scorecard(config, outcomes, egress, lineage, accuracy_target=0.9)
        assert card.pure_local is True
        assert card.cloud_calls == 0
        assert card.coverage_at_target == 1.0
        d = card.as_dict()
        assert d["setting"] == "full_s"
        assert d["reader"] == "qwen3:8b"
        assert d["judge"] == "gpt-4o"
        assert d["pure_local"] is True
        assert d["lineage_completeness"] == 0.5
        assert d["accuracy"] == 1.0


class TestFleetRecallFold:
    def test_axis_dark_by_default(self) -> None:
        config = RunConfig("full_s", "gpt-4o-mini", "gpt-4o")
        egress = audit_endpoints(["https://api.openai.com/v1"])
        lineage = LineageReport(total=3, visible=3, completeness=1.0, incomplete_answers=())
        card = build_scorecard(config, _outcomes(), egress, lineage)
        assert card.fleet is None
        assert card.as_dict()["fleet_recall"] == {"measured": False}

    def test_axis_folds_when_supplied(self) -> None:
        from fleet_recall_scorer import score_fleet_recall
        from recall_ledger import RecallQuery

        records = [
            RecallQuery(
                event_id="e1",
                ts=1.0,
                by="claude",
                query="q",
                top_k=5,
                project="",
                returned_ids=("src:a",),
                found=True,
                was_used=True,
                was_correct=True,
            ),
            RecallQuery(
                event_id="e2",
                ts=2.0,
                by="codex",
                query="q2",
                top_k=5,
                project="",
                returned_ids=(),
                found=False,
            ),
        ]
        fleet = score_fleet_recall(records)
        config = RunConfig("full_s", "qwen3:8b", "gpt-4o")
        egress = audit_endpoints(["http://localhost:11434/v1"])
        lineage = LineageReport(total=2, visible=2, completeness=1.0, incomplete_answers=())
        card = build_scorecard(config, _outcomes(), egress, lineage, fleet=fleet)
        d = card.as_dict()["fleet_recall"]
        assert isinstance(d, dict)
        assert d["measured"] is True
        assert d["fleet_fed"] is True
        assert d["answered_rate"] == 0.5
        assert d["fleet_accuracy"] == 1.0
