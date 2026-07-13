# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Tests for scorecard_report

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lineage_completeness import ProvenanceNode
from scorecard_report import (
    _confidence,
    build_run_report,
    load_provenance_store,
    parse_results,
)


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def _basic_rows() -> list[dict[str, object]]:
    return [
        {"judge_label": True, "judge_model": "gpt-4o-mini", "question_type": "multi-session"},
        {"judge_label": False, "judge_model": "gpt-4o-mini", "question_type": "multi-session"},
        {"judge_label": True, "judge_model": "gpt-4o-mini", "question_type": "temporal-reasoning"},
        {"hypothesis": "unjudged row, no judge_label"},  # skipped
    ]


class TestConfidenceHelper:
    def test_bool_rejected(self) -> None:
        assert _confidence(True) is None  # judge_label is bool — must not be a confidence

    def test_int_and_float(self) -> None:
        assert _confidence(1) == 1.0
        assert _confidence(0.7) == 0.7

    def test_non_numeric(self) -> None:
        assert _confidence("0.5") is None
        assert _confidence(None) is None


class TestParseResults:
    def test_accuracy_and_judge_no_axes(self, tmp_path: Path) -> None:
        p = tmp_path / "r.jsonl"
        _write(p, _basic_rows())
        s = parse_results(p)
        assert s.total == 3  # unjudged row skipped
        assert s.correct == 2
        assert abs(s.accuracy - 2 / 3) < 1e-9
        assert s.judge_models == ("gpt-4o-mini",)
        assert s.outcomes == ()  # no confidence -> abstention axis dark
        assert s.lineages == ()  # no cited_ids -> lineage axis dark

    def test_blank_lines_skipped(self, tmp_path: Path) -> None:
        p = tmp_path / "r.jsonl"
        p.write_text('\n{"judge_label": true, "judge_model": "x"}\n\n', encoding="utf-8")
        assert parse_results(p).total == 1

    def test_empty_file(self, tmp_path: Path) -> None:
        p = tmp_path / "r.jsonl"
        p.write_text("", encoding="utf-8")
        s = parse_results(p)
        assert s.total == 0
        assert s.accuracy == 0.0
        assert s.judge_models == ()
        assert s.outcomes == ()

    def test_confidence_axis_activates_when_all_rows_have_it(self, tmp_path: Path) -> None:
        p = tmp_path / "r.jsonl"
        _write(
            p,
            [
                {"judge_label": True, "judge_model": "m", "confidence": 0.9},
                {"judge_label": False, "judge_model": "m", "confidence": 0.2},
            ],
        )
        s = parse_results(p)
        assert len(s.outcomes) == 2
        assert s.outcomes[0].correct is True
        assert s.outcomes[0].confidence == 0.9

    def test_confidence_axis_dark_if_one_row_missing(self, tmp_path: Path) -> None:
        p = tmp_path / "r.jsonl"
        _write(
            p,
            [
                {"judge_label": True, "confidence": 0.9},
                {"judge_label": False},  # no confidence -> whole axis dark
            ],
        )
        assert parse_results(p).outcomes == ()

    def test_lineage_axis_with_question_id_and_dedup(self, tmp_path: Path) -> None:
        p = tmp_path / "r.jsonl"
        _write(
            p,
            [
                {"judge_label": True, "question_id": "q-A", "cited_ids": ["f1", "f2"]},
                {"judge_label": False, "cited_ids": ["f1"]},  # no question_id -> fallback id
            ],
        )
        s = parse_results(p)
        assert len(s.lineages) == 2
        assert s.lineages[0].answer_id == "q-A"
        assert s.lineages[0].cited_ids == ("f1", "f2")
        assert s.lineages[1].answer_id == "q2"  # fallback f"q{total}"

    def test_lineage_axis_dark_if_one_row_missing(self, tmp_path: Path) -> None:
        p = tmp_path / "r.jsonl"
        _write(
            p,
            [
                {"judge_label": True, "cited_ids": ["f1"]},
                {"judge_label": False},  # no cited_ids -> whole axis dark
            ],
        )
        assert parse_results(p).lineages == ()


class TestBuildRunReport:
    def test_no_axes_cloud(self, tmp_path: Path) -> None:
        p = tmp_path / "r.jsonl"
        _write(p, _basic_rows())
        r = build_run_report(
            p,
            setting="full_s",
            reader="gpt-4o-mini",
            reader_endpoints=["https://api.openai.com/v1"],
        )
        assert r.setting == "full_s"
        assert r.judge == "gpt-4o-mini"
        assert r.questions == 3
        assert r.pure_local is False
        assert r.cloud_calls == 1
        assert r.abstention_measured is False
        assert r.aurc == 0.0
        assert r.coverage_at_target == 0.0
        assert r.citation_measured is False
        assert r.citation_presence == 0.0
        assert r.lineage_measured is False
        assert r.lineage_completeness == 0.0

    def test_both_axes_measured_local(self, tmp_path: Path) -> None:
        p = tmp_path / "r.jsonl"
        _write(
            p,
            [
                {
                    "judge_label": True,
                    "judge_model": "gpt-4o",
                    "confidence": 0.9,
                    "cited_ids": ["f1"],
                },
                {
                    "judge_label": True,
                    "judge_model": "gpt-4o",
                    "confidence": 0.7,
                    "cited_ids": ["f2"],
                },
                {"judge_label": False, "judge_model": "gpt-4o", "confidence": 0.1, "cited_ids": []},
            ],
        )
        r = build_run_report(
            p,
            setting="full_s",
            reader="qwen3:8b",
            reader_endpoints=["http://localhost:11434/v1"],
            accuracy_target=0.9,
        )
        assert r.pure_local is True
        assert r.judge == "gpt-4o"
        assert r.abstention_measured is True
        # two high-confidence correct answers form a prefix >= 0.90 accuracy
        assert r.coverage_at_target > 0.0
        assert 0.0 <= r.aurc <= 1.0
        assert r.citation_measured is True
        # 2 of 3 answers cite at least one memory (third cites nothing)
        assert abs(r.citation_presence - 2 / 3) < 1e-9
        # No provenance store supplied -> lineage cannot be proven, stays dark.
        assert r.lineage_measured is False
        assert r.lineage_completeness == 0.0
        d = r.as_dict()
        assert d["abstention_measured"] is True
        assert d["citation_measured"] is True
        assert d["lineage_measured"] is False
        assert d["accuracy_target"] == 0.9

    def test_lineage_axis_verifies_against_real_store(self, tmp_path: Path) -> None:
        """A dangling citation must drag lineage below citation presence.

        This is the regression pin for the fabricated-provenance defect: the
        parser used to mint an origin node for every cited id, so the lineage
        axis was satisfied by construction and could never catch a citation
        that resolves nowhere.
        """
        p = tmp_path / "r.jsonl"
        _write(
            p,
            [
                {"judge_label": True, "question_id": "qa", "cited_ids": ["real"]},
                {"judge_label": True, "question_id": "qb", "cited_ids": ["chained"]},
                {"judge_label": False, "question_id": "qc", "cited_ids": ["dangling"]},
            ],
        )
        store = {
            "real": ProvenanceNode(id="real", origin=True, parent=None),
            "chained": ProvenanceNode(id="chained", origin=False, parent="real"),
            # "dangling" is deliberately absent — cited but resolves nowhere.
        }
        r = build_run_report(
            p,
            setting="full_s",
            reader="qwen3:8b",
            reader_endpoints=["http://localhost:11434/v1"],
            provenance_store=store,
        )
        assert r.citation_measured is True
        assert r.citation_presence == 1.0  # every answer cited something
        assert r.lineage_measured is True
        # ...but only 2 of 3 citations resolve to an originating write.
        assert abs(r.lineage_completeness - 2 / 3) < 1e-9

    def test_lineage_dark_without_citations_even_with_store(self, tmp_path: Path) -> None:
        p = tmp_path / "r.jsonl"
        _write(p, [{"judge_label": True}])  # no cited_ids at all
        r = build_run_report(
            p,
            setting="full_s",
            reader="qwen3:8b",
            reader_endpoints=["http://localhost:11434/v1"],
            provenance_store={"x": ProvenanceNode(id="x", origin=True, parent=None)},
        )
        assert r.citation_measured is False
        assert r.lineage_measured is False

    def test_unknown_judge_when_absent(self, tmp_path: Path) -> None:
        p = tmp_path / "r.jsonl"
        _write(p, [{"judge_label": True}])
        r = build_run_report(
            p,
            setting="oracle",
            reader="qwen3:8b",
            reader_endpoints=["http://localhost:11434/v1"],
        )
        assert r.judge == "unknown"
        assert r.as_dict()["accuracy"] == 1.0


class TestLoadProvenanceStore:
    def test_loads_nodes_with_defaults_and_blank_lines(self, tmp_path: Path) -> None:
        p = tmp_path / "prov.jsonl"
        p.write_text(
            '{"id": "root", "origin": true}\n\n{"id": "child", "parent": "root"}\n',
            encoding="utf-8",
        )
        store = load_provenance_store(p)
        assert store["root"] == ProvenanceNode(id="root", origin=True, parent=None)
        assert store["child"] == ProvenanceNode(id="child", origin=False, parent="root")

    def test_rejects_missing_or_empty_id(self, tmp_path: Path) -> None:
        p = tmp_path / "prov.jsonl"
        p.write_text('{"origin": true}\n', encoding="utf-8")
        with pytest.raises(ValueError, match="non-empty string id"):
            load_provenance_store(p)
        p.write_text('{"id": ""}\n', encoding="utf-8")
        with pytest.raises(ValueError, match="non-empty string id"):
            load_provenance_store(p)

    def test_rejects_non_boolean_origin(self, tmp_path: Path) -> None:
        p = tmp_path / "prov.jsonl"
        p.write_text('{"id": "n", "origin": "yes"}\n', encoding="utf-8")
        with pytest.raises(ValueError, match="origin must be a boolean"):
            load_provenance_store(p)

    def test_rejects_non_string_parent(self, tmp_path: Path) -> None:
        p = tmp_path / "prov.jsonl"
        p.write_text('{"id": "n", "parent": 7}\n', encoding="utf-8")
        with pytest.raises(ValueError, match="parent must be a string or null"):
            load_provenance_store(p)


class TestFleetRecallAxis:
    def test_axis_dark_without_ledger(self, tmp_path: Path) -> None:
        p = tmp_path / "r.jsonl"
        _write(p, _basic_rows())
        r = build_run_report(
            p,
            setting="full_s",
            reader="gpt-4o-mini",
            reader_endpoints=["https://api.openai.com/v1"],
        )
        assert r.fleet is None
        assert r.as_dict()["fleet_recall"] == {"measured": False}

    def test_axis_dark_with_empty_ledger(self, tmp_path: Path) -> None:
        p = tmp_path / "r.jsonl"
        _write(p, _basic_rows())
        r = build_run_report(
            p,
            setting="full_s",
            reader="gpt-4o-mini",
            reader_endpoints=["https://api.openai.com/v1"],
            recall_ledger=tmp_path / "absent_ledger.jsonl",
        )
        assert r.fleet is None
        assert r.as_dict()["fleet_recall"] == {"measured": False}

    def test_axis_scores_from_production_ledger(self, tmp_path: Path) -> None:
        from recall_ledger import RecallLedger

        p = tmp_path / "r.jsonl"
        _write(p, _basic_rows())
        ledger_path = tmp_path / "ledger.jsonl"
        ledger = RecallLedger(ledger_path)
        event = ledger.record("who owns portal", ["src:a"], top_k=5, by="claude")
        ledger.record("pin floor", ["src:b"], top_k=5, by="codex")
        ledger.record_outcome(event, was_used=True, was_correct=True)

        r = build_run_report(
            p,
            setting="full_s",
            reader="gpt-4o-mini",
            reader_endpoints=["https://api.openai.com/v1"],
            recall_ledger=ledger_path,
        )

        assert r.fleet is not None
        assert r.fleet.measured is True
        assert r.fleet.agents == 2
        assert r.fleet.fleet_fed is True
        d = r.as_dict()["fleet_recall"]
        assert isinstance(d, dict)
        assert d["queries"] == 2
        assert d["fleet_accuracy"] == 1.0
