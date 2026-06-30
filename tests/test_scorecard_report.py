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

from scorecard_report import _confidence, build_run_report, parse_results


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
        assert s.lineages[1].answer_id == "q2"  # fallback f"q{total}"
        ids = {n.id for n in s.provenance}
        assert ids == {"f1", "f2"}  # f1 deduped via setdefault

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
        assert r.lineage_measured is True
        # 2 of 3 answers cite provenance (third cites nothing -> not visible)
        assert abs(r.lineage_completeness - 2 / 3) < 1e-9
        d = r.as_dict()
        assert d["abstention_measured"] is True
        assert d["lineage_measured"] is True
        assert d["accuracy_target"] == 0.9

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
