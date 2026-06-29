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

from scorecard_report import build_run_report, parse_results


def _write_results(path: Path) -> None:
    rows = [
        {"judge_label": True, "judge_model": "gpt-4o-mini", "question_type": "multi-session"},
        {"judge_label": False, "judge_model": "gpt-4o-mini", "question_type": "multi-session"},
        {"judge_label": True, "judge_model": "gpt-4o-mini", "question_type": "temporal-reasoning"},
        {"hypothesis": "unjudged row, no judge_label"},  # skipped
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


class TestParseResults:
    def test_accuracy_and_judge(self, tmp_path: Path) -> None:
        p = tmp_path / "r.jsonl"
        _write_results(p)
        s = parse_results(p)
        assert s.total == 3  # unjudged row skipped
        assert s.correct == 2
        assert abs(s.accuracy - 2 / 3) < 1e-9
        assert s.judge_models == ("gpt-4o-mini",)

    def test_blank_lines_skipped(self, tmp_path: Path) -> None:
        p = tmp_path / "r.jsonl"
        p.write_text('\n{"judge_label": true, "judge_model": "x"}\n\n', encoding="utf-8")
        s = parse_results(p)
        assert s.total == 1

    def test_empty_file(self, tmp_path: Path) -> None:
        p = tmp_path / "r.jsonl"
        p.write_text("", encoding="utf-8")
        s = parse_results(p)
        assert s.total == 0
        assert s.accuracy == 0.0
        assert s.judge_models == ()


class TestBuildRunReport:
    def test_cloud_reader(self, tmp_path: Path) -> None:
        p = tmp_path / "r.jsonl"
        _write_results(p)
        report = build_run_report(
            p,
            setting="full_s",
            reader="gpt-4o-mini",
            reader_endpoints=["https://api.openai.com/v1"],
        )
        assert report.setting == "full_s"
        assert report.judge == "gpt-4o-mini"
        assert report.questions == 3
        assert report.pure_local is False
        assert report.cloud_calls == 1
        assert report.abstention_measured is False
        assert report.lineage_measured is False

    def test_local_reader_and_unknown_judge(self, tmp_path: Path) -> None:
        p = tmp_path / "r.jsonl"
        p.write_text(json.dumps({"judge_label": True}) + "\n", encoding="utf-8")  # no judge_model
        report = build_run_report(
            p,
            setting="oracle",
            reader="qwen3:8b",
            reader_endpoints=["http://localhost:11434/v1"],
        )
        assert report.pure_local is True
        assert report.judge == "unknown"
        d = report.as_dict()
        assert d["setting"] == "oracle"
        assert d["reader"] == "qwen3:8b"
        assert d["pure_local"] is True
        assert d["accuracy"] == 1.0
