# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Tests for the channel-ablation harness

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ablation_channels import (
    ALL_CHANNELS,
    CONFIGS,
    ItemResult,
    _derive_findings,
    aggregate,
    format_report,
    main,
    run_item,
)


# ── Config shape ─────────────────────────────────────────────────────


class TestConfigs:
    def test_all_configs_present(self):
        assert set(CONFIGS) == {"ALL", "no_bm25", "no_entity", "no_temporal", "no_session"}

    def test_all_channels_is_baseline(self):
        assert CONFIGS["ALL"] == ALL_CHANNELS

    def test_each_ablation_removes_one(self):
        for name, channels in CONFIGS.items():
            if name == "ALL":
                continue
            removed = name.removeprefix("no_")
            assert removed not in channels
            assert len(channels) == len(ALL_CHANNELS) - 1


# ── run_item ─────────────────────────────────────────────────────────


def _tiny_oracle_item() -> dict:
    """Minimal oracle-shaped item sufficient for retrieval plumbing."""
    return {
        "question_id": "q1",
        "question_type": "multi-session",
        "question": "When did the YouTube views hit 542?",
        "answer": "On March 15",
        "question_date": "2026-03-20",
        "haystack_dates": ["2026-03-15", "2026-03-22"],
        "haystack_session_ids": ["s1", "s2"],
        "haystack_sessions": [
            [
                {"role": "user", "content": "My YouTube tutorial has 542 views."},
                {"role": "assistant", "content": "Impressive reach."},
            ],
            [
                {"role": "user", "content": "I uploaded a second tutorial last week."},
                {"role": "assistant", "content": "What did you cover this time?"},
            ],
        ],
        "answer_session_ids": ["s1"],
    }


class TestRunItem:
    def test_returns_result_per_config(self):
        rows = run_item(_tiny_oracle_item(), top_k=5)
        assert len(rows) == len(CONFIGS)
        assert {r.config for r in rows} == set(CONFIGS)

    def test_recall_between_zero_and_one(self):
        for r in run_item(_tiny_oracle_item(), top_k=5):
            assert 0.0 <= r.recall <= 1.0

    def test_question_id_propagates(self):
        rows = run_item(_tiny_oracle_item(), top_k=5)
        assert all(r.question_id == "q1" for r in rows)

    def test_qtype_propagates(self):
        rows = run_item(_tiny_oracle_item(), top_k=5)
        assert all(r.qtype == "multi-session" for r in rows)


# ── aggregate ────────────────────────────────────────────────────────


class TestAggregate:
    def test_empty_input(self):
        assert aggregate([]) == {}

    def test_single_config_single_qtype(self):
        rows = [
            ItemResult("q1", "multi-session", "ALL", 1.0),
            ItemResult("q2", "multi-session", "ALL", 0.5),
        ]
        out = aggregate(rows)
        assert out["ALL"]["multi-session"]["mean"] == pytest.approx(0.75)
        assert out["ALL"]["multi-session"]["n"] == 2

    def test_overall_bucket_accumulates_all_qtypes(self):
        rows = [
            ItemResult("q1", "multi-session", "ALL", 1.0),
            ItemResult("q2", "temporal-reasoning", "ALL", 0.0),
        ]
        out = aggregate(rows)
        assert out["ALL"]["_overall"]["mean"] == pytest.approx(0.5)
        assert out["ALL"]["_overall"]["n"] == 2


# ── _derive_findings ─────────────────────────────────────────────────


class TestDeriveFindings:
    def test_flags_counterproductive_channel(self):
        agg = {
            "ALL": {"temporal-reasoning": {"mean": 0.95, "n": 100}},
            "no_temporal": {"temporal-reasoning": {"mean": 0.98, "n": 100}},
        }
        prose = _derive_findings(agg)
        assert "temporal" in prose
        assert "hurts" in prose or "rises" in prose

    def test_flags_load_bearing_channel(self):
        agg = {
            "ALL": {"multi-session": {"mean": 0.98, "n": 100}},
            "no_session": {"multi-session": {"mean": 0.96, "n": 100}},
        }
        prose = _derive_findings(agg)
        assert "session" in prose
        assert "helps" in prose or "drops" in prose

    def test_no_movement_reports_ceiling(self):
        agg = {
            "ALL": {"qt": {"mean": 1.0, "n": 10}},
            "no_bm25": {"qt": {"mean": 1.0, "n": 10}},
            "no_entity": {"qt": {"mean": 1.0, "n": 10}},
            "no_temporal": {"qt": {"mean": 1.0, "n": 10}},
            "no_session": {"qt": {"mean": 1.0, "n": 10}},
        }
        prose = _derive_findings(agg)
        assert "saturating" in prose or "roughly equal" in prose


# ── format_report ────────────────────────────────────────────────────


class TestFormatReport:
    def test_contains_every_config_column(self):
        agg = {c: {"qt": {"mean": 0.9, "n": 5}, "_overall": {"mean": 0.9, "n": 5}} for c in CONFIGS}
        md = format_report(agg, top_k=10, seed=42, limit=5, runtime_s=1.0)
        for config in CONFIGS:
            assert config in md

    def test_mentions_runtime(self):
        agg = {
            "ALL": {"_overall": {"mean": 1.0, "n": 1}},
            "no_bm25": {"_overall": {"mean": 1.0, "n": 1}},
            "no_entity": {"_overall": {"mean": 1.0, "n": 1}},
            "no_temporal": {"_overall": {"mean": 1.0, "n": 1}},
            "no_session": {"_overall": {"mean": 1.0, "n": 1}},
        }
        md = format_report(agg, top_k=5, seed=7, limit=1, runtime_s=2.5)
        assert "2.5" in md
        assert "seed=7" in md


# ── CLI smoke ────────────────────────────────────────────────────────


class TestCLI:
    def test_missing_oracle_returns_nonzero(self, tmp_path: Path):
        missing = tmp_path / "nope.json"
        code = main(
            [
                "--oracle",
                str(missing),
                "--limit",
                "1",
                "--out",
                str(tmp_path / "out.md"),
            ]
        )
        assert code == 1

    def test_end_to_end_on_synthetic_oracle(self, tmp_path: Path):
        oracle_path = tmp_path / "oracle.json"
        oracle_path.write_text(
            json.dumps([_tiny_oracle_item(), _tiny_oracle_item()]),
            encoding="utf-8",
        )
        out_md = tmp_path / "ablations.md"
        out_jsonl = tmp_path / "ablations.jsonl"
        code = main(
            [
                "--oracle",
                str(oracle_path),
                "--limit",
                "2",
                "--top-k",
                "5",
                "--out",
                str(out_md),
                "--json-out",
                str(out_jsonl),
                "--progress-every",
                "10",
                "--seed",
                "42",
            ]
        )
        assert code == 0
        assert out_md.exists()
        assert "ArcaneRetriever" in out_md.read_text(encoding="utf-8")
        # JSONL has one line per (item × config).
        lines = out_jsonl.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2 * len(CONFIGS)
