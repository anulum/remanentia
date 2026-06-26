# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — tests for recall calibration and abstention

"""Real-ledger tests for recall calibration and abstention decisions."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from recall_ledger import RecallLedger

from recall_calibration import (
    CalibratedRecallGate,
    CalibrationExample,
    calibration_examples_from_ledger,
    build_report,
    evaluate_holdout,
    fit_gate_from_ledger,
    main,
    split_examples,
)


def _labelled_ledger(path: Path) -> RecallLedger:
    """Create a persisted recall ledger with score and correctness labels."""

    ledger = RecallLedger(path)
    rows = [
        ("high correct", 0.95, True),
        ("strong correct", 0.82, True),
        ("middle wrong", 0.61, False),
        ("low correct", 0.42, True),
        ("weak wrong", 0.31, False),
        ("missing score", None, False),
    ]
    for query, score, was_correct in rows:
        event_id = ledger.record(
            query,
            ["semantic:memory.md"] if score is not None else [],
            top_k=3,
            score=score,
            by="calibration-test",
        )
        ledger.record_outcome(event_id, was_correct=was_correct)
    return ledger


def test_calibration_examples_load_real_recall_ledger(tmp_path: Path) -> None:
    """Calibration examples should come from scored, correctness-labelled recalls."""

    ledger = _labelled_ledger(tmp_path / "recall.jsonl")

    examples = calibration_examples_from_ledger(ledger)

    assert [example.query for example in examples] == [
        "high correct",
        "strong correct",
        "middle wrong",
        "low correct",
        "weak wrong",
    ]
    assert examples[0].score == 0.95
    assert examples[-1].was_correct is False


def test_gate_fits_threshold_and_abstains_below_it(tmp_path: Path) -> None:
    """The fitted gate should hold empirical error at or below the target."""

    ledger = _labelled_ledger(tmp_path / "recall.jsonl")

    gate = fit_gate_from_ledger(ledger, target_error_rate=0.2, min_labelled=5)

    assert gate.threshold == 0.82
    accept = gate.decide(0.95)
    reject = gate.decide(0.61)
    missing = gate.decide(None)
    assert accept.abstain is False
    assert accept.estimated_correctness == 1.0
    assert reject.abstain is True
    assert reject.reason == "score_below_threshold"
    assert missing.abstain is True
    assert missing.reason == "no_score"


def test_cold_start_gate_abstains_without_enough_labels(tmp_path: Path) -> None:
    """Cold-start calibration should abstain rather than invent a threshold."""

    ledger = RecallLedger(tmp_path / "recall.jsonl")
    event_id = ledger.record("only one", ["semantic:memory.md"], top_k=3, score=0.9)
    ledger.record_outcome(event_id, was_correct=True)

    gate = fit_gate_from_ledger(ledger, target_error_rate=0.1, min_labelled=3)

    assert gate.threshold is None
    assert gate.decide(0.99).abstain is True
    assert gate.decide(0.99).reason == "insufficient_labels"


def test_holdout_evaluation_measures_coverage_and_calibration_error(tmp_path: Path) -> None:
    """Held-out evaluation should report abstention coverage and calibration error."""

    ledger = _labelled_ledger(tmp_path / "recall.jsonl")
    examples = calibration_examples_from_ledger(ledger)
    train, holdout = split_examples(examples, holdout_fraction=0.4)
    gate = CalibratedRecallGate.fit(train, target_error_rate=0.34, min_labelled=3)

    report = evaluate_holdout(gate, holdout)

    assert report.n_total == 2
    assert report.n_accepted == 1
    assert report.coverage == 0.5
    assert report.accuracy == 1.0
    assert report.calibration_error == 0.0


def test_gate_reports_cold_start_and_invalid_configuration(tmp_path: Path) -> None:
    """Cold-start summaries and invalid operator settings should be explicit."""

    ledger = RecallLedger(tmp_path / "recall.jsonl")
    event_id = ledger.record("only one", ["semantic:memory.md"], top_k=3, score=0.9)
    ledger.record_outcome(event_id, was_correct=True)
    examples = calibration_examples_from_ledger(ledger)

    gate = CalibratedRecallGate.fit(examples, target_error_rate=0.1, min_labelled=3)

    assert gate.to_dict()["threshold"] is None
    assert gate.estimate_correctness(0.1) == (1.0, 1)
    assert CalibratedRecallGate.fit((), min_labelled=3).estimate_correctness(0.5) == (0.0, 0)
    try:
        CalibratedRecallGate.fit(examples, target_error_rate=1.1)
    except ValueError as exc:
        assert "target_error_rate" in str(exc)
    else:
        raise AssertionError("invalid target_error_rate accepted")
    try:
        CalibratedRecallGate.fit(examples, min_labelled=0)
    except ValueError as exc:
        assert "min_labelled" in str(exc)
    else:
        raise AssertionError("invalid min_labelled accepted")


def test_fallback_threshold_when_target_error_is_unreachable() -> None:
    """If every threshold violates the target, the gate should answer only at max score."""

    examples = (
        CalibrationExample("a", "wrong high", 0.9, False),
        CalibrationExample("b", "wrong low", 0.4, False),
    )

    gate = CalibratedRecallGate.fit(examples, target_error_rate=0.0, min_labelled=2)

    assert gate.threshold == 0.9
    assert gate.decide(0.8).abstain is True


def test_no_accepted_holdout_and_split_edge_cases(tmp_path: Path) -> None:
    """Evaluation should handle empty accepted sets and small deterministic splits."""

    ledger = _labelled_ledger(tmp_path / "recall.jsonl")
    examples = calibration_examples_from_ledger(ledger)
    gate = CalibratedRecallGate.fit(examples, target_error_rate=0.0, min_labelled=5)

    report = evaluate_holdout(gate, [examples[-1]])

    assert report.to_dict()["accuracy"] is None
    assert report.n_abstained == 1
    assert split_examples((), holdout_fraction=0.4) == ((), ())
    assert split_examples(examples, holdout_fraction=0.0) == (examples, ())
    single = (examples[0],)
    assert split_examples(single, holdout_fraction=0.5) == (single, ())
    try:
        split_examples(examples, holdout_fraction=1.0)
    except ValueError as exc:
        assert "holdout_fraction" in str(exc)
    else:
        raise AssertionError("invalid holdout_fraction accepted")


def test_build_report_and_text_cli_use_default_ledger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The operator report path should work through the default ledger env var."""

    ledger = _labelled_ledger(tmp_path / "recall.jsonl")
    monkeypatch.setenv("REMANENTIA_RECALL_LEDGER", str(ledger.path))

    report = build_report(ledger, target_error_rate=0.2, min_labelled=5, holdout_fraction=0.4)
    exit_code = main(["--target-error-rate", "0.2", "--min-labelled", "5"])

    assert report["threshold"] == 0.82
    assert exit_code == 0
    assert "Recall calibration" in capsys.readouterr().out


def test_main_json_cli_accepts_explicit_ledger_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The in-process CLI path should cover explicit ledgers and JSON output."""

    ledger = _labelled_ledger(tmp_path / "recall.jsonl")

    exit_code = main(
        [
            "--ledger",
            str(ledger.path),
            "--target-error-rate",
            "0.2",
            "--min-labelled",
            "5",
            "--holdout-fraction",
            "0.4",
            "--json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["threshold"] == 0.82
    assert payload["holdout"]["n_total"] == 2


def test_cli_reports_json_calibration_summary(tmp_path: Path) -> None:
    """The module CLI should read the ledger and emit a machine-readable report."""

    ledger = _labelled_ledger(tmp_path / "recall.jsonl")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "recall_calibration",
            "--ledger",
            str(ledger.path),
            "--target-error-rate",
            "0.2",
            "--min-labelled",
            "5",
            "--holdout-fraction",
            "0.4",
            "--json",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["threshold"] == 0.82
    assert payload["n_labelled"] == 5
    assert payload["holdout"]["n_total"] == 2
    assert payload["target_error_rate"] == 0.2
