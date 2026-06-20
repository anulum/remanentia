# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Benchmark accuracy-history recorder
"""Record LongMemEval/LOCOMO runs into a version-controlled accuracy ledger.

``bench_longmemeval.py`` overwrites a single ``results.jsonl`` on each run, so
per-round accuracy is otherwise lost (only manually archived baselines and prose
in ``CHANGELOG``/``docs`` survive). This module turns one judged ``results.jsonl``
— records of ``{"question_id", "judge_label", "question_type"}`` — into a compact
summary and appends it to an append-only JSONL ledger tracked in git, giving a
durable, machine-readable per-run history from which accuracy and the run-to-run
variance envelope can be recomputed at any time.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LEDGER = REPO_ROOT / "benchmarks" / "longmemeval_history.jsonl"


def _pct(part: int, whole: int) -> float:
    """Return *part* as a percentage of *whole* (one decimal); ``0.0`` if *whole* is 0."""
    return round(100.0 * part / whole, 1) if whole else 0.0


def summarise(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate judged question records into overall and per-category accuracy.

    Parameters
    ----------
    records
        Judged questions, each at least ``{"question_type": str, "judge_label":
        bool}``. A missing ``question_type`` is bucketed as ``"unknown"``;
        ``judge_label`` is coerced with :func:`bool` (``None``/absent counts as
        incorrect).

    Returns
    -------
    dict
        Keys ``n_total``, ``overall_correct``, ``overall_accuracy`` and
        ``by_category`` — the latter mapping each question type to
        ``{"correct", "total", "accuracy"}``. Accuracies are percentages rounded
        to one decimal place.
    """
    correct_by: dict[str, int] = defaultdict(int)
    total_by: dict[str, int] = defaultdict(int)
    for record in records:
        qtype = str(record.get("question_type", "unknown"))
        total_by[qtype] += 1
        if bool(record.get("judge_label")):
            correct_by[qtype] += 1
    n_total = sum(total_by.values())
    overall_correct = sum(correct_by.values())
    by_category = {
        qtype: {
            "correct": correct_by[qtype],
            "total": total_by[qtype],
            "accuracy": _pct(correct_by[qtype], total_by[qtype]),
        }
        for qtype in sorted(total_by)
    }
    return {
        "n_total": n_total,
        "overall_correct": overall_correct,
        "overall_accuracy": _pct(overall_correct, n_total),
        "by_category": by_category,
    }


def load_results(path: Path) -> list[dict[str, Any]]:
    """Load a judged ``results.jsonl`` into a list of records (blank lines skipped)."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def git_sha(repo: Path = REPO_ROOT) -> str:
    """Return the short HEAD SHA of *repo*, or ``"unknown"`` when git is unavailable."""
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def build_record(
    summary: dict[str, Any],
    *,
    timestamp: str,
    round_label: str,
    benchmark: str = "longmemeval",
    synth_model: str = "",
    judge_model: str = "",
    seed: int | None = None,
    sha: str = "",
    note: str = "",
) -> dict[str, Any]:
    """Assemble one self-describing ledger record from *summary* plus run metadata.

    Parameters
    ----------
    summary
        Output of :func:`summarise`.
    timestamp
        ISO-8601 run timestamp, supplied by the caller (not read from the clock)
        so records are deterministic under test.
    round_label
        Human round identifier, e.g. ``"R12"``.
    benchmark
        Benchmark name; defaults to ``"longmemeval"``.
    synth_model, judge_model
        Model identifiers used for answer synthesis and LLM-as-judge scoring.
    seed
        RNG seed pinned for the run, or ``None`` when unpinned.
    sha
        Source commit the run was produced from.
    note
        Free-text annotation (e.g. ``"P0.4 variance run 1/3"``).

    Returns
    -------
    dict
        A flat, JSON-serialisable record ready for :func:`append_record`.
    """
    return {
        "timestamp": timestamp,
        "benchmark": benchmark,
        "round": round_label,
        "git_sha": sha,
        "config": {"synth_model": synth_model, "judge_model": judge_model, "seed": seed},
        "n_total": summary["n_total"],
        "overall_correct": summary["overall_correct"],
        "overall_accuracy": summary["overall_accuracy"],
        "by_category": summary["by_category"],
        "note": note,
    }


def append_record(record: dict[str, Any], ledger: Path = DEFAULT_LEDGER) -> None:
    """Append *record* as one JSON line to *ledger*, creating parent directories."""
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def read_history(ledger: Path = DEFAULT_LEDGER) -> list[dict[str, Any]]:
    """Read all ledger records, returning ``[]`` when the ledger does not exist."""
    if not ledger.exists():
        return []
    lines = ledger.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def record_run(
    results_path: Path,
    *,
    round_label: str,
    timestamp: str | None = None,
    benchmark: str = "longmemeval",
    synth_model: str = "",
    judge_model: str = "",
    seed: int | None = None,
    note: str = "",
    ledger: Path = DEFAULT_LEDGER,
) -> dict[str, Any]:
    """Summarise *results_path*, build a record, append it to *ledger* and return it.

    *timestamp* defaults to the current UTC time in ISO-8601 when not given.
    """
    stamp = timestamp if timestamp is not None else datetime.now(timezone.utc).isoformat()
    summary = summarise(load_results(results_path))
    record = build_record(
        summary,
        timestamp=stamp,
        round_label=round_label,
        benchmark=benchmark,
        synth_model=synth_model,
        judge_model=judge_model,
        seed=seed,
        sha=git_sha(),
        note=note,
    )
    append_record(record, ledger)
    return record


def _build_arg_parser() -> argparse.ArgumentParser:
    """Construct the command-line parser for the recorder."""
    parser = argparse.ArgumentParser(
        prog="record_benchmark.py",
        description="Append a judged benchmark run to the accuracy-history ledger.",
    )
    parser.add_argument("results", type=Path, help="path to the judged results.jsonl")
    parser.add_argument("--round", dest="round_label", required=True, help="round label, e.g. R12")
    parser.add_argument("--benchmark", default="longmemeval", help="benchmark name")
    parser.add_argument("--synth-model", default="", help="answer-synthesis model id")
    parser.add_argument("--judge-model", default="", help="LLM-as-judge model id")
    parser.add_argument("--seed", type=int, default=None, help="pinned RNG seed")
    parser.add_argument("--note", default="", help="free-text annotation")
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER, help="ledger path")
    parser.add_argument("--timestamp", default=None, help="ISO-8601 override (default: now UTC)")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Record one run from the command line; prints the overall accuracy line."""
    args = _build_arg_parser().parse_args(argv)
    record = record_run(
        args.results,
        round_label=args.round_label,
        timestamp=args.timestamp,
        benchmark=args.benchmark,
        synth_model=args.synth_model,
        judge_model=args.judge_model,
        seed=args.seed,
        note=args.note,
        ledger=args.ledger,
    )
    print(
        f"recorded {record['benchmark']} {record['round']} "
        f"{record['overall_accuracy']}% ({record['overall_correct']}/{record['n_total']}) "
        f"-> {args.ledger}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
