# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — benchmark efficiency frontier reporting

"""Build benchmark efficiency-frontier reports from committed evidence files."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, cast

JsonObject = dict[str, object]

SCHEMA_VERSION = 1
DEFAULT_OUTPUT = Path("benchmarks/efficiency_frontier_report.json")


@dataclass(frozen=True)
class EfficiencyCandidate:
    """Comparable benchmark point for accuracy, token, and latency evidence."""

    benchmark: str
    label: str
    source: str
    accuracy: float
    correct: int | None
    total: int | None
    tokens_total: int | None
    p95_latency_ms: float | None
    implementation: str

    @property
    def accuracy_points_per_1k_tokens(self) -> float | None:
        """Return accuracy points per 1,000 measured tokens, if available."""
        if self.tokens_total is None or self.tokens_total <= 0:
            return None
        return round(self.accuracy * 1000.0 / self.tokens_total, 3)

    @property
    def has_complete_frontier_metrics(self) -> bool:
        """Return whether the point can participate in frontier comparison."""
        return self.accuracy_points_per_1k_tokens is not None and self.p95_latency_ms is not None

    def as_dict(self, *, frontier: bool) -> JsonObject:
        """Serialise the candidate for a JSON report."""
        return {
            "benchmark": self.benchmark,
            "label": self.label,
            "implementation": self.implementation,
            "source": self.source,
            "accuracy": self.accuracy,
            "correct": self.correct,
            "total": self.total,
            "tokens_total": self.tokens_total,
            "accuracy_points_per_1k_tokens": self.accuracy_points_per_1k_tokens,
            "p95_latency_ms": self.p95_latency_ms,
            "frontier": frontier,
        }


def build_efficiency_report(
    *,
    report_paths: list[Path],
    baseline_path: Path | None = None,
    generated_at: str | None = None,
) -> JsonObject:
    """Build a deterministic efficiency-frontier report.

    Parameters
    ----------
    report_paths
        Benchmark evidence report JSON paths produced by
        ``remanentia-benchmark-report``.
    baseline_path
        Optional baseline JSON file with a top-level ``baselines`` array.
    generated_at
        Optional ISO-8601 timestamp. Defaults to current UTC time.

    Returns
    -------
    dict[str, object]
        JSON-serialisable frontier report.
    """
    if not report_paths:
        raise ValueError("At least one benchmark report path is required")

    stamp = generated_at or datetime.now(timezone.utc).isoformat()
    candidates = [_candidate_from_report(path) for path in report_paths]
    if baseline_path is not None:
        candidates.extend(_load_baselines(baseline_path))

    frontier_flags = _frontier_flags(candidates)
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": stamp,
        "inputs": {
            "reports": [str(path) for path in report_paths],
            "baselines": None if baseline_path is None else str(baseline_path),
        },
        "metric_notes": {
            "accuracy_points_per_1k_tokens": (
                "accuracy percentage points divided by measured total tokens and scaled to 1,000"
            ),
            "missing_values": "null means the committed artefact does not contain that measurement",
            "frontier": (
                "true only for same-benchmark points with accuracy, tokens, and p95 latency present"
            ),
        },
        "candidates": [
            candidate.as_dict(frontier=frontier_flags[index])
            for index, candidate in enumerate(candidates)
        ],
    }


def _candidate_from_report(path: Path) -> EfficiencyCandidate:
    """Load one benchmark evidence report into a frontier candidate."""
    payload = _load_object(path)
    benchmark = _required_str(payload, "benchmark", path)
    score = _required_object(payload, "score", path)
    tokens = _optional_object(payload.get("tokens"))
    runtime = _optional_object(payload.get("runtime"))
    latency = _optional_object(runtime.get("latency_ms")) if runtime is not None else None

    prompt_tokens = _optional_int(tokens.get("judge_prompt_estimate")) if tokens else None
    completion_tokens = _optional_int(tokens.get("judge_completion")) if tokens else None
    tokens_total = _sum_optional_tokens(prompt_tokens, completion_tokens)
    source_path = payload.get("source_path")

    return EfficiencyCandidate(
        benchmark=benchmark,
        label="Remanentia",
        source=str(source_path) if isinstance(source_path, str) else str(path),
        accuracy=_required_number(score, "accuracy", path),
        correct=_optional_int(score.get("correct")),
        total=_optional_int(score.get("total")),
        tokens_total=tokens_total,
        p95_latency_ms=(_optional_number(latency.get("p95")) if latency is not None else None),
        implementation="remanentia",
    )


def _load_baselines(path: Path) -> list[EfficiencyCandidate]:
    """Load explicit baseline candidates from ``path``."""
    payload = _load_object(path)
    raw_baselines = payload.get("baselines")
    if not isinstance(raw_baselines, list):
        raise ValueError(f"Baseline file must contain a baselines array: {path}")

    baselines: list[EfficiencyCandidate] = []
    for index, value in enumerate(raw_baselines):
        if not isinstance(value, dict):
            raise ValueError(f"baseline item {index} is not an object")
        item = cast(Mapping[str, object], value)
        baselines.append(
            EfficiencyCandidate(
                benchmark=_required_str(item, "benchmark", path),
                label=_required_str(item, "label", path),
                source=_required_str(item, "source", path),
                accuracy=_required_number(item, "accuracy", path),
                correct=_optional_int(item.get("correct")),
                total=_optional_int(item.get("total")),
                tokens_total=_optional_int(item.get("tokens_total")),
                p95_latency_ms=_optional_number(item.get("p95_latency_ms")),
                implementation=_optional_str(item.get("implementation")) or "baseline",
            )
        )
    return baselines


def _frontier_flags(candidates: list[EfficiencyCandidate]) -> list[bool]:
    """Return Pareto-frontier flags for candidates within each benchmark."""
    flags: list[bool] = []
    for candidate in candidates:
        if not candidate.has_complete_frontier_metrics:
            flags.append(False)
            continue
        flags.append(
            not any(
                _dominates(other, candidate)
                for other in candidates
                if other.benchmark == candidate.benchmark and other is not candidate
            )
        )
    return flags


def _dominates(left: EfficiencyCandidate, right: EfficiencyCandidate) -> bool:
    """Return whether ``left`` dominates ``right`` on accuracy, tokens, and p95."""
    left_efficiency = left.accuracy_points_per_1k_tokens
    right_efficiency = right.accuracy_points_per_1k_tokens
    if (
        left_efficiency is None
        or right_efficiency is None
        or left.p95_latency_ms is None
        or right.p95_latency_ms is None
    ):
        return False
    no_worse = (
        left.accuracy >= right.accuracy
        and left_efficiency >= right_efficiency
        and left.p95_latency_ms <= right.p95_latency_ms
    )
    strictly_better = (
        left.accuracy > right.accuracy
        or left_efficiency > right_efficiency
        or left.p95_latency_ms < right.p95_latency_ms
    )
    return no_worse and strictly_better


def _sum_optional_tokens(prompt_tokens: int | None, completion_tokens: int | None) -> int | None:
    """Return the total measured tokens, or ``None`` when unavailable."""
    total = (prompt_tokens or 0) + (completion_tokens or 0)
    return total if total > 0 else None


def _load_object(path: Path) -> Mapping[str, object]:
    """Load a JSON object from ``path``."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return cast(Mapping[str, object], payload)


def _required_object(parent: Mapping[str, object], key: str, path: Path) -> Mapping[str, object]:
    """Return required object field ``key`` from ``parent``."""
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain object field {key}")
    return cast(Mapping[str, object], value)


def _optional_object(value: object) -> Mapping[str, object] | None:
    """Return ``value`` as an object when it has object shape."""
    if isinstance(value, dict):
        return cast(Mapping[str, object], value)
    return None


def _required_str(parent: Mapping[str, object], key: str, path: Path) -> str:
    """Return required string field ``key`` from ``parent``."""
    value = parent.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path} must contain non-empty string field {key}")
    return value


def _optional_str(value: object) -> str | None:
    """Return ``value`` when it is a non-empty string."""
    return value if isinstance(value, str) and value else None


def _required_number(parent: Mapping[str, object], key: str, path: Path) -> float:
    """Return required numeric field ``key`` from ``parent``."""
    value = _optional_number(parent.get(key))
    if value is None:
        raise ValueError(f"{path} must contain numeric field {key}")
    return value


def _optional_number(value: object) -> float | None:
    """Coerce JSON numeric values to ``float`` while rejecting booleans."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _optional_int(value: object) -> int | None:
    """Coerce integral JSON values to ``int`` while rejecting booleans."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _build_parser() -> argparse.ArgumentParser:
    """Construct the efficiency-frontier CLI parser."""
    parser = argparse.ArgumentParser(
        prog="remanentia-efficiency-frontier",
        description="Build an accuracy/token/latency frontier report from benchmark evidence.",
    )
    parser.add_argument(
        "--report",
        action="append",
        type=Path,
        required=True,
        help="Benchmark evidence report JSON path; repeat for multiple reports",
    )
    parser.add_argument("--baseline", type=Path, default=None, help="Optional baseline JSON path")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Report output path")
    parser.add_argument("--generated-at", default=None, help="ISO-8601 timestamp override")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the efficiency-frontier CLI."""
    args = _build_parser().parse_args(argv)
    report = build_efficiency_report(
        report_paths=cast(list[Path], args.report),
        baseline_path=cast(Path | None, args.baseline),
        generated_at=cast(str | None, args.generated_at),
    )
    output = cast(Path, args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote efficiency frontier -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
