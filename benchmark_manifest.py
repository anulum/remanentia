# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — benchmark reproducibility manifests

"""Build reproducibility manifests for committed benchmark evidence artefacts.

Beyond hashing the evidence files, the manifest aggregates a per-benchmark
``headlines`` section: how many runs exist, which distinct RNG seeds they used,
and the run-to-run accuracy variance band. A benchmark is *headline-eligible*
only when at least two distinct seeds back it (roadmap "never a single run");
``--require-multi-seed`` turns that eligibility into a hard exit-code gate so a
docs or CI lane can refuse single-seed headline claims.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, cast

JsonObject = dict[str, object]

SCHEMA_VERSION = 1
DEFAULT_OUTPUT = Path("benchmarks/reproducibility_manifest.json")


def build_benchmark_manifest(
    *,
    report_paths: list[Path],
    artifact_paths: list[Path] | None = None,
    commands: list[str] | None = None,
    repo_root: Path = Path("."),
    generated_at: str | None = None,
    git_sha: str | None = None,
) -> JsonObject:
    """Build a deterministic benchmark reproducibility manifest.

    Parameters
    ----------
    report_paths
        Benchmark evidence report JSON files to summarise and hash.
    artifact_paths
        Additional committed artefacts to hash, such as frontier reports.
    commands
        Reproduction commands associated with the reports.
    repo_root
        Repository root used to render relative paths and resolve source files.
    generated_at
        Optional ISO-8601 timestamp. Defaults to current UTC time.
    git_sha
        Optional Git commit SHA override. When ``None``, the current repository
        ``HEAD`` is used when available.

    Returns
    -------
    dict[str, object]
        JSON-serialisable reproducibility manifest.
    """
    if not report_paths:
        raise ValueError("At least one benchmark report path is required")

    root = repo_root.resolve()
    reports = [_report_entry(path, root) for path in report_paths]
    artifacts = [_file_record(path, root) for path in artifact_paths or []]
    stamp = generated_at or datetime.now(timezone.utc).isoformat()
    head = _git_head(root) if git_sha is None else git_sha

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": stamp,
        "git": {"head": head},
        "environment": {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "commands": commands or [],
        "reports": reports,
        "artifacts": artifacts,
        "headlines": _headline_entries(reports),
    }


def write_benchmark_manifest(
    output: Path,
    *,
    report_paths: list[Path],
    artifact_paths: list[Path] | None = None,
    commands: list[str] | None = None,
    repo_root: Path = Path("."),
    generated_at: str | None = None,
    git_sha: str | None = None,
) -> JsonObject:
    """Write a benchmark reproducibility manifest to ``output`` and return it."""
    manifest = build_benchmark_manifest(
        report_paths=report_paths,
        artifact_paths=artifact_paths,
        commands=commands,
        repo_root=repo_root,
        generated_at=generated_at,
        git_sha=git_sha,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def _report_entry(path: Path, repo_root: Path) -> JsonObject:
    """Load and summarise one benchmark evidence report."""
    payload = _load_object(path)
    score = _required_object(payload, "score", path)
    source_path = payload.get("source_path")
    source_record: JsonObject
    if isinstance(source_path, str) and source_path:
        source_record = _optional_file_record(repo_root / source_path, repo_root, source_path)
    else:
        source_record = {"path": "", "exists": False}

    return {
        "benchmark": _required_str(payload, "benchmark", path),
        "file": _file_record(path, repo_root),
        "source_artifact": source_record,
        "source_format": payload.get("source_format")
        if isinstance(payload.get("source_format"), str)
        else "",
        "n_records": _optional_int(payload.get("n_records")),
        "score": {
            "correct": _optional_int(score.get("correct")),
            "total": _optional_int(score.get("total")),
            "accuracy": _optional_number(score.get("accuracy")),
        },
        "runtime": _optional_object(payload.get("runtime")) or {},
        "tokens": _optional_object(payload.get("tokens")) or {},
        "judge": _optional_object(payload.get("judge")) or {},
        "seeds": _seed_list(payload.get("seeds")),
        "settings": _string_list(payload.get("settings")),
        "readers": _string_list(payload.get("readers")),
    }


def _seed_list(value: object) -> list[int]:
    """Return the report's distinct integral seeds, sorted; never fabricated."""
    if not isinstance(value, list):
        return []
    seeds = {seed for seed in (_optional_int(item) for item in value) if seed is not None}
    return sorted(seeds)


def _string_list(value: object) -> list[str]:
    """Return the report's distinct non-empty string labels, sorted; never fabricated."""
    if not isinstance(value, list):
        return []
    return sorted({item for item in value if isinstance(item, str) and item})


def _group_label(entry: JsonObject, field: str) -> str:
    """Collapse one report's *field* label list into a headline group label.

    A single results file carrying two labels is an integrity fault
    (``mixed``) and artefacts predating the stamp are ``unknown``. Applies to
    ``settings`` (oracle vs full-S inflates ~30 %) and ``readers`` (a
    sovereign local reader and a cloud reader are different systems).
    """
    labels = cast(list[str], entry[field])
    if len(labels) == 1:
        return labels[0]
    return "mixed" if labels else "unknown"


def _headline_entries(reports: list[JsonObject]) -> list[JsonObject]:
    """Aggregate per-(benchmark, setting, reader) headline eligibility and variance.

    Groups the manifest's report entries by benchmark name, setting label AND
    reader label — oracle and full-S runs never pool into one band, and a
    sovereign local-reader run never pools with a cloud-reader run of the same
    setting (SOVEREIGN_MEMORY_EVALUATION guardrail 2). A group is
    headline-eligible only when its reader is pinned and its runs carry at
    least two distinct RNG seeds; a single run, runs without seed metadata
    (every pre-L6.5 artefact), runs without reader metadata (every pre-L6.7
    artefact), or a mixed-label results file are honestly marked ineligible
    instead of being presented as a settled number. Accuracy is reported as a
    variance band across runs, not a point estimate.
    """
    grouped: dict[tuple[str, str, str], list[JsonObject]] = {}
    for report in reports:
        key = (
            cast(str, report["benchmark"]),
            _group_label(report, "settings"),
            _group_label(report, "readers"),
        )
        grouped.setdefault(key, []).append(report)

    headlines: list[JsonObject] = []
    for benchmark, setting, reader in sorted(grouped):
        entries = grouped[(benchmark, setting, reader)]
        seeds = sorted({seed for entry in entries for seed in cast(list[int], entry["seeds"])})
        accuracies = [
            accuracy
            for entry in entries
            if (accuracy := cast(JsonObject, entry["score"]).get("accuracy")) is not None
            and isinstance(accuracy, float)
        ]
        if setting == "mixed":
            eligible = False
            reason = "mixed_settings"
        elif setting == "unknown":
            eligible = False
            reason = "no_setting_metadata"
        elif reader == "mixed":
            eligible = False
            reason = "mixed_readers"
        elif reader == "unknown":
            eligible = False
            reason = "no_reader_metadata"
        elif len(seeds) >= 2:
            eligible = True
            reason = "ok"
        elif not seeds:
            eligible = False
            reason = "no_seed_metadata"
        else:
            eligible = False
            reason = "single_seed"
        headlines.append(
            {
                "benchmark": benchmark,
                "setting": setting,
                "reader": reader,
                "runs": len(entries),
                "seeds": seeds,
                "accuracy": {
                    "n": len(accuracies),
                    "mean": round(statistics.fmean(accuracies), 4) if accuracies else None,
                    "min": round(min(accuracies), 4) if accuracies else None,
                    "max": round(max(accuracies), 4) if accuracies else None,
                    "stdev": round(statistics.stdev(accuracies), 4)
                    if len(accuracies) >= 2
                    else None,
                },
                "variance_band": [round(min(accuracies), 4), round(max(accuracies), 4)]
                if accuracies
                else None,
                "headline_eligible": eligible,
                "reason": reason,
            }
        )
    return headlines


def _file_record(path: Path, repo_root: Path) -> JsonObject:
    """Return path, byte-size, and SHA-256 metadata for an existing file."""
    if not path.exists():
        raise FileNotFoundError(path)
    data = path.read_bytes()
    return {
        "path": _relative_path(path, repo_root),
        "exists": True,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _optional_file_record(path: Path, repo_root: Path, display_path: str) -> JsonObject:
    """Return file metadata or an explicit missing-file record."""
    if not path.exists():
        return {"path": display_path, "exists": False}
    return _file_record(path, repo_root)


def _relative_path(path: Path, repo_root: Path) -> str:
    """Render ``path`` relative to ``repo_root`` when possible."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(repo_root).as_posix()
    except ValueError:
        return str(resolved)


def _load_object(path: Path) -> Mapping[str, object]:
    """Load a JSON object from ``path``."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Benchmark report must contain a JSON object: {path}")
    return cast(Mapping[str, object], payload)


def _required_object(parent: Mapping[str, object], key: str, path: Path) -> Mapping[str, object]:
    """Return a required object field from ``parent``."""
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Benchmark report must contain object field {key}: {path}")
    return cast(Mapping[str, object], value)


def _optional_object(value: object) -> Mapping[str, object] | None:
    """Return ``value`` as an object when it has object shape."""
    if not isinstance(value, dict):
        return None
    return cast(Mapping[str, object], value)


def _required_str(parent: Mapping[str, object], key: str, path: Path) -> str:
    """Return a required non-empty string field from ``parent``."""
    value = parent.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Benchmark report must contain non-empty string field {key}: {path}")
    return value


def _optional_number(value: object) -> float | None:
    """Return JSON numeric values as ``float`` while rejecting booleans."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _optional_int(value: object) -> int | None:
    """Return integral JSON values as ``int`` while rejecting booleans."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _git_head(repo_root: Path) -> str:
    """Return the current Git HEAD SHA, or an empty string outside Git.

    The resolver reads Git metadata directly so the manifest generator never
    shells out while handling benchmark artefacts.
    """
    git_dir = repo_root / ".git"
    if git_dir.is_file():
        content = _read_text(git_dir)
        if not content.startswith("gitdir:"):
            return ""
        git_dir = (repo_root / content.split(":", 1)[1].strip()).resolve()
    if not git_dir.is_dir():
        return ""
    head = _read_text(git_dir / "HEAD")
    if not head:
        return ""
    if not head.startswith("ref:"):
        return head
    ref = head.split(":", 1)[1].strip()
    return _read_text(git_dir / ref)


def _read_text(path: Path) -> str:
    """Read a small Git metadata file, returning an empty string on failure."""
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _build_parser() -> argparse.ArgumentParser:
    """Construct the benchmark-manifest command-line parser."""
    parser = argparse.ArgumentParser(
        prog="remanentia-benchmark-manifest",
        description="Build a reproducibility manifest for benchmark evidence artefacts.",
    )
    parser.add_argument(
        "--report",
        action="append",
        type=Path,
        required=True,
        help="Benchmark evidence report JSON path; repeat for multiple reports",
    )
    parser.add_argument(
        "--artifact",
        action="append",
        type=Path,
        default=[],
        help="Additional artefact to hash; repeat for multiple artefacts",
    )
    parser.add_argument(
        "--command",
        action="append",
        default=[],
        help="Reproduction command to record; repeat for multiple commands",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Manifest output path")
    parser.add_argument("--repo-root", type=Path, default=Path("."), help="Repository root")
    parser.add_argument("--generated-at", default=None, help="ISO-8601 timestamp override")
    parser.add_argument(
        "--git-sha", default=None, help="Git SHA override for deterministic reports"
    )
    parser.add_argument(
        "--require-multi-seed",
        action="store_true",
        help="exit 1 when any benchmark lacks >=2 distinct seeds (single-seed headline gate)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark-manifest CLI."""
    args = _build_parser().parse_args(argv)
    output = cast(Path, args.output)
    manifest = write_benchmark_manifest(
        output,
        report_paths=cast(list[Path], args.report),
        artifact_paths=cast(list[Path], args.artifact),
        commands=cast(list[str], args.command),
        repo_root=cast(Path, args.repo_root),
        generated_at=cast(str | None, args.generated_at),
        git_sha=cast(str | None, args.git_sha),
    )
    print(f"wrote benchmark manifest -> {output}")
    if cast(bool, args.require_multi_seed):
        ineligible = [
            headline
            for headline in cast(list[JsonObject], manifest["headlines"])
            if not headline["headline_eligible"]
        ]
        if ineligible:
            for headline in ineligible:
                print(
                    f"single-seed headline gate: {headline['benchmark']}"
                    f"/{headline['setting']}/{headline['reader']} ({headline['reason']}, "
                    f"runs={headline['runs']}, seeds={headline['seeds']})"
                )
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
