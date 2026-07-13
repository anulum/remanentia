# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — benchmark reproducibility manifest tests

"""Tests for benchmark reproducibility manifests built from real files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, cast

import pytest

from benchmark_manifest import build_benchmark_manifest, main, write_benchmark_manifest


def _map(value: object) -> Mapping[str, object]:
    """Assert and cast JSON object values for strict-mypy assertions."""
    assert isinstance(value, dict)
    return cast(Mapping[str, object], value)


def _list(value: object) -> list[object]:
    """Assert and cast JSON array values for strict-mypy assertions."""
    assert isinstance(value, list)
    return value


def _write_json(path: Path, payload: object) -> None:
    """Write JSON through the same filesystem path used by production code."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    """Return the SHA-256 hash for a test artefact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _report(path: Path, *, source_path: str = "data/results.jsonl") -> None:
    """Create a benchmark evidence report compatible with the manifest builder."""
    _write_json(
        path,
        {
            "schema_version": 1,
            "benchmark": "longmemeval",
            "source_path": source_path,
            "source_format": "jsonl",
            "generated_at": "2026-06-27T12:00:00+00:00",
            "n_records": 2,
            "score": {"correct": 1, "total": 2, "accuracy": 50.0},
            "runtime": {"latency_ms": {"mean": 10.0, "p50": 10.0, "p95": 12.0}},
            "tokens": {"judge_prompt_estimate": 100, "judge_completion": 4},
            "judge": {"models": ["judge"], "prompt_sha256": ["a" * 64]},
        },
    )


def test_build_benchmark_manifest_hashes_reports_and_sources(tmp_path: Path) -> None:
    """The manifest records report/source hashes, commands, and score summaries."""
    report_path = tmp_path / "benchmarks" / "longmemeval_report.json"
    source_path = tmp_path / "data" / "results.jsonl"
    extra_path = tmp_path / "benchmarks" / "efficiency_frontier_report.json"
    source_path.parent.mkdir(parents=True)
    source_path.write_text('{"question_id":"q1"}\n', encoding="utf-8")
    extra_path.parent.mkdir(parents=True)
    extra_path.write_text('{"schema_version":1}\n', encoding="utf-8")
    _report(report_path)

    manifest = build_benchmark_manifest(
        report_paths=[report_path],
        artifact_paths=[extra_path],
        commands=["remanentia-benchmark-report data/results.jsonl --benchmark longmemeval"],
        repo_root=tmp_path,
        generated_at="2026-06-27T13:00:00+00:00",
        git_sha="abc123",
    )

    assert manifest["schema_version"] == 1
    assert manifest["generated_at"] == "2026-06-27T13:00:00+00:00"
    assert manifest["git"] == {"head": "abc123"}
    assert manifest["commands"] == [
        "remanentia-benchmark-report data/results.jsonl --benchmark longmemeval"
    ]
    reports = _list(manifest["reports"])
    report = _map(reports[0])
    report_file = _map(report["file"])
    source_file = _map(report["source_artifact"])
    assert report["benchmark"] == "longmemeval"
    assert report["score"] == {"correct": 1, "total": 2, "accuracy": 50.0}
    assert report_file["path"] == "benchmarks/longmemeval_report.json"
    assert report_file["sha256"] == _sha256(report_path)
    assert report_file["bytes"] == report_path.stat().st_size
    assert source_file["path"] == "data/results.jsonl"
    assert source_file["sha256"] == _sha256(source_path)
    artifacts = _list(manifest["artifacts"])
    assert _map(artifacts[0])["path"] == "benchmarks/efficiency_frontier_report.json"


def test_build_benchmark_manifest_preserves_missing_source(tmp_path: Path) -> None:
    """Missing report source paths stay explicit instead of being fabricated."""
    report_path = tmp_path / "report.json"
    _report(report_path, source_path="data/missing.jsonl")

    manifest = build_benchmark_manifest(
        report_paths=[report_path],
        repo_root=tmp_path,
        generated_at="2026-06-27T13:00:00+00:00",
        git_sha=None,
    )

    report = _map(_list(manifest["reports"])[0])
    source = _map(report["source_artifact"])
    assert source == {"path": "data/missing.jsonl", "exists": False}
    assert _map(manifest["git"])["head"] == ""


def test_write_benchmark_manifest_creates_parent_directories(tmp_path: Path) -> None:
    """The writer persists deterministic JSON and returns the manifest object."""
    report_path = tmp_path / "report.json"
    output = tmp_path / "out" / "manifest.json"
    _report(report_path, source_path="")

    manifest = write_benchmark_manifest(
        output,
        report_paths=[report_path],
        repo_root=tmp_path,
        generated_at="2026-06-27T13:00:00+00:00",
        git_sha="abc123",
    )

    assert json.loads(output.read_text(encoding="utf-8")) == manifest
    assert output.read_text(encoding="utf-8").endswith("\n")


def test_build_benchmark_manifest_rejects_missing_report_list() -> None:
    """At least one benchmark evidence report is required."""
    with pytest.raises(ValueError, match="At least one"):
        build_benchmark_manifest(report_paths=[])


def test_build_benchmark_manifest_rejects_non_object_report(tmp_path: Path) -> None:
    """Report files must contain JSON objects."""
    report_path = tmp_path / "report.json"
    _write_json(report_path, [])

    with pytest.raises(ValueError, match="JSON object"):
        build_benchmark_manifest(report_paths=[report_path], repo_root=tmp_path)


def test_build_benchmark_manifest_rejects_missing_score(tmp_path: Path) -> None:
    """Report files must contain score objects."""
    report_path = tmp_path / "report.json"
    _write_json(report_path, {"benchmark": "longmemeval", "score": []})

    with pytest.raises(ValueError, match="score"):
        build_benchmark_manifest(report_paths=[report_path], repo_root=tmp_path)


def test_build_benchmark_manifest_rejects_missing_extra_artifact(tmp_path: Path) -> None:
    """Explicit artefact paths must exist so hashes cannot be fabricated."""
    report_path = tmp_path / "report.json"
    _report(report_path, source_path="")

    with pytest.raises(FileNotFoundError):
        build_benchmark_manifest(
            report_paths=[report_path],
            artifact_paths=[tmp_path / "missing.json"],
            repo_root=tmp_path,
        )


def test_build_benchmark_manifest_rejects_missing_benchmark(tmp_path: Path) -> None:
    """Report files must identify the benchmark they summarise."""
    report_path = tmp_path / "report.json"
    _write_json(report_path, {"benchmark": "", "score": {"accuracy": 1.0}})

    with pytest.raises(ValueError, match="benchmark"):
        build_benchmark_manifest(report_paths=[report_path], repo_root=tmp_path)


def test_build_benchmark_manifest_handles_malformed_optional_fields(
    tmp_path: Path,
) -> None:
    """Malformed optional runtime/token/judge fields are ignored, not invented."""
    report_path = tmp_path / "report.json"
    _write_json(
        report_path,
        {
            "benchmark": "longmemeval",
            "source_path": "",
            "source_format": 7,
            "n_records": 2.5,
            "score": {"correct": True, "total": 2.5, "accuracy": "unknown"},
            "runtime": [],
            "tokens": [],
            "judge": [],
        },
    )

    manifest = build_benchmark_manifest(
        report_paths=[report_path],
        repo_root=tmp_path,
        git_sha="abc123",
    )

    report = _map(_list(manifest["reports"])[0])
    assert report["source_format"] == ""
    assert report["n_records"] is None
    assert report["score"] == {"correct": None, "total": None, "accuracy": None}
    assert report["runtime"] == {}
    assert report["tokens"] == {}
    assert report["judge"] == {}


def test_build_benchmark_manifest_handles_bool_and_integral_float_metrics(
    tmp_path: Path,
) -> None:
    """Bool metrics are ignored while integral floats remain valid counters."""
    report_path = tmp_path / "report.json"
    _write_json(
        report_path,
        {
            "benchmark": "longmemeval",
            "source_path": "",
            "n_records": 2.0,
            "score": {"correct": 1.0, "total": True, "accuracy": False},
        },
    )

    manifest = build_benchmark_manifest(
        report_paths=[report_path],
        repo_root=tmp_path,
        git_sha="abc123",
    )

    report = _map(_list(manifest["reports"])[0])
    assert report["n_records"] == 2
    assert report["score"] == {"correct": 1, "total": None, "accuracy": None}


def test_build_benchmark_manifest_records_absolute_artifact_paths(
    tmp_path: Path,
) -> None:
    """Artefacts outside repo_root retain explicit absolute paths."""
    report_path = tmp_path / "report.json"
    outside = tmp_path.parent / f"{tmp_path.name}_outside.json"
    _report(report_path, source_path="")
    outside.write_text('{"outside":true}\n', encoding="utf-8")
    try:
        manifest = build_benchmark_manifest(
            report_paths=[report_path],
            artifact_paths=[outside],
            repo_root=tmp_path,
            git_sha="abc123",
        )
    finally:
        outside.unlink(missing_ok=True)

    artifact = _map(_list(manifest["artifacts"])[0])
    assert artifact["path"] == str(outside.resolve())


def test_build_benchmark_manifest_uses_empty_git_head_outside_git(tmp_path: Path) -> None:
    """A non-Git repository root records an empty head instead of failing."""
    report_path = tmp_path / "report.json"
    _report(report_path, source_path="")

    manifest = build_benchmark_manifest(report_paths=[report_path], repo_root=tmp_path)

    assert _map(manifest["git"])["head"] == ""


def test_build_benchmark_manifest_records_git_head_from_metadata(tmp_path: Path) -> None:
    """When Git metadata is available, the manifest records the selected HEAD SHA."""
    report_path = tmp_path / "report.json"
    _report(report_path, source_path="")
    git_dir = tmp_path / ".git"
    ref = git_dir / "refs" / "heads" / "main"
    ref.parent.mkdir(parents=True)
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    ref.write_text("abc123\n", encoding="utf-8")

    manifest = build_benchmark_manifest(report_paths=[report_path], repo_root=tmp_path)

    assert _map(manifest["git"])["head"] == "abc123"


def test_build_benchmark_manifest_reads_detached_git_head(tmp_path: Path) -> None:
    """Detached HEAD metadata is recorded directly."""
    report_path = tmp_path / "report.json"
    _report(report_path, source_path="")
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "HEAD").write_text("def456\n", encoding="utf-8")

    manifest = build_benchmark_manifest(report_paths=[report_path], repo_root=tmp_path)

    assert _map(manifest["git"])["head"] == "def456"


def test_build_benchmark_manifest_reads_gitfile_worktree_head(tmp_path: Path) -> None:
    """A Git worktree-style .git file resolves to its metadata directory."""
    report_path = tmp_path / "report.json"
    actual_git_dir = tmp_path / "metadata" / "git"
    _report(report_path, source_path="")
    (tmp_path / ".git").write_text("gitdir: metadata/git\n", encoding="utf-8")
    ref = actual_git_dir / "refs" / "heads" / "main"
    ref.parent.mkdir(parents=True)
    (actual_git_dir / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    ref.write_text("fedcba\n", encoding="utf-8")

    manifest = build_benchmark_manifest(report_paths=[report_path], repo_root=tmp_path)

    assert _map(manifest["git"])["head"] == "fedcba"


def test_build_benchmark_manifest_ignores_invalid_gitfile(tmp_path: Path) -> None:
    """Malformed .git files produce an empty head."""
    report_path = tmp_path / "report.json"
    _report(report_path, source_path="")
    (tmp_path / ".git").write_text("not a gitdir file\n", encoding="utf-8")

    manifest = build_benchmark_manifest(report_paths=[report_path], repo_root=tmp_path)

    assert _map(manifest["git"])["head"] == ""


def test_build_benchmark_manifest_ignores_empty_git_head(tmp_path: Path) -> None:
    """Empty or missing HEAD metadata produces an empty head."""
    report_path = tmp_path / "report.json"
    _report(report_path, source_path="")
    (tmp_path / ".git").mkdir()

    manifest = build_benchmark_manifest(report_paths=[report_path], repo_root=tmp_path)

    assert _map(manifest["git"])["head"] == ""


def test_main_writes_manifest(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The CLI writes a benchmark manifest through the public entrypoint."""
    report_path = tmp_path / "report.json"
    output = tmp_path / "manifest.json"
    _report(report_path, source_path="")

    assert (
        main(
            [
                "--report",
                str(report_path),
                "--output",
                str(output),
                "--generated-at",
                "2026-06-27T13:00:00+00:00",
                "--git-sha",
                "abc123",
                "--command",
                "remanentia-benchmark-report report.json --benchmark longmemeval",
            ]
        )
        == 0
    )

    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert persisted["generated_at"] == "2026-06-27T13:00:00+00:00"
    assert persisted["commands"] == [
        "remanentia-benchmark-report report.json --benchmark longmemeval"
    ]
    assert "wrote benchmark manifest" in capsys.readouterr().out


def _seeded_report(
    path: Path,
    *,
    benchmark: str = "longmemeval",
    seeds: list[int] | None = None,
    accuracy: float = 50.0,
) -> None:
    """Create a manifest-compatible evidence report carrying seed metadata."""
    _write_json(
        path,
        {
            "schema_version": 1,
            "benchmark": benchmark,
            "source_path": "data/results.jsonl",
            "source_format": "jsonl",
            "generated_at": "2026-07-13T02:00:00+00:00",
            "n_records": 2,
            "score": {"correct": 1, "total": 2, "accuracy": accuracy},
            "runtime": {},
            "tokens": {},
            "judge": {},
            "seeds": seeds if seeds is not None else [],
        },
    )


def test_headline_eligible_with_two_distinct_seeds_and_variance_band(tmp_path: Path) -> None:
    """Two runs with distinct seeds make a headline; accuracy is a band, not a point."""
    a = tmp_path / "run_a.json"
    b = tmp_path / "run_b.json"
    _seeded_report(a, seeds=[17], accuracy=50.0)
    _seeded_report(b, seeds=[43], accuracy=54.0)

    manifest = build_benchmark_manifest(report_paths=[a, b], repo_root=tmp_path)
    (headline,) = manifest["headlines"]

    assert headline["benchmark"] == "longmemeval"
    assert headline["runs"] == 2
    assert headline["seeds"] == [17, 43]
    assert headline["headline_eligible"] is True
    assert headline["reason"] == "ok"
    assert headline["variance_band"] == [50.0, 54.0]
    assert headline["accuracy"]["n"] == 2
    assert headline["accuracy"]["mean"] == 52.0
    assert headline["accuracy"]["stdev"] == round(2.8284, 4)


def test_headline_single_seed_and_no_seed_reasons(tmp_path: Path) -> None:
    """One distinct seed (even across runs) or no seed metadata blocks the headline."""
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    c = tmp_path / "c.json"
    _seeded_report(a, benchmark="locomo", seeds=[7], accuracy=80.0)
    _seeded_report(b, benchmark="locomo", seeds=[7], accuracy=82.0)
    _seeded_report(c, benchmark="longmemeval", seeds=[], accuracy=56.6)

    manifest = build_benchmark_manifest(report_paths=[a, b, c], repo_root=tmp_path)
    by_name = {h["benchmark"]: h for h in manifest["headlines"]}

    locomo = by_name["locomo"]
    assert locomo["headline_eligible"] is False
    assert locomo["reason"] == "single_seed"
    assert locomo["seeds"] == [7]
    lme = by_name["longmemeval"]
    assert lme["headline_eligible"] is False
    assert lme["reason"] == "no_seed_metadata"
    assert lme["accuracy"]["stdev"] is None
    assert lme["variance_band"] == [56.6, 56.6]


def test_headline_without_accuracies_has_null_band(tmp_path: Path) -> None:
    """Runs whose score carries no accuracy yield a null band, not a fabricated one."""
    a = tmp_path / "a.json"
    _write_json(
        a,
        {
            "schema_version": 1,
            "benchmark": "longmemeval",
            "source_path": "data/results.jsonl",
            "source_format": "jsonl",
            "generated_at": "2026-07-13T02:00:00+00:00",
            "n_records": 0,
            "score": {"correct": None, "total": None, "accuracy": None},
            "seeds": [1, 2],
        },
    )

    manifest = build_benchmark_manifest(report_paths=[a], repo_root=tmp_path)
    (headline,) = manifest["headlines"]

    assert headline["headline_eligible"] is True
    assert headline["variance_band"] is None
    assert headline["accuracy"]["n"] == 0
    assert headline["accuracy"]["mean"] is None


def test_seed_list_ignores_malformed_values(tmp_path: Path) -> None:
    """Non-integral seed entries are dropped, never coerced or fabricated."""
    a = tmp_path / "a.json"
    _write_json(
        a,
        {
            "schema_version": 1,
            "benchmark": "longmemeval",
            "source_path": "data/results.jsonl",
            "source_format": "jsonl",
            "generated_at": "2026-07-13T02:00:00+00:00",
            "n_records": 1,
            "score": {"correct": 1, "total": 1, "accuracy": 100.0},
            "seeds": [3, "x", True, 3.0, 2.5, None],
        },
    )

    manifest = build_benchmark_manifest(report_paths=[a], repo_root=tmp_path)

    assert manifest["reports"][0]["seeds"] == [3]
    assert manifest["headlines"][0]["seeds"] == [3]


def test_main_require_multi_seed_gate(tmp_path: Path, capsys: object) -> None:
    """--require-multi-seed exits 1 on single-seed benchmarks and 0 when eligible."""
    import benchmark_manifest as module

    single = tmp_path / "single.json"
    _seeded_report(single, seeds=[7])
    out = tmp_path / "manifest.json"

    code = module.main(
        [
            "--report",
            str(single),
            "--output",
            str(out),
            "--repo-root",
            str(tmp_path),
            "--require-multi-seed",
        ]
    )
    assert code == 1

    second = tmp_path / "second.json"
    _seeded_report(second, seeds=[8], accuracy=52.0)
    code = module.main(
        [
            "--report",
            str(single),
            "--report",
            str(second),
            "--output",
            str(out),
            "--repo-root",
            str(tmp_path),
            "--require-multi-seed",
        ]
    )
    assert code == 0


def test_headlines_never_pool_across_settings(tmp_path: Path) -> None:
    """Oracle and full-S runs of one benchmark form separate headline groups."""
    oracle = tmp_path / "oracle.json"
    full_a = tmp_path / "full_a.json"
    full_b = tmp_path / "full_b.json"
    _write_json(
        oracle,
        {
            "schema_version": 1,
            "benchmark": "longmemeval",
            "source_path": "data/results.jsonl",
            "source_format": "jsonl",
            "generated_at": "t",
            "n_records": 2,
            "score": {"correct": 1, "total": 2, "accuracy": 72.2},
            "seeds": [],
            "settings": ["oracle"],
        },
    )
    for path, seed, accuracy in ((full_a, 42, 56.8), (full_b, 1337, 55.4)):
        _write_json(
            path,
            {
                "schema_version": 1,
                "benchmark": "longmemeval",
                "source_path": "data/results.jsonl",
                "source_format": "jsonl",
                "generated_at": "t",
                "n_records": 2,
                "score": {"correct": 1, "total": 2, "accuracy": accuracy},
                "seeds": [seed],
                "settings": ["full_s"],
            },
        )

    manifest = build_benchmark_manifest(report_paths=[oracle, full_a, full_b], repo_root=tmp_path)
    by_key = {(h["benchmark"], h["setting"]): h for h in manifest["headlines"]}

    full = by_key[("longmemeval", "full_s")]
    assert full["headline_eligible"] is True
    assert full["seeds"] == [42, 1337]
    assert full["variance_band"] == [55.4, 56.8]
    oracle_headline = by_key[("longmemeval", "oracle")]
    assert oracle_headline["headline_eligible"] is False
    assert oracle_headline["reason"] == "no_seed_metadata"


def test_mixed_setting_report_is_ineligible(tmp_path: Path) -> None:
    """A single results file carrying both settings is an integrity fault."""
    mixed = tmp_path / "mixed.json"
    _write_json(
        mixed,
        {
            "schema_version": 1,
            "benchmark": "longmemeval",
            "source_path": "data/results.jsonl",
            "source_format": "jsonl",
            "generated_at": "t",
            "n_records": 2,
            "score": {"correct": 1, "total": 2, "accuracy": 60.0},
            "seeds": [1, 2],
            "settings": ["full_s", "oracle"],
        },
    )

    manifest = build_benchmark_manifest(report_paths=[mixed], repo_root=tmp_path)
    (headline,) = manifest["headlines"]

    assert headline["setting"] == "mixed"
    assert headline["headline_eligible"] is False
    assert headline["reason"] == "mixed_settings"
