# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — tests for the index-freshness gate

"""Tests for :mod:`index_freshness`.

Modification times are set explicitly with ``os.utime`` so drift is exact and
deterministic — the suite asserts the gate's verdict on a corpus shaped like
the real 2026-05-12 stall (a fresh firehose, a frozen index) without depending
on wall-clock timing.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from index_freshness import (
    DEFAULT_MAX_DRIFT_DAYS,
    SECONDS_PER_DAY,
    PipelineFreshness,
    StageFreshness,
    _iso,
    assess_default,
    assess_pipeline,
    default_stages,
    probe_stage,
)


def _touch(path: Path, mtime: float) -> Path:
    """Create *path* (and parents) and pin its mtime to *mtime* epoch seconds."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x", encoding="utf-8")
    os.utime(path, (mtime, mtime))
    return path


def _stage(name: str, newest: float | None, count: int = 1) -> StageFreshness:
    return StageFreshness(name=name, newest=newest, count=count, root=f"/{name}")


class TestProbeStage:
    def test_missing_directory_is_empty(self, tmp_path: Path):
        st = probe_stage("x", tmp_path / "nope", ["*.json"])
        assert st.newest is None
        assert st.count == 0
        assert st.present is False

    def test_reports_newest_and_count(self, tmp_path: Path):
        _touch(tmp_path / "a.json", 1000.0)
        _touch(tmp_path / "b.json", 3000.0)
        _touch(tmp_path / "c.json", 2000.0)
        st = probe_stage("stimuli", tmp_path, ["*.json"])
        assert st.count == 3
        assert st.newest == 3000.0
        assert st.present is True

    def test_multiple_patterns_union(self, tmp_path: Path):
        _touch(tmp_path / "v.npz", 5000.0)
        _touch(tmp_path / "c.sqlite", 4000.0)
        st = probe_stage("index", tmp_path, ["*.npz", "*.sqlite"])
        assert st.count == 2
        assert st.newest == 5000.0

    def test_recursive_pattern(self, tmp_path: Path):
        _touch(tmp_path / "sub" / "deep.md", 7000.0)
        st = probe_stage("findings", tmp_path, ["**/*.md"])
        assert st.count == 1
        assert st.newest == 7000.0

    def test_directories_matching_glob_are_skipped(self, tmp_path: Path):
        # A directory whose name matches the glob must not count as an artifact.
        (tmp_path / "sub.json").mkdir()
        _touch(tmp_path / "real.json", 9000.0)
        st = probe_stage("stimuli", tmp_path, ["*.json"])
        assert st.count == 1
        assert st.newest == 9000.0


class TestDriftAndVerdict:
    def test_fresh_pipeline(self):
        stages = [_stage("stimuli", 1000.0), _stage("index", 1000.0 - 3 * SECONDS_PER_DAY)]
        report = assess_pipeline(stages, max_drift_days=7.0)
        assert report.stale is False
        assert report.drift_days == pytest.approx(3.0)

    def test_stale_when_drift_exceeds_tolerance(self):
        stages = [_stage("stimuli", 1000.0), _stage("index", 1000.0 - 56 * SECONDS_PER_DAY)]
        report = assess_pipeline(stages, max_drift_days=7.0)
        assert report.stale is True
        assert report.drift_days == pytest.approx(56.0)

    def test_index_fresher_than_source_is_not_stale(self):
        stages = [_stage("stimuli", 1000.0), _stage("index", 2000.0)]
        report = assess_pipeline(stages, max_drift_days=7.0)
        assert report.drift_seconds == pytest.approx(-1000.0)
        assert report.stale is False

    def test_source_with_no_sink_is_stale(self):
        stages = [_stage("stimuli", 1000.0), _stage("index", None, count=0)]
        report = assess_pipeline(stages, max_drift_days=7.0)
        # The index was never built — infinite lag the drift figure can't show.
        assert report.sink is report.source
        assert report.drift_seconds is None
        assert report.stale is True

    def test_empty_pipeline_is_not_stale(self):
        stages = [_stage("stimuli", None, count=0), _stage("index", None, count=0)]
        report = assess_pipeline(stages, max_drift_days=7.0)
        assert report.source is None
        assert report.sink is None
        assert report.drift_seconds is None
        assert report.stale is False

    def test_source_skips_empty_leading_stages(self):
        stages = [
            _stage("stimuli", None, count=0),
            _stage("findings", 5000.0),
            _stage("index", 1000.0),
        ]
        report = assess_pipeline(stages, max_drift_days=7.0)
        assert report.source.name == "findings"
        assert report.sink.name == "index"
        assert report.drift_seconds == pytest.approx(4000.0)


class TestReportRendering:
    def test_summary_marks_stale_and_absent_stages(self):
        stages = [_stage("stimuli", 1000.0, count=4), _stage("index", None, count=0)]
        report = assess_pipeline(stages, max_drift_days=7.0)
        text = report.summary()
        assert "stimuli" in text
        assert "no artifact" in text
        assert "STALE" in text

    def test_summary_marks_fresh(self):
        stages = [_stage("stimuli", 1000.0), _stage("index", 1000.0)]
        text = assess_pipeline(stages, max_drift_days=7.0).summary()
        assert "fresh" in text
        assert "n/a" not in text  # both present → a real drift figure

    def test_summary_drift_na_when_single_stage(self):
        stages = [_stage("stimuli", 1000.0)]
        text = assess_pipeline(stages, max_drift_days=7.0).summary()
        assert "n/a" in text

    def test_as_dict_round_trips_fields(self):
        stages = [_stage("stimuli", 1000.0, count=2), _stage("index", 1000.0 - SECONDS_PER_DAY)]
        d = assess_pipeline(stages, max_drift_days=7.0).as_dict()
        assert d["stale"] is False
        assert d["drift_days"] == pytest.approx(1.0)
        assert d["stages"][0]["name"] == "stimuli"
        assert d["stages"][0]["count"] == 2
        assert d["stages"][1]["newest_iso"] != "—"


class TestIso:
    def test_none_is_dash(self):
        assert _iso(None) == "—"

    def test_epoch_formats_utc(self):
        # 2026-05-12T00:00:00Z = 1778544000 (verified via datetime.timestamp)
        assert _iso(1778544000.0) == "2026-05-12T00:00:00Z"


class TestDefaultStages:
    def test_probes_canonical_chain(self, tmp_path: Path, monkeypatch):
        stimuli = tmp_path / "stim"
        _touch(stimuli / "s.json", 6000.0)
        _touch(tmp_path / "memory" / "semantic" / "findings" / "f.md", 5000.0)
        _touch(tmp_path / "memory" / "digests" / "d.md", 4000.0)
        _touch(tmp_path / "snn_state" / "vector_index" / "v.npz", 3000.0)
        monkeypatch.setenv("REMANENTIA_BASE", str(tmp_path))
        monkeypatch.setenv("REMANENTIA_STIMULI_DIR", str(stimuli))

        stages = default_stages()
        names = [s.name for s in stages]
        assert names == ["stimuli", "findings", "digests", "vector-index"]
        assert stages[0].newest == 6000.0
        assert stages[-1].newest == 3000.0

    def test_default_stimuli_dir_falls_back_to_base(self, tmp_path: Path, monkeypatch):
        _touch(tmp_path / "snn_stimuli" / "s.json", 6000.0)
        monkeypatch.setenv("REMANENTIA_BASE", str(tmp_path))
        monkeypatch.delenv("REMANENTIA_STIMULI_DIR", raising=False)
        stages = default_stages()
        assert stages[0].name == "stimuli"
        assert stages[0].newest == 6000.0

    def test_base_argument_overrides_env(self, tmp_path: Path, monkeypatch):
        _touch(tmp_path / "snn_stimuli" / "s.json", 1234.0)
        monkeypatch.delenv("REMANENTIA_BASE", raising=False)
        monkeypatch.delenv("REMANENTIA_STIMULI_DIR", raising=False)
        stages = default_stages(base=tmp_path)
        assert stages[0].newest == 1234.0

    def test_assess_default_reports_stall(self, tmp_path: Path, monkeypatch):
        # Shape the real stall: fresh stimuli, an 8-week-old index.
        _touch(tmp_path / "snn_stimuli" / "s.json", 100 * SECONDS_PER_DAY)
        _touch(tmp_path / "snn_state" / "vector_index" / "v.npz", 44 * SECONDS_PER_DAY)
        monkeypatch.setenv("REMANENTIA_BASE", str(tmp_path))
        monkeypatch.delenv("REMANENTIA_STIMULI_DIR", raising=False)
        report = assess_default(max_drift_days=DEFAULT_MAX_DRIFT_DAYS)
        assert report.stale is True
        assert report.drift_days == pytest.approx(56.0)


def test_stage_freshness_is_frozen():
    st = _stage("x", 1.0)
    with pytest.raises(AttributeError):
        st.newest = 2.0  # type: ignore[misc]  # frozen dataclass — testing immutability


def test_pipeline_freshness_is_frozen():
    report = PipelineFreshness(stages=(), max_drift_seconds=1.0)
    with pytest.raises(AttributeError):
        report.max_drift_seconds = 2.0  # type: ignore[misc]  # frozen dataclass
