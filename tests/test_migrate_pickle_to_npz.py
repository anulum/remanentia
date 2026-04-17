# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Tests for the legacy-pickle migrator

from __future__ import annotations

import gzip
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parent.parent / "tools"
sys.path.insert(0, str(TOOLS))

import migrate_pickle_to_npz as mig  # noqa: E402 — tools/ path added above


@pytest.fixture
def npz_dict_pickle(tmp_path: Path) -> Path:
    """Write a legacy pickle containing a dict of ndarrays."""
    data = {
        "weights": np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "n_neurons": 42,
        "name": "test",
    }
    path = tmp_path / "legacy_state.pkl"
    with open(path, "wb") as f:
        pickle.dump(data, f)
    return path


@pytest.fixture
def nested_dict_pickle(tmp_path: Path) -> Path:
    """Write a legacy pickle containing a nested dict (json.gz target)."""
    data = {
        "config": {"rate": 0.01, "decay": 0.99},
        "history": [1, 2, 3],
        "notes": "ok",
    }
    path = tmp_path / "legacy_nested.pkl"
    with open(path, "wb") as f:
        pickle.dump(data, f)
    return path


class TestMigrateNpz:
    def test_writes_npz_and_backup(self, npz_dict_pickle):
        rc = mig.main(["--path", str(npz_dict_pickle.parent)])
        assert rc == 0
        npz = npz_dict_pickle.with_suffix(".npz")
        bak = npz_dict_pickle.with_suffix(".pkl.bak")
        assert npz.exists()
        assert bak.exists()
        assert not npz_dict_pickle.exists()

    def test_npz_round_trips_dict(self, npz_dict_pickle):
        mig.main(["--path", str(npz_dict_pickle.parent)])
        npz = npz_dict_pickle.with_suffix(".npz")
        loaded = np.load(npz, allow_pickle=False)
        assert "weights" in loaded
        assert loaded["weights"].shape == (2, 2)
        assert loaded["n_neurons"].item() == 42


class TestMigrateJsonGz:
    def test_writes_json_gz(self, nested_dict_pickle):
        rc = mig.main(["--path", str(nested_dict_pickle.parent)])
        assert rc == 0
        out = nested_dict_pickle.with_suffix(".json.gz")
        assert out.exists()

    def test_json_round_trips(self, nested_dict_pickle):
        mig.main(["--path", str(nested_dict_pickle.parent)])
        out = nested_dict_pickle.with_suffix(".json.gz")
        with gzip.open(out, "rb") as f:
            payload = json.loads(f.read())
        assert payload["config"]["rate"] == 0.01
        assert payload["history"] == [1, 2, 3]
        assert payload["notes"] == "ok"


class TestDryRun:
    def test_dry_run_does_not_write(self, npz_dict_pickle):
        rc = mig.main(["--dry-run", "--path", str(npz_dict_pickle.parent)])
        assert rc == 0
        assert npz_dict_pickle.exists()
        assert not npz_dict_pickle.with_suffix(".npz").exists()
        assert not npz_dict_pickle.with_suffix(".pkl.bak").exists()


class TestRestore:
    def test_restore_renames_bak_to_pkl(self, npz_dict_pickle):
        mig.main(["--path", str(npz_dict_pickle.parent)])
        mig.main(["--restore", "--path", str(npz_dict_pickle.parent)])
        assert npz_dict_pickle.exists()
        assert not npz_dict_pickle.with_suffix(".pkl.bak").exists()


class TestEmptyScan:
    def test_no_files_returns_zero(self, tmp_path, capsys):
        rc = mig.main(["--path", str(tmp_path)])
        assert rc == 0
        assert "no legacy .pkl files" in capsys.readouterr().out


class TestCorruptPickle:
    def test_corrupt_reports_failure(self, tmp_path):
        bad = tmp_path / "broken.pkl"
        bad.write_bytes(b"\x80not a real pickle")
        rc = mig.main(["--path", str(tmp_path)])
        assert rc == 1  # at least one failure


class TestSelectors:
    def test_looks_like_dict_of_arrays(self):
        assert mig._looks_like_dict_of_arrays({"a": np.zeros(3)})
        assert mig._looks_like_dict_of_arrays({"a": [1, 2, 3], "b": "s"})
        assert not mig._looks_like_dict_of_arrays({"a": {"nested": 1}})
        assert not mig._looks_like_dict_of_arrays([1, 2, 3])

    def test_iter_targets_file_vs_dir(self, tmp_path):
        f = tmp_path / "x.pkl"
        f.write_bytes(b"")
        (tmp_path / "sub").mkdir()
        sub = tmp_path / "sub" / "y.pkl"
        sub.write_bytes(b"")
        hits = mig._iter_targets([f], suffix=".pkl")
        assert hits == [f]
        hits = mig._iter_targets([tmp_path], suffix=".pkl")
        assert set(hits) == {f, sub}
