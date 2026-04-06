# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Tests for SNN state serialisation (pickle → npz migration)

"""Tests for numpy.savez serialisation in snn_backend and snn_daemon.

Covers:
- Round-trip save/load for all three network classes
- Legacy pickle backward compatibility
- Data integrity (arrays, scalars, dtypes)
- Double round-trip stability
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from snn_backend import DenseCPULIFNetwork, _load_state
from snn_daemon import SimpleLIFNetwork, _load_daemon_state


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture()
def cpu_net() -> DenseCPULIFNetwork:
    """Small DenseCPU network for testing."""
    return DenseCPULIFNetwork(n=50, dt=0.001, seed=42)


@pytest.fixture()
def daemon_net() -> SimpleLIFNetwork:
    """Small SimpleLIF network for testing."""
    return SimpleLIFNetwork(n_neurons=50, dt=0.001)


# ── DenseCPULIFNetwork ──────────────────────────────────────────────


class TestDenseCPUSaveLoad:
    """Round-trip tests for DenseCPULIFNetwork npz serialisation."""

    def test_round_trip(self, cpu_net: DenseCPULIFNetwork, tmp_path: Path) -> None:
        path = tmp_path / "dense_cpu.npz"
        cpu_net.save(path)
        loaded = DenseCPULIFNetwork.load(path)

        np.testing.assert_array_equal(loaded.v, cpu_net.v)
        np.testing.assert_array_equal(loaded.w, cpu_net.w)
        np.testing.assert_array_equal(loaded.mask, cpu_net.mask)
        np.testing.assert_array_equal(loaded.last_spike, cpu_net.last_spike)
        assert loaded.t == cpu_net.t
        assert loaded.n == cpu_net.n
        assert loaded.dt == cpu_net.dt

    def test_saved_file_is_npz(self, cpu_net: DenseCPULIFNetwork, tmp_path: Path) -> None:
        path = tmp_path / "net"
        cpu_net.save(path)
        # numpy.savez appends .npz
        assert (tmp_path / "net.npz").exists()

    def test_double_round_trip(self, cpu_net: DenseCPULIFNetwork, tmp_path: Path) -> None:
        p1 = tmp_path / "r1.npz"
        p2 = tmp_path / "r2.npz"
        cpu_net.save(p1)
        net2 = DenseCPULIFNetwork.load(p1)
        net2.save(p2)
        net3 = DenseCPULIFNetwork.load(p2)
        np.testing.assert_array_equal(net3.v, cpu_net.v)
        np.testing.assert_array_equal(net3.w, cpu_net.w)

    def test_dtypes_preserved(self, cpu_net: DenseCPULIFNetwork, tmp_path: Path) -> None:
        path = tmp_path / "dtypes.npz"
        cpu_net.save(path)
        loaded = DenseCPULIFNetwork.load(path)
        assert loaded.v.dtype == np.float32
        assert loaded.w.dtype == np.float32
        assert loaded.last_spike.dtype == np.float32
        assert isinstance(loaded.t, float)
        assert isinstance(loaded.n, int)

    def test_legacy_pickle_compat(self, cpu_net: DenseCPULIFNetwork, tmp_path: Path) -> None:
        """Verify that load() can read old pickle files."""
        path = tmp_path / "legacy.pkl"
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "v": cpu_net.v,
                    "w": cpu_net.w,
                    "mask": cpu_net.mask,
                    "last_spike": cpu_net.last_spike,
                    "t": cpu_net.t,
                    "n": cpu_net.n,
                    "dt": cpu_net.dt,
                    "backend": "dense_cpu",
                },
                f,
            )
        loaded = DenseCPULIFNetwork.load(path)
        np.testing.assert_array_equal(loaded.v, cpu_net.v)
        assert loaded.n == cpu_net.n

    def test_mask_generated_when_absent(self, cpu_net: DenseCPULIFNetwork, tmp_path: Path) -> None:
        """Verify mask is reconstructed from w when missing (old format)."""
        path = tmp_path / "no_mask.pkl"
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "v": cpu_net.v,
                    "w": cpu_net.w,
                    "last_spike": cpu_net.last_spike,
                    "t": cpu_net.t,
                    "n": cpu_net.n,
                    "dt": cpu_net.dt,
                },
                f,
            )
        loaded = DenseCPULIFNetwork.load(path)
        expected_mask = (cpu_net.w > 0).astype(np.float32)
        np.testing.assert_array_equal(loaded.mask, expected_mask)


# ── SimpleLIFNetwork (daemon) ───────────────────────────────────────


class TestDaemonSaveLoad:
    """Round-trip tests for SimpleLIFNetwork npz serialisation."""

    def test_round_trip(self, daemon_net: SimpleLIFNetwork, tmp_path: Path) -> None:
        path = tmp_path / "daemon.npz"
        daemon_net.save(path)
        loaded = SimpleLIFNetwork.load(path)

        np.testing.assert_array_equal(loaded.v, daemon_net.v)
        np.testing.assert_array_equal(loaded.w, daemon_net.w)
        np.testing.assert_array_equal(loaded.last_spike, daemon_net.last_spike)
        assert loaded.t == daemon_net.t
        assert loaded.n == daemon_net.n
        assert loaded.dt == daemon_net.dt

    def test_double_round_trip(self, daemon_net: SimpleLIFNetwork, tmp_path: Path) -> None:
        p1 = tmp_path / "r1.npz"
        p2 = tmp_path / "r2.npz"
        daemon_net.save(p1)
        net2 = SimpleLIFNetwork.load(p1)
        net2.save(p2)
        net3 = SimpleLIFNetwork.load(p2)
        np.testing.assert_array_equal(net3.v, daemon_net.v)
        np.testing.assert_array_equal(net3.w, daemon_net.w)

    def test_legacy_pickle_compat(self, daemon_net: SimpleLIFNetwork, tmp_path: Path) -> None:
        path = tmp_path / "legacy.pkl"
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "v": daemon_net.v,
                    "w": daemon_net.w,
                    "last_spike": daemon_net.last_spike,
                    "t": daemon_net.t,
                    "n": daemon_net.n,
                    "dt": daemon_net.dt,
                },
                f,
            )
        loaded = SimpleLIFNetwork.load(path)
        np.testing.assert_array_equal(loaded.v, daemon_net.v)
        assert loaded.n == daemon_net.n

    def test_dtypes_preserved(self, daemon_net: SimpleLIFNetwork, tmp_path: Path) -> None:
        path = tmp_path / "dtypes.npz"
        daemon_net.save(path)
        loaded = SimpleLIFNetwork.load(path)
        assert loaded.v.dtype == np.float32
        assert loaded.w.dtype == np.float32
        assert isinstance(loaded.t, float)


# ── _load_state / _load_daemon_state ────────────────────────────────


class TestLoadStateHelper:
    """Tests for the format-detection helpers."""

    def test_load_npz(self, tmp_path: Path) -> None:
        data = {"v": np.ones(10), "t": np.float64(1.5)}
        np.savez_compressed(tmp_path / "test", **data)
        result = _load_state(tmp_path / "test")
        np.testing.assert_array_equal(result["v"], data["v"])

    def test_load_npz_with_suffix(self, tmp_path: Path) -> None:
        data = {"v": np.ones(10)}
        np.savez_compressed(tmp_path / "test.npz", **data)
        result = _load_state(tmp_path / "test.npz")
        np.testing.assert_array_equal(result["v"], data["v"])

    def test_load_legacy_pickle(self, tmp_path: Path) -> None:
        path = tmp_path / "old.bin"
        with open(path, "wb") as f:
            pickle.dump({"v": np.zeros(5), "n": 5}, f)
        result = _load_state(path)
        assert result["n"] == 5

    def test_daemon_load_npz(self, tmp_path: Path) -> None:
        data = {"v": np.ones(8), "n": np.int64(8)}
        np.savez_compressed(tmp_path / "d", **data)
        result = _load_daemon_state(tmp_path / "d")
        assert int(result["n"]) == 8


# ── load_network ────────────────────────────────────────────────────


class TestLoadNetwork:
    """Tests for the top-level load_network function."""

    def test_load_dense_cpu(self, cpu_net: DenseCPULIFNetwork, tmp_path: Path) -> None:
        from snn_backend import load_network

        path = tmp_path / "net"
        cpu_net.save(path)
        loaded = load_network(tmp_path / "net")
        assert isinstance(loaded, DenseCPULIFNetwork)
        np.testing.assert_array_equal(loaded.v, cpu_net.v)

    def test_load_network_legacy_pickle(self, cpu_net: DenseCPULIFNetwork, tmp_path: Path) -> None:
        from snn_backend import load_network

        path = tmp_path / "legacy.pkl"
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "v": cpu_net.v,
                    "w": cpu_net.w,
                    "mask": cpu_net.mask,
                    "last_spike": cpu_net.last_spike,
                    "t": cpu_net.t,
                    "n": cpu_net.n,
                    "dt": cpu_net.dt,
                    "backend": "dense_cpu",
                },
                f,
            )
        loaded = load_network(path)
        assert isinstance(loaded, DenseCPULIFNetwork)
        assert loaded.n == cpu_net.n
