# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Installed PyO3 full-episode parity test

"""Compare the installed Rust extension with the public Python oracle."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Protocol, cast

import numpy as np
import numpy.typing as npt

from snn_memory.contracts import ModelConfig
from snn_memory.reference import run_episode
from snn_memory.state import initialise_state


class _RustState(Protocol):
    """State fields read through the installed extension boundary."""

    spikes: list[bool]
    pre_trace: list[float]
    post_trace: list[float]
    refractory_steps: list[int]
    voltage_mv: list[float]


class _RustWeights(Protocol):
    """Weight fields read through the installed extension boundary."""

    values: list[float]
    topology: list[bool]


class _RustEpisode(Protocol):
    """Episode fields returned by the installed extension."""

    spike_indices: list[list[int]]
    voltage_mv: list[list[float]]
    refractory_steps: list[list[int]]
    pre_trace: list[list[float]]
    post_trace: list[list[float]]
    recurrent_current: list[list[float]]
    final_state: _RustState
    final_weights: _RustWeights


class _RustModule(Protocol):
    """Typed subset of the installed PyO3 module used by this test."""

    ModelConfig: Callable[..., object]
    NetworkState: Callable[..., _RustState]
    WeightMatrix: Callable[..., _RustWeights]
    run_episode: Callable[..., _RustEpisode]


class _RustConfig(Protocol):
    """Configuration fields exposed by the installed extension."""

    excitatory_weight_max: float


rust_snn_memory = cast(_RustModule, importlib.import_module("rust_snn_memory"))


def test_installed_extension_default_matches_python_weight_maximum() -> None:
    """Keep the public PyO3 default aligned with the Python contract."""
    rust_config = cast(_RustConfig, rust_snn_memory.ModelConfig(3))
    assert rust_config.excitatory_weight_max == ModelConfig().weight_max


def _spike_matrix(
    spike_indices: list[list[int]], timesteps: int, neurons: int
) -> npt.NDArray[np.bool_]:
    """Convert sparse extension output to the oracle's boolean raster."""
    spikes: npt.NDArray[np.bool_] = np.zeros((timesteps, neurons), dtype=np.bool_)
    for step, indices in enumerate(spike_indices):
        spikes[step, indices] = True
    return spikes


def test_installed_extension_matches_python_for_complete_plastic_episode() -> None:
    """Match spikes, voltage, traces, refractory state, and weights per episode."""
    config = ModelConfig(
        n_neurons=4,
        excitatory_fraction=0.75,
        dt_ms=1.0,
        tau_m_ms=5.0,
        v_rest_mv=-65.0,
        v_reset_mv=-70.0,
        v_threshold_mv=-64.0,
        refractory_ms=2.0,
        tau_plus_ms=20.0,
        tau_minus_ms=20.0,
        a_plus=0.005,
        a_minus=0.006,
        weight_max=1.0,
        connectivity=0.5,
    )
    weights = np.asarray(
        [
            [0.0, 0.2, 0.0, 0.0],
            [0.1, 0.0, 0.15, 0.0],
            [0.0, 0.1, 0.0, 0.2],
            [-0.1, 0.0, -0.2, 0.0],
        ],
        dtype=np.float64,
    )
    topology = weights != 0.0
    topology[0, 2] = True
    currents = np.asarray(
        [
            [2.0, 0.0, 0.0, 2.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 2.0],
            [2.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    python_result = run_episode(
        initialise_state(config),
        weights,
        topology,
        currents,
        config,
        plasticity_enabled=True,
    )

    rust_config = rust_snn_memory.ModelConfig(
        config.n_excitatory,
        dt_ms=config.dt_ms,
        tau_m_ms=config.tau_m_ms,
        v_rest_mv=config.v_rest_mv,
        v_reset_mv=config.v_reset_mv,
        v_threshold_mv=config.v_threshold_mv,
        refractory_ms=config.refractory_ms,
        tau_plus_ms=config.tau_plus_ms,
        tau_minus_ms=config.tau_minus_ms,
        a_plus=config.a_plus,
        a_minus=config.a_minus,
        excitatory_weight_max=config.weight_max,
    )
    rust_state = rust_snn_memory.NetworkState(config.n_neurons, rust_config)
    rust_weights = rust_snn_memory.WeightMatrix(
        config.n_neurons,
        config.n_excitatory,
        weights.ravel().tolist(),
        topology.ravel().tolist(),
    )
    original_rust_voltage = list(rust_state.voltage_mv)
    original_rust_weights = list(rust_weights.values)
    original_rust_topology = list(rust_weights.topology)
    rust_result = rust_snn_memory.run_episode(
        rust_state,
        rust_weights,
        currents.tolist(),
        True,
        rust_config,
    )
    assert rust_state.voltage_mv == original_rust_voltage
    assert rust_weights.values == original_rust_weights
    assert rust_weights.topology == original_rust_topology

    np.testing.assert_array_equal(
        _spike_matrix(rust_result.spike_indices, currents.shape[0], config.n_neurons),
        python_result.spikes,
    )
    np.testing.assert_array_equal(np.asarray(rust_result.voltage_mv), python_result.voltages)
    np.testing.assert_array_equal(
        np.asarray(rust_result.refractory_steps), python_result.refractory
    )
    np.testing.assert_array_equal(np.asarray(rust_result.pre_trace), python_result.pre_traces)
    np.testing.assert_array_equal(np.asarray(rust_result.post_trace), python_result.post_traces)
    np.testing.assert_array_equal(
        np.asarray(rust_result.recurrent_current), python_result.recurrent_current
    )
    np.testing.assert_array_equal(
        np.asarray(rust_result.final_state.pre_trace),
        python_result.final_state.pre_trace,
    )
    np.testing.assert_array_equal(
        np.asarray(rust_result.final_state.post_trace),
        python_result.final_state.post_trace,
    )
    np.testing.assert_array_equal(
        np.asarray(rust_result.final_state.refractory_steps),
        python_result.final_state.refractory,
    )
    np.testing.assert_array_equal(
        np.asarray(rust_result.final_weights.values).reshape(weights.shape),
        python_result.final_weights,
    )
    np.testing.assert_array_equal(
        np.asarray(rust_result.final_weights.topology).reshape(topology.shape),
        topology,
    )
    assert python_result.final_weights[0, 2] > 0.0
