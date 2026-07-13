# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Temporal SNN reference-engine tests

"""Real public-step tests for LIF, online STDP and state invariants."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from snn_memory.contracts import ModelConfig
from snn_memory.reference import run_episode, step_network
from snn_memory.state import initialise_state, initialise_weights, validate_weights


def _pair_network() -> tuple[ModelConfig, np.ndarray, np.ndarray]:
    config = ModelConfig(n_neurons=4, excitatory_fraction=0.75, connectivity=1.0)
    weights: npt.NDArray[np.float64] = np.full((4, 4), 0.1, dtype=np.float64)
    weights[3] = -0.1
    np.fill_diagonal(weights, 0.0)
    topology: npt.NDArray[np.bool_] = ~np.eye(4, dtype=np.bool_)
    return config, weights, topology


def test_causal_pair_potentiates_row_pre_column_post_synapse() -> None:
    config, weights, topology = _pair_network()
    state = initialise_state(config)
    state.voltage[0] = config.v_threshold_mv + 1.0
    state, weights, _ = step_network(
        state, weights, topology, np.zeros(4), config, plasticity_enabled=True
    )
    before = weights[0, 1]
    state.voltage[1] = config.v_threshold_mv + 1.0
    _, weights, _ = step_network(
        state, weights, topology, np.zeros(4), config, plasticity_enabled=True
    )
    expected = before + config.a_plus * np.exp(-config.dt_ms / config.tau_plus_ms)
    assert weights[0, 1] == expected


def test_reverse_pair_depresses_row_pre_column_post_synapse() -> None:
    config, weights, topology = _pair_network()
    state = initialise_state(config)
    state.voltage[1] = config.v_threshold_mv + 1.0
    state, weights, _ = step_network(
        state, weights, topology, np.zeros(4), config, plasticity_enabled=True
    )
    before = weights[0, 1]
    state.voltage[0] = config.v_threshold_mv + 1.0
    _, weights, _ = step_network(
        state, weights, topology, np.zeros(4), config, plasticity_enabled=True
    )
    expected = before - config.a_minus * np.exp(-config.dt_ms / config.tau_minus_ms)
    assert weights[0, 1] == expected


def test_simultaneous_first_spikes_leave_weights_unchanged() -> None:
    config, weights, topology = _pair_network()
    state = initialise_state(config)
    state.voltage[:2] = config.v_threshold_mv + 1.0
    _, updated, _ = step_network(
        state, weights, topology, np.zeros(4), config, plasticity_enabled=True
    )
    np.testing.assert_array_equal(updated, weights)


def test_refractory_state_blocks_immediate_respike() -> None:
    config, weights, topology = _pair_network()
    currents = np.full((3, 4), 100.0)
    result = run_episode(
        initialise_state(config), weights, topology, currents, config, plasticity_enabled=False
    )
    assert result.spikes[0].all()
    assert not result.spikes[1].any()
    assert not result.spikes[2].any()


@settings(max_examples=30, deadline=None)  # type: ignore[untyped-decorator] # Hypothesis.
@given(  # type: ignore[untyped-decorator] # Hypothesis decorators are untyped.
    seed=st.integers(min_value=0, max_value=2**32 - 1), steps=st.integers(1, 12)
)
def test_public_episode_preserves_dale_topology_and_bounds(seed: int, steps: int) -> None:
    config = ModelConfig(n_neurons=12, connectivity=0.3)
    weights, topology = initialise_weights(config, seed)
    rng = np.random.default_rng(seed)
    currents = rng.uniform(0.0, 25.0, size=(steps, config.n_neurons))
    result = run_episode(
        initialise_state(config), weights, topology, currents, config, plasticity_enabled=True
    )
    validate_weights(result.final_weights, topology, config)
    assert np.all(result.final_weights[: config.n_excitatory] <= config.weight_max)


def test_step_network_rejects_malformed_input_current() -> None:
    config, weights, topology = _pair_network()
    state = initialise_state(config)
    with pytest.raises(ValueError, match="one finite value per neuron"):
        step_network(state, weights, topology, np.zeros(3), config, plasticity_enabled=False)


def test_run_episode_rejects_wrong_current_shape() -> None:
    config, weights, topology = _pair_network()
    with pytest.raises(ValueError, match="timesteps, n_neurons"):
        run_episode(
            initialise_state(config), weights, topology, np.zeros(4), config, plasticity_enabled=False
        )
