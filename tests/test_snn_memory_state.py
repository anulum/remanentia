# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Temporal SNN state and invariant tests

"""Real state construction, Dale-law and topology-invariant tests."""

from __future__ import annotations

import numpy as np
import pytest

from snn_memory.contracts import ModelConfig
from snn_memory.state import (
    initialise_state,
    initialise_weights,
    plasticity_mask,
    validate_weights,
)


def _config() -> ModelConfig:
    return ModelConfig(n_neurons=5, excitatory_fraction=0.8, connectivity=1.0)


def test_initialise_state_is_deterministic_resting() -> None:
    config = _config()
    state = initialise_state(config)
    assert np.all(state.voltage == config.v_rest_mv)
    assert not state.spikes.any()
    assert not state.refractory.any()


def test_state_copy_is_independent() -> None:
    state = initialise_state(_config())
    clone = state.copy()
    clone.voltage[0] = 1.0
    clone.spikes[0] = True
    assert state.voltage[0] == _config().v_rest_mv
    assert not state.spikes[0]


def test_initialise_weights_obey_dale_topology_and_diagonal() -> None:
    config = _config()
    weights, topology = initialise_weights(config, seed=7)
    validate_weights(weights, topology, config)
    assert np.all(weights[: config.n_excitatory] >= 0.0)
    assert np.all(weights[config.n_excitatory :] <= 0.0)
    assert np.all(np.diag(weights) == 0.0)


def test_plasticity_mask_selects_only_excitatory_to_excitatory() -> None:
    config = _config()
    _, topology = initialise_weights(config, seed=7)
    mask = plasticity_mask(topology, config)
    excitatory = config.n_excitatory
    assert mask[:excitatory, :excitatory].sum() > 0
    assert mask[excitatory:].sum() == 0
    assert mask[:, excitatory:].sum() == 0


def test_validate_weights_rejects_wrong_shape() -> None:
    config = _config()
    topology = ~np.eye(5, dtype=np.bool_)
    with pytest.raises(ValueError, match="shapes must match"):
        validate_weights(np.zeros((4, 4)), topology, config)


def test_validate_weights_rejects_non_finite_values() -> None:
    config = _config()
    weights, topology = initialise_weights(config, seed=7)
    weights[0, 1] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        validate_weights(weights, topology, config)


def test_validate_weights_rejects_topology_or_diagonal_violation() -> None:
    config = _config()
    weights, topology = initialise_weights(config, seed=7)
    weights[0, 0] = 0.1
    with pytest.raises(ValueError, match="topology or zero diagonal"):
        validate_weights(weights, topology, config)


def test_validate_weights_rejects_negative_excitatory_row() -> None:
    config = _config()
    weights, topology = initialise_weights(config, seed=7)
    weights[0, 1] = -0.1
    with pytest.raises(ValueError, match="excitatory presynaptic"):
        validate_weights(weights, topology, config)


def test_validate_weights_rejects_positive_inhibitory_row() -> None:
    config = _config()
    weights, topology = initialise_weights(config, seed=7)
    weights[config.n_excitatory, 0] = 0.1
    with pytest.raises(ValueError, match="inhibitory presynaptic"):
        validate_weights(weights, topology, config)


def test_validate_weights_rejects_excitatory_above_maximum() -> None:
    config = _config()
    weights, topology = initialise_weights(config, seed=7)
    weights[0, 1] = config.weight_max + 1.0
    with pytest.raises(ValueError, match="exceed the declared maximum"):
        validate_weights(weights, topology, config)


def test_validate_weights_rejects_inhibitory_below_negative_maximum() -> None:
    config = _config()
    weights, topology = initialise_weights(config, seed=7)
    weights[config.n_excitatory, 0] = -(config.weight_max + 1.0)
    with pytest.raises(ValueError, match="exceed the declared magnitude"):
        validate_weights(weights, topology, config)
