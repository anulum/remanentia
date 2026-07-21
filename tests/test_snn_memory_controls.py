# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Matched-control tests

"""Public-control tests: learned-pathway locality over real recurrent matrices."""

from __future__ import annotations

import numpy as np

from snn_memory.contracts import ModelConfig
from snn_memory.controls import make_control
from snn_memory.state import initialise_weights, validate_weights


def _fixed_scaffold(shuffled: np.ndarray, weights: np.ndarray, excitatory: int) -> None:
    """Assert only the E→E block moved; every fixed pathway is byte-identical."""
    np.testing.assert_array_equal(
        shuffled[:excitatory, excitatory:], weights[:excitatory, excitatory:]
    )
    np.testing.assert_array_equal(shuffled[excitatory:], weights[excitatory:])


def test_trained_control_returns_an_independent_copy() -> None:
    model = ModelConfig(n_neurons=20, connectivity=0.4)
    weights, topology = initialise_weights(model, 11)
    trained = make_control(weights, topology, model, "trained", 29)
    np.testing.assert_array_equal(trained, weights)
    assert trained is not weights


def test_shuffled_control_permutes_only_the_learned_excitatory_block() -> None:
    model = ModelConfig(n_neurons=20, connectivity=0.4)
    weights, topology = initialise_weights(model, 11)
    shuffled = make_control(weights, topology, model, "shuffled", 29)
    excitatory = model.n_excitatory
    np.testing.assert_array_equal(shuffled != 0.0, topology)
    _fixed_scaffold(shuffled, weights, excitatory)
    block = shuffled[:excitatory, :excitatory]
    trained_block = weights[:excitatory, :excitatory]
    np.testing.assert_allclose(np.sort(block.ravel()), np.sort(trained_block.ravel()))
    assert not np.array_equal(block, trained_block)


def test_random_control_resamples_the_block_and_preserves_invariants() -> None:
    model = ModelConfig(n_neurons=20, connectivity=0.4)
    weights, topology = initialise_weights(model, 11)
    random = make_control(weights, topology, model, "random", 29)
    validate_weights(random, topology, model)
    _fixed_scaffold(random, weights, model.n_excitatory)
    assert np.all(random[: model.n_excitatory] <= model.weight_max)


def test_zero_control_removes_all_recurrent_weight() -> None:
    model = ModelConfig(n_neurons=20, connectivity=0.4)
    weights, topology = initialise_weights(model, 11)
    zero = make_control(weights, topology, model, "zero", 29)
    assert not zero.any()


def test_control_is_a_noop_when_no_excitatory_connections_are_learned() -> None:
    model = ModelConfig(n_neurons=5, excitatory_fraction=0.8, connectivity=1.0)
    excitatory = model.n_excitatory
    topology = np.zeros((5, 5), dtype=np.bool_)
    topology[:excitatory, excitatory:] = True
    topology[excitatory:, :excitatory] = True
    weights = np.zeros((5, 5), dtype=np.float64)
    weights[:excitatory, excitatory:] = 0.1
    weights[excitatory:, :excitatory] = -0.1
    validate_weights(weights, topology, model)
    shuffled = make_control(weights, topology, model, "shuffled", 29)
    np.testing.assert_array_equal(shuffled, weights)
