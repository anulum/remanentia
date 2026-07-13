# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Temporal SNN state and invariants

"""Network state construction and row-pre/column-post invariants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

from snn_memory.contracts import ModelConfig

FloatArray: TypeAlias = npt.NDArray[np.float64]
IntArray: TypeAlias = npt.NDArray[np.int64]
BoolArray: TypeAlias = npt.NDArray[np.bool_]


@dataclass
class NetworkState:
    """Mutable timestep state separated from frozen recurrent weights."""

    voltage: FloatArray
    refractory: IntArray
    pre_trace: FloatArray
    post_trace: FloatArray
    spikes: BoolArray
    step: int = 0

    def copy(self) -> NetworkState:
        """Return a deep state copy suitable for an independent episode."""
        return NetworkState(
            self.voltage.copy(),
            self.refractory.copy(),
            self.pre_trace.copy(),
            self.post_trace.copy(),
            self.spikes.copy(),
            self.step,
        )


def initialise_state(config: ModelConfig) -> NetworkState:
    """Create a deterministic resting network state."""
    n = config.n_neurons
    return NetworkState(
        voltage=np.full(n, config.v_rest_mv, dtype=np.float64),
        refractory=np.zeros(n, dtype=np.int64),
        pre_trace=np.zeros(n, dtype=np.float64),
        post_trace=np.zeros(n, dtype=np.float64),
        spikes=np.zeros(n, dtype=np.bool_),
    )


def initialise_weights(config: ModelConfig, seed: int) -> tuple[FloatArray, BoolArray]:
    """Create seeded recurrent weights obeying topology and Dale signs."""
    rng = np.random.default_rng(seed)
    topology = rng.random((config.n_neurons, config.n_neurons)) < config.connectivity
    np.fill_diagonal(topology, False)
    weights: FloatArray = np.zeros(
        (config.n_neurons, config.n_neurons), dtype=np.float64
    )
    excitatory = rng.uniform(0.05, 0.2, size=weights.shape)
    inhibitory = -rng.uniform(0.05, 0.2, size=weights.shape)
    weights[: config.n_excitatory] = excitatory[: config.n_excitatory]
    weights[config.n_excitatory :] = inhibitory[config.n_excitatory :]
    weights *= topology
    validate_weights(weights, topology, config)
    return weights, topology


def plasticity_mask(topology: BoolArray, config: ModelConfig) -> BoolArray:
    """Return the primary experiment's E-to-E plastic connections."""
    mask = np.zeros_like(topology)
    mask[: config.n_excitatory, : config.n_excitatory] = topology[
        : config.n_excitatory, : config.n_excitatory
    ]
    return mask


def validate_weights(weights: FloatArray, topology: BoolArray, config: ModelConfig) -> None:
    """Raise when shape, finiteness, topology, diagonal, or Dale signs differ."""
    shape = (config.n_neurons, config.n_neurons)
    if weights.shape != shape or topology.shape != shape:
        raise ValueError("weight and topology shapes must match model size")
    if not np.all(np.isfinite(weights)):
        raise ValueError("weights contain non-finite values")
    if np.any(weights[~topology] != 0.0) or np.any(np.diag(weights) != 0.0):
        raise ValueError("weights violate topology or zero diagonal")
    if np.any(weights[: config.n_excitatory] < 0.0):
        raise ValueError("excitatory presynaptic rows must be non-negative")
    if np.any(weights[: config.n_excitatory] > config.weight_max):
        raise ValueError("excitatory weights exceed the declared maximum")
    if np.any(weights[config.n_excitatory :] > 0.0):
        raise ValueError("inhibitory presynaptic rows must be non-positive")
    if np.any(weights[config.n_excitatory :] < -config.weight_max):
        raise ValueError("inhibitory weights exceed the declared magnitude")
