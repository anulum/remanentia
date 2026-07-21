# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Reference LIF and online STDP dynamics

"""Readable timestep oracle for the temporal SNN memory experiment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from snn_memory.contracts import ModelConfig
from snn_memory.state import BoolArray, FloatArray, NetworkState, plasticity_mask, validate_weights


@dataclass(frozen=True)
class EpisodeResult:
    """Per-timestep observables returned by the public episode surface."""

    spikes: npt.NDArray[np.bool_]
    voltages: npt.NDArray[np.float64]
    recurrent_current: npt.NDArray[np.float64]
    refractory: npt.NDArray[np.int64]
    pre_traces: npt.NDArray[np.float64]
    post_traces: npt.NDArray[np.float64]
    final_state: NetworkState
    final_weights: FloatArray


def step_network(
    state: NetworkState,
    weights: FloatArray,
    topology: BoolArray,
    input_current: FloatArray,
    config: ModelConfig,
    *,
    plasticity_enabled: bool,
) -> tuple[NetworkState, FloatArray, FloatArray]:
    """Advance one LIF/STDP timestep using ``W[pre, post]`` orientation."""
    if input_current.shape != (config.n_neurons,) or not np.all(np.isfinite(input_current)):
        raise ValueError("input_current must be one finite value per neuron")
    validate_weights(weights, topology, config)
    next_state = state.copy()
    next_weights = weights.copy()
    recurrent_current = np.zeros(config.n_neurons, dtype=np.float64)
    for pre in range(config.n_neurons):
        if not state.spikes[pre]:
            continue
        for post in range(config.n_neurons):
            recurrent_current[post] += weights[pre, post]
    active = state.refractory == 0
    dv = (
        -(state.voltage - config.v_rest_mv) / config.tau_m_ms + input_current + recurrent_current
    ) * config.dt_ms
    next_state.voltage[active] += dv[active]
    next_state.voltage[~active] = config.v_reset_mv
    next_state.refractory = np.maximum(state.refractory - 1, 0)
    spikes = active & (next_state.voltage >= config.v_threshold_mv)
    next_state.voltage[spikes] = config.v_reset_mv
    next_state.refractory[spikes] = config.refractory_steps

    pre_decay = np.exp(-config.dt_ms / config.tau_plus_ms)
    post_decay = np.exp(-config.dt_ms / config.tau_minus_ms)
    decayed_pre = state.pre_trace * pre_decay
    decayed_post = state.post_trace * post_decay
    if plasticity_enabled:
        mask = plasticity_mask(topology, config)
        ltp = config.a_plus * np.outer(decayed_pre, spikes.astype(np.float64))
        ltd = config.a_minus * np.outer(spikes.astype(np.float64), decayed_post)
        next_weights[mask] += (ltp - ltd)[mask]
        np.clip(
            next_weights[: config.n_excitatory],
            config.weight_min,
            config.weight_max,
            out=next_weights[: config.n_excitatory],
        )
        next_weights[~topology] = 0.0

    next_state.pre_trace = decayed_pre + spikes
    next_state.post_trace = decayed_post + spikes
    next_state.spikes = spikes
    next_state.step += 1
    validate_weights(next_weights, topology, config)
    return next_state, next_weights, recurrent_current


def run_episode(
    state: NetworkState,
    weights: FloatArray,
    topology: BoolArray,
    currents: FloatArray,
    config: ModelConfig,
    *,
    plasticity_enabled: bool,
) -> EpisodeResult:
    """Run a complete current sequence through the public reference engine."""
    if currents.ndim != 2 or currents.shape[1] != config.n_neurons:
        raise ValueError("currents must have shape (timesteps, n_neurons)")
    if currents.shape[0] == 0:
        raise ValueError("episode requires at least one timestep")
    episode_state = state.copy()
    episode_weights = weights.copy()
    spike_rows: list[BoolArray] = []
    voltage_rows: list[FloatArray] = []
    recurrent_rows: list[FloatArray] = []
    refractory_rows: list[npt.NDArray[np.int64]] = []
    pre_trace_rows: list[FloatArray] = []
    post_trace_rows: list[FloatArray] = []
    for current in currents:
        episode_state, episode_weights, recurrent = step_network(
            episode_state,
            episode_weights,
            topology,
            current,
            config,
            plasticity_enabled=plasticity_enabled,
        )
        spike_rows.append(episode_state.spikes.copy())
        voltage_rows.append(episode_state.voltage.copy())
        recurrent_rows.append(recurrent)
        refractory_rows.append(episode_state.refractory.copy())
        pre_trace_rows.append(episode_state.pre_trace.copy())
        post_trace_rows.append(episode_state.post_trace.copy())
    return EpisodeResult(
        np.asarray(spike_rows, dtype=np.bool_),
        np.asarray(voltage_rows, dtype=np.float64),
        np.asarray(recurrent_rows, dtype=np.float64),
        np.asarray(refractory_rows, dtype=np.int64),
        np.asarray(pre_trace_rows, dtype=np.float64),
        np.asarray(post_trace_rows, dtype=np.float64),
        episode_state,
        episode_weights,
    )
