// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Remanentia — Temporal LIF and online STDP engine

use pyo3::prelude::*;

use crate::{EngineError, EngineResult, ModelConfig, NetworkState, WeightMatrix};

/// Snapshot returned after one LIF timestep.
#[pyclass(get_all)]
#[derive(Clone, Debug, PartialEq)]
pub struct StepResult {
    /// Indices of neurons that emitted a spike.
    pub spike_indices: Vec<usize>,
    /// Membrane potential after threshold/reset processing.
    pub voltage_mv: Vec<f64>,
    /// Remaining refractory timesteps after the step.
    pub refractory_steps: Vec<u32>,
}

/// Full time-resolved output from one episode.
#[pyclass(get_all)]
#[derive(Clone, Debug, PartialEq)]
pub struct EpisodeResult {
    /// Spike indices emitted at every timestep.
    pub spike_indices: Vec<Vec<usize>>,
    /// Membrane potential snapshots at every timestep.
    pub voltage_mv: Vec<Vec<f64>>,
    /// Refractory counter snapshots at every timestep.
    pub refractory_steps: Vec<Vec<u32>>,
    /// Presynaptic trace snapshots at every timestep.
    pub pre_trace: Vec<Vec<f64>>,
    /// Postsynaptic trace snapshots at every timestep.
    pub post_trace: Vec<Vec<f64>>,
    /// Recurrent current before the membrane update at every timestep.
    pub recurrent_current: Vec<Vec<f64>>,
    /// Final state after the episode.
    pub final_state: NetworkState,
    /// Final weights after the episode.
    pub final_weights: WeightMatrix,
}

/// Advance LIF membrane and refractory state by one timestep.
pub fn step_lif_core(
    state: &mut NetworkState,
    input_current: &[f64],
    config: &ModelConfig,
) -> EngineResult<StepResult> {
    state.validate()?;
    let neurons = state.neurons();
    config.validate(neurons)?;
    if input_current.len() != neurons {
        return Err(EngineError::Shape(
            "input current length must match network state",
        ));
    }
    if input_current.iter().any(|value| !value.is_finite()) {
        return Err(EngineError::NonFinite("input current"));
    }

    let mut spike_indices = Vec::new();
    state.spikes.fill(false);
    for (neuron, current) in input_current.iter().enumerate() {
        if state.refractory_steps[neuron] > 0 {
            state.refractory_steps[neuron] -= 1;
            state.voltage_mv[neuron] = config.v_reset_mv;
            continue;
        }
        let voltage = state.voltage_mv[neuron];
        let delta = (-(voltage - config.v_rest_mv) / config.tau_m_ms + current) * config.dt_ms;
        state.voltage_mv[neuron] += delta;
        if state.voltage_mv[neuron] >= config.v_threshold_mv {
            state.spikes[neuron] = true;
            spike_indices.push(neuron);
            state.voltage_mv[neuron] = config.v_reset_mv;
            state.refractory_steps[neuron] = config.refractory_steps();
        }
    }
    Ok(StepResult {
        spike_indices,
        voltage_mv: state.voltage_mv.clone(),
        refractory_steps: state.refractory_steps.clone(),
    })
}

/// Apply one online pair-based STDP update and advance both traces.
pub fn step_stdp_core(
    weights: &mut WeightMatrix,
    pre_trace: &mut [f64],
    post_trace: &mut [f64],
    spikes: &[bool],
    config: &ModelConfig,
) -> EngineResult<()> {
    weights.validate_for_config(config)?;
    let neurons = weights.neurons;
    config.validate(neurons)?;
    if pre_trace.len() != neurons || post_trace.len() != neurons || spikes.len() != neurons {
        return Err(EngineError::Shape(
            "weights, traces, spikes, and configuration must describe one population",
        ));
    }
    if pre_trace
        .iter()
        .chain(post_trace.iter())
        .any(|value| !value.is_finite())
    {
        return Err(EngineError::NonFinite("STDP traces"));
    }
    let excitatory = config.excitatory_neurons;
    let pre_decay = (-config.dt_ms / config.tau_plus_ms).exp();
    let post_decay = (-config.dt_ms / config.tau_minus_ms).exp();
    for neuron in 0..neurons {
        pre_trace[neuron] *= pre_decay;
        post_trace[neuron] *= post_decay;
    }
    for pre in 0..excitatory {
        for post in 0..excitatory {
            if pre == post || !weights.is_connected(pre, post) {
                continue;
            }
            let mut delta = 0.0;
            if spikes[post] {
                delta += config.a_plus * pre_trace[pre];
            }
            if spikes[pre] {
                delta -= config.a_minus * post_trace[post];
            }
            let updated = (weights.get(pre, post) + delta).clamp(0.0, config.excitatory_weight_max);
            weights.set(pre, post, updated);
        }
    }

    for neuron in 0..neurons {
        let spike = u8::from(spikes[neuron]) as f64;
        pre_trace[neuron] += spike;
        post_trace[neuron] += spike;
    }
    weights.validate_for_config(config)
}

fn recurrent_current(state: &NetworkState, weights: &WeightMatrix) -> Vec<f64> {
    let mut current = vec![0.0; weights.neurons];
    for pre in 0..weights.neurons {
        if !state.spikes[pre] {
            continue;
        }
        for (post, target) in current.iter_mut().enumerate() {
            *target += weights.get(pre, post);
        }
    }
    current
}

fn advance_traces_without_plasticity(state: &mut NetworkState, config: &ModelConfig) {
    let pre_decay = (-config.dt_ms / config.tau_plus_ms).exp();
    let post_decay = (-config.dt_ms / config.tau_minus_ms).exp();
    for neuron in 0..state.neurons() {
        let spike = u8::from(state.spikes[neuron]) as f64;
        state.pre_trace[neuron] = state.pre_trace[neuron] * pre_decay + spike;
        state.post_trace[neuron] = state.post_trace[neuron] * post_decay + spike;
    }
}

/// Run a sequence of external-current packets through the recurrent network.
pub fn run_episode_core(
    state: &mut NetworkState,
    weights: &mut WeightMatrix,
    spike_packets: &[Vec<f64>],
    plasticity_enabled: bool,
    config: &ModelConfig,
) -> EngineResult<EpisodeResult> {
    state.validate()?;
    weights.validate_for_config(config)?;
    let neurons = state.neurons();
    config.validate(neurons)?;
    if weights.neurons != neurons {
        return Err(EngineError::Shape(
            "state, weights, and configuration must describe one population",
        ));
    }
    if spike_packets.is_empty() {
        return Err(EngineError::Shape("episode requires at least one timestep"));
    }

    let mut result = EpisodeResult {
        spike_indices: Vec::with_capacity(spike_packets.len()),
        voltage_mv: Vec::with_capacity(spike_packets.len()),
        refractory_steps: Vec::with_capacity(spike_packets.len()),
        pre_trace: Vec::with_capacity(spike_packets.len()),
        post_trace: Vec::with_capacity(spike_packets.len()),
        recurrent_current: Vec::with_capacity(spike_packets.len()),
        final_state: state.clone(),
        final_weights: weights.clone(),
    };
    for packet in spike_packets {
        if packet.len() != neurons {
            return Err(EngineError::Shape(
                "every spike packet must match the population size",
            ));
        }
        let recurrent = recurrent_current(state, weights);
        let total: Vec<f64> = packet
            .iter()
            .zip(&recurrent)
            .map(|(external, recurrent)| external + recurrent)
            .collect();
        let step = step_lif_core(state, &total, config)?;
        if plasticity_enabled {
            step_stdp_core(
                weights,
                &mut state.pre_trace,
                &mut state.post_trace,
                &state.spikes,
                config,
            )?;
        } else {
            advance_traces_without_plasticity(state, config);
        }
        result.spike_indices.push(step.spike_indices);
        result.voltage_mv.push(step.voltage_mv);
        result.refractory_steps.push(step.refractory_steps);
        result.pre_trace.push(state.pre_trace.clone());
        result.post_trace.push(state.post_trace.clone());
        result.recurrent_current.push(recurrent);
    }
    result.final_state = state.clone();
    result.final_weights = weights.clone();
    Ok(result)
}

/// Probe cloned frozen state and weights with plasticity disabled.
pub fn probe_episode_core(
    frozen_state: &NetworkState,
    frozen_weights: &WeightMatrix,
    cue_packets: &[Vec<f64>],
    config: &ModelConfig,
) -> EngineResult<EpisodeResult> {
    let mut state = frozen_state.clone();
    let mut weights = frozen_weights.clone();
    run_episode_core(&mut state, &mut weights, cue_packets, false, config)
}
