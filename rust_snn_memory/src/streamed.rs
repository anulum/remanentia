// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Remanentia — Sparse streamed temporal SNN episode engine

//! Sparse recurrence, immutable adjacency, exact traces, and current evidence.

use std::collections::BTreeSet;

use sha2::{Digest, Sha256};

use crate::{EngineError, EngineResult, ModelConfig, NetworkState, WeightMatrix};

const TOPOLOGY_DIGEST_DOMAIN: &[u8] = b"remanentia:snn-memory:topology:v2";
const STATE_DIGEST_DOMAIN: &[u8] = b"remanentia:snn-memory:state:v2";

/// Deterministic episode phase attached to every timestep summary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EpisodePhase {
    /// Externally supplied cue/training portion of an episode.
    Cue,
    /// Autonomous or no-input completion portion of an episode.
    Completion,
}

impl EpisodePhase {
    /// Return the stable lowercase value used by JSON schemas and adapters.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Cue => "cue",
            Self::Completion => "completion",
        }
    }
}

/// Evidence summary for one ordered target-neuron current vector.
#[derive(Clone, Debug, PartialEq)]
pub struct CurrentVectorSummary {
    /// Signed sum in ascending target-neuron order.
    pub signed_sum: f64,
    /// Sum of absolute target-neuron currents.
    pub l1: f64,
    /// Euclidean norm, computed as an ordered `hypot` fold (never squared energy).
    pub l2_norm: f64,
    /// Number of target neurons whose current entry is nonzero, not active edges.
    pub nonzero_count: usize,
}

/// Typed per-timestep current and phase evidence.
#[derive(Clone, Debug, PartialEq)]
pub struct CurrentSummary {
    /// Cue or completion phase derived from the declared cue-step boundary.
    pub phase: EpisodePhase,
    /// External current across target neurons.
    pub external: CurrentVectorSummary,
    /// Recurrent contribution from excitatory presynaptic neurons.
    pub recurrent_excitatory: CurrentVectorSummary,
    /// Recurrent contribution from inhibitory presynaptic neurons (signed negative).
    pub recurrent_inhibitory: CurrentVectorSummary,
    /// Per-target excitatory plus inhibitory recurrent current.
    pub recurrent_net: CurrentVectorSummary,
}

/// Sparse episode output without dense voltage, refractory, or trace histories.
#[derive(Clone, Debug, PartialEq)]
pub struct StreamedEpisodeResult {
    /// Ascending spike indices emitted at every timestep.
    pub spike_indices: Vec<Vec<usize>>,
    /// Per-timestep typed current evidence.
    pub current_summaries: Vec<CurrentSummary>,
    /// Compact authenticated state digest after every complete timestep.
    pub state_digests: Vec<String>,
    /// Fully validated final state.
    pub final_state: NetworkState,
    /// Fully validated final dense weights for checkpoint compatibility.
    pub final_weights: WeightMatrix,
}

/// Immutable authenticated dense topology plus reciprocal sparse adjacency.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImmutableAdjacency {
    neurons: usize,
    excitatory_neurons: usize,
    topology: Vec<bool>,
    outgoing: Vec<Vec<usize>>,
    incoming: Vec<Vec<usize>>,
    topology_digest: String,
}

impl ImmutableAdjacency {
    /// Derive strictly ascending incoming/outgoing lists from a validated matrix.
    pub fn from_weights(weights: &WeightMatrix) -> EngineResult<Self> {
        weights.validate()?;
        let mut outgoing = vec![Vec::new(); weights.neurons];
        let mut incoming = vec![Vec::new(); weights.neurons];
        for (pre, outgoing_for_pre) in outgoing.iter_mut().enumerate() {
            for (post, incoming_for_post) in incoming.iter_mut().enumerate() {
                if weights.is_connected(pre, post) {
                    outgoing_for_pre.push(post);
                    incoming_for_post.push(pre);
                }
            }
        }
        let topology = weights.topology().to_vec();
        let topology_digest =
            topology_digest(weights.neurons, weights.excitatory_neurons, &topology);
        let adjacency = Self {
            neurons: weights.neurons,
            excitatory_neurons: weights.excitatory_neurons,
            topology,
            outgoing,
            incoming,
            topology_digest,
        };
        adjacency.validate_against(weights)?;
        Ok(adjacency)
    }

    /// Population size bound into the topology digest.
    pub const fn neurons(&self) -> usize {
        self.neurons
    }

    /// Excitatory population size bound into the topology digest.
    pub const fn excitatory_neurons(&self) -> usize {
        self.excitatory_neurons
    }

    /// Read-only dense topology mask in row-pre/column-post order.
    pub fn topology(&self) -> &[bool] {
        &self.topology
    }

    /// Read-only strictly ascending outgoing adjacency lists.
    pub fn outgoing(&self) -> &[Vec<usize>] {
        &self.outgoing
    }

    /// Read-only strictly ascending incoming adjacency lists.
    pub fn incoming(&self) -> &[Vec<usize>] {
        &self.incoming
    }

    /// Lowercase SHA-256 authentication digest.
    pub fn topology_digest(&self) -> &str {
        &self.topology_digest
    }

    /// Validate digest, exact mask identity, reciprocal lists, order, and diagonal.
    pub fn validate_against(&self, weights: &WeightMatrix) -> EngineResult<()> {
        if self.neurons != weights.neurons
            || self.excitatory_neurons != weights.excitatory_neurons
            || self.topology.len() != self.neurons * self.neurons
            || self.outgoing.len() != self.neurons
            || self.incoming.len() != self.neurons
        {
            return Err(EngineError::Shape(
                "adjacency and weight population metadata must match",
            ));
        }
        if self.topology != weights.topology() {
            return Err(EngineError::InvalidState(
                "adjacency topology differs from immutable weight topology",
            ));
        }
        let expected_digest =
            topology_digest(self.neurons, self.excitatory_neurons, &self.topology);
        if self.topology_digest != expected_digest {
            return Err(EngineError::InvalidState(
                "adjacency topology digest authentication failed",
            ));
        }
        for pre in 0..self.neurons {
            validate_sorted_unique(&self.outgoing[pre], self.neurons)?;
            for post in 0..self.neurons {
                let connected = self.topology[pre * self.neurons + post];
                let outgoing = self.outgoing[pre].binary_search(&post).is_ok();
                let incoming = self.incoming[post].binary_search(&pre).is_ok();
                if (pre == post && connected) || connected != outgoing || connected != incoming {
                    return Err(EngineError::InvalidState(
                        "adjacency lists, topology, or diagonal disagree",
                    ));
                }
            }
        }
        for incoming in &self.incoming {
            validate_sorted_unique(incoming, self.neurons)?;
        }
        Ok(())
    }
}

/// SHA-256 framing is: u32-BE domain length, domain bytes, u64-BE neurons,
/// u64-BE excitatory neurons, u64-BE mask-bit count, then one byte (0 or 1) per
/// row-major mask bit. This is delimiter-free and reproducible by Python.
fn topology_digest(neurons: usize, excitatory_neurons: usize, topology: &[bool]) -> String {
    let mut digest = Sha256::new();
    digest.update((TOPOLOGY_DIGEST_DOMAIN.len() as u32).to_be_bytes());
    digest.update(TOPOLOGY_DIGEST_DOMAIN);
    digest.update((neurons as u64).to_be_bytes());
    digest.update((excitatory_neurons as u64).to_be_bytes());
    digest.update((topology.len() as u64).to_be_bytes());
    for connected in topology {
        digest.update([u8::from(*connected)]);
    }
    digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// Hash a complete timestep state using frozen language-independent framing.
///
/// Framing is u32-BE domain length, domain bytes, u64-BE neuron count, then for
/// voltage, refractory, spikes, pre-trace, and post-trace respectively: u64-BE
/// element count followed by f64 bit patterns BE, u32 values BE, bytes 0/1, and
/// f64 bit patterns BE. No platform-native representation enters the digest.
pub fn state_digest(state: &NetworkState) -> EngineResult<String> {
    state.validate()?;
    let mut digest = Sha256::new();
    digest.update((STATE_DIGEST_DOMAIN.len() as u32).to_be_bytes());
    digest.update(STATE_DIGEST_DOMAIN);
    digest.update((state.neurons() as u64).to_be_bytes());

    digest.update((state.voltage_mv.len() as u64).to_be_bytes());
    for value in &state.voltage_mv {
        digest.update(value.to_bits().to_be_bytes());
    }
    digest.update((state.refractory_steps.len() as u64).to_be_bytes());
    for value in &state.refractory_steps {
        digest.update(value.to_be_bytes());
    }
    digest.update((state.spikes.len() as u64).to_be_bytes());
    for value in &state.spikes {
        digest.update([u8::from(*value)]);
    }
    digest.update((state.pre_trace.len() as u64).to_be_bytes());
    for value in &state.pre_trace {
        digest.update(value.to_bits().to_be_bytes());
    }
    digest.update((state.post_trace.len() as u64).to_be_bytes());
    for value in &state.post_trace {
        digest.update(value.to_bits().to_be_bytes());
    }
    Ok(digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

fn validate_sorted_unique(indices: &[usize], neurons: usize) -> EngineResult<()> {
    if indices.iter().any(|index| *index >= neurons)
        || indices.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err(EngineError::InvalidState(
            "adjacency indices must be in-range, strictly ascending, and unique",
        ));
    }
    Ok(())
}

fn summarize_current(values: &[f64]) -> EngineResult<CurrentVectorSummary> {
    let mut signed_sum = 0.0_f64;
    let mut l1 = 0.0_f64;
    let mut l2_norm = 0.0_f64;
    let mut nonzero_count = 0;
    for value in values {
        signed_sum += value;
        l1 += value.abs();
        let high = l2_norm.max(value.abs());
        let low = l2_norm.min(value.abs());
        l2_norm = if high == 0.0 {
            0.0
        } else {
            let ratio = low / high;
            high * (1.0 + ratio * ratio).sqrt()
        };
        nonzero_count += usize::from(*value != 0.0);
    }
    if !signed_sum.is_finite() || !l1.is_finite() || !l2_norm.is_finite() {
        return Err(EngineError::NonFinite("current summary"));
    }
    Ok(CurrentVectorSummary {
        signed_sum,
        l1,
        l2_norm,
        nonzero_count,
    })
}

fn recurrent_currents(
    state: &NetworkState,
    weights: &WeightMatrix,
    adjacency: &ImmutableAdjacency,
) -> EngineResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    let mut excitatory = vec![0.0; weights.neurons];
    let mut inhibitory = vec![0.0; weights.neurons];
    let mut net = vec![0.0; weights.neurons];
    for pre in 0..weights.neurons {
        if !state.spikes[pre] {
            continue;
        }
        let target = if pre < weights.excitatory_neurons {
            &mut excitatory
        } else {
            &mut inhibitory
        };
        for post in &adjacency.outgoing[pre] {
            let value = weights.get(pre, *post);
            target[*post] += value;
            net[*post] += value;
            if !target[*post].is_finite() || !net[*post].is_finite() {
                return Err(EngineError::NonFinite("recurrent current"));
            }
        }
    }
    Ok((excitatory, inhibitory, net))
}

fn step_streamed_lif(
    state: &mut NetworkState,
    external: &[f64],
    recurrent: &[f64],
    config: &ModelConfig,
) -> EngineResult<Vec<usize>> {
    let neurons = state.neurons();
    if external.len() != neurons || recurrent.len() != neurons {
        return Err(EngineError::Shape(
            "external and recurrent current must match network state",
        ));
    }
    let mut spike_indices = Vec::new();
    state.spikes.fill(false);
    for neuron in 0..neurons {
        if state.refractory_steps[neuron] > 0 {
            state.refractory_steps[neuron] -= 1;
            state.voltage_mv[neuron] = config.v_reset_mv;
            continue;
        }
        let voltage = state.voltage_mv[neuron];
        // This left association is the declared Python-oracle recurrence.
        let leak_and_external = -(voltage - config.v_rest_mv) / config.tau_m_ms + external[neuron];
        let delta = (leak_and_external + recurrent[neuron]) * config.dt_ms;
        if !delta.is_finite() {
            return Err(EngineError::NonFinite("membrane voltage delta"));
        }
        state.voltage_mv[neuron] += delta;
        if !state.voltage_mv[neuron].is_finite() {
            return Err(EngineError::NonFinite("membrane voltage"));
        }
        if state.voltage_mv[neuron] >= config.v_threshold_mv {
            state.spikes[neuron] = true;
            spike_indices.push(neuron);
            state.voltage_mv[neuron] = config.v_reset_mv;
            state.refractory_steps[neuron] = config.refractory_steps();
        }
    }
    Ok(spike_indices)
}

fn validate_touched_edge(
    weights: &WeightMatrix,
    adjacency: &ImmutableAdjacency,
    pre: usize,
    post: usize,
    config: &ModelConfig,
) -> EngineResult<()> {
    if pre >= config.excitatory_neurons
        || post >= config.excitatory_neurons
        || pre == post
        || !weights.is_connected(pre, post)
        || adjacency.outgoing[pre].binary_search(&post).is_err()
        || adjacency.incoming[post].binary_search(&pre).is_err()
    {
        return Err(EngineError::InvalidState(
            "touched plastic edge violates immutable E-to-E topology",
        ));
    }
    let value = weights.get(pre, post);
    if !value.is_finite() || value < 0.0 || value > config.excitatory_weight_max {
        return Err(EngineError::InvalidState(
            "touched plastic edge violates Dale sign or weight bounds",
        ));
    }
    Ok(())
}

fn advance_streamed_traces_and_plasticity(
    state: &mut NetworkState,
    weights: &mut WeightMatrix,
    adjacency: &ImmutableAdjacency,
    plasticity_enabled: bool,
    config: &ModelConfig,
) -> EngineResult<()> {
    let pre_decay = (-config.dt_ms / config.tau_plus_ms).exp();
    let post_decay = (-config.dt_ms / config.tau_minus_ms).exp();
    for neuron in 0..state.neurons() {
        state.pre_trace[neuron] *= pre_decay;
        state.post_trace[neuron] *= post_decay;
    }

    if plasticity_enabled {
        let mut touched = BTreeSet::new();
        for neuron in 0..state.neurons() {
            if !state.spikes[neuron] || neuron >= config.excitatory_neurons {
                continue;
            }
            for post in &adjacency.outgoing[neuron] {
                if *post < config.excitatory_neurons {
                    touched.insert((neuron, *post));
                }
            }
            for pre in &adjacency.incoming[neuron] {
                if *pre < config.excitatory_neurons {
                    touched.insert((*pre, neuron));
                }
            }
        }
        for (pre, post) in touched {
            validate_touched_edge(weights, adjacency, pre, post, config)?;
            let ltp = if state.spikes[post] {
                config.a_plus * state.pre_trace[pre]
            } else {
                0.0
            };
            let ltd = if state.spikes[pre] {
                config.a_minus * state.post_trace[post]
            } else {
                0.0
            };
            let delta = ltp - ltd;
            if !delta.is_finite() {
                return Err(EngineError::NonFinite("plasticity delta"));
            }
            let updated = (weights.get(pre, post) + delta).clamp(0.0, config.excitatory_weight_max);
            weights.set(pre, post, updated);
            validate_touched_edge(weights, adjacency, pre, post, config)?;
        }
    }

    for neuron in 0..state.neurons() {
        if state.spikes[neuron] {
            state.pre_trace[neuron] += 1.0;
            state.post_trace[neuron] += 1.0;
        }
    }
    Ok(())
}

/// Run one sparse streamed episode with a deterministic cue/completion boundary.
pub fn run_streamed_episode_core(
    state: &mut NetworkState,
    weights: &mut WeightMatrix,
    adjacency: &ImmutableAdjacency,
    packets: &[Vec<f64>],
    cue_steps: usize,
    plasticity_enabled: bool,
    config: &ModelConfig,
) -> EngineResult<StreamedEpisodeResult> {
    state.validate()?;
    weights.validate_for_config(config)?;
    config.validate(state.neurons())?;
    adjacency.validate_against(weights)?;
    if state.neurons() != weights.neurons {
        return Err(EngineError::Shape(
            "state, weights, adjacency, and configuration must describe one population",
        ));
    }
    if packets.is_empty() {
        return Err(EngineError::Shape("episode requires at least one timestep"));
    }
    if cue_steps > packets.len() {
        return Err(EngineError::Shape(
            "cue steps cannot exceed episode timesteps",
        ));
    }
    for packet in packets {
        if packet.len() != state.neurons() {
            return Err(EngineError::Shape(
                "every packet must match the population size",
            ));
        }
        if packet.iter().any(|value| !value.is_finite()) {
            return Err(EngineError::NonFinite("external current packet"));
        }
    }

    let mut spike_indices = Vec::with_capacity(packets.len());
    let mut current_summaries = Vec::with_capacity(packets.len());
    let mut state_digests = Vec::with_capacity(packets.len());
    for (timestep, packet) in packets.iter().enumerate() {
        let (excitatory, inhibitory, net) = recurrent_currents(state, weights, adjacency)?;
        let phase = if timestep < cue_steps {
            EpisodePhase::Cue
        } else {
            EpisodePhase::Completion
        };
        let summary = CurrentSummary {
            phase,
            external: summarize_current(packet)?,
            recurrent_excitatory: summarize_current(&excitatory)?,
            recurrent_inhibitory: summarize_current(&inhibitory)?,
            recurrent_net: summarize_current(&net)?,
        };
        let timestep_spikes = step_streamed_lif(state, packet, &net, config)?;
        advance_streamed_traces_and_plasticity(
            state,
            weights,
            adjacency,
            plasticity_enabled,
            config,
        )?;
        spike_indices.push(timestep_spikes);
        current_summaries.push(summary);
        state_digests.push(state_digest(state)?);
    }

    state.validate()?;
    weights.validate_for_config(config)?;
    adjacency.validate_against(weights)?;
    Ok(StreamedEpisodeResult {
        spike_indices,
        current_summaries,
        state_digests,
        final_state: state.clone(),
        final_weights: weights.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> ModelConfig {
        ModelConfig {
            dt_ms: 1.0,
            tau_m_ms: 1.0,
            v_rest_mv: 0.0,
            v_reset_mv: 0.0,
            v_threshold_mv: 1.0,
            refractory_ms: 0.0,
            tau_plus_ms: 20.0,
            tau_minus_ms: 20.0,
            a_plus: 0.005,
            a_minus: 0.006,
            excitatory_weight_max: 1.0,
            excitatory_neurons: 2,
        }
    }

    fn matrix() -> WeightMatrix {
        WeightMatrix::from_flat(
            3,
            2,
            vec![0.0, 0.2, 0.0, 0.3, 0.0, 0.0, -0.4, -0.5, 0.0],
            vec![false, true, false, true, false, false, true, true, false],
        )
        .expect("valid fixture")
    }

    #[test]
    fn corrupted_adjacency_digest_and_lists_are_rejected() {
        let weights = matrix();
        let mut metadata_corrupt = ImmutableAdjacency::from_weights(&weights).expect("adjacency");
        metadata_corrupt.neurons += 1;
        assert!(metadata_corrupt.validate_against(&weights).is_err());

        let mut digest_corrupt = ImmutableAdjacency::from_weights(&weights).expect("adjacency");
        digest_corrupt.topology_digest.replace_range(0..1, "f");
        assert!(digest_corrupt.validate_against(&weights).is_err());

        let mut list_corrupt = ImmutableAdjacency::from_weights(&weights).expect("adjacency");
        list_corrupt.outgoing[0] = vec![2, 1];
        assert!(list_corrupt.validate_against(&weights).is_err());

        let mut reciprocal_corrupt = ImmutableAdjacency::from_weights(&weights).expect("adjacency");
        reciprocal_corrupt.outgoing[0].clear();
        assert!(reciprocal_corrupt.validate_against(&weights).is_err());

        let mut topology_corrupt = ImmutableAdjacency::from_weights(&weights).expect("adjacency");
        topology_corrupt.topology[1] = false;
        assert!(topology_corrupt.validate_against(&weights).is_err());
    }

    #[test]
    fn topology_digest_framing_is_frozen() {
        let adjacency = ImmutableAdjacency::from_weights(&matrix()).expect("adjacency");
        assert_eq!(
            adjacency.topology_digest(),
            "78726f211c1f9110c3f074155b5853159eda87dd3732e9aa5852e4b0741fd325"
        );
    }

    #[test]
    fn state_digest_framing_is_frozen() {
        let cfg = config();
        let mut state = NetworkState::reset(3, &cfg).expect("state");
        state.voltage_mv = vec![0.0, -0.0, 1.5];
        state.refractory_steps = vec![0, 1, u32::MAX];
        state.spikes = vec![false, true, false];
        state.pre_trace = vec![0.25, 0.5, 1.0];
        state.post_trace = vec![1.0, 0.5, 0.25];
        assert_eq!(
            state_digest(&state).expect("digest"),
            "e6d34eda6504fbdae1b375e2cc96eba2bb043455dd6d4ed66eeb0206ee39a156"
        );
    }

    #[test]
    fn touched_edge_validation_rejects_corrupt_value_and_diagonal() {
        let cfg = config();
        let mut weights = matrix();
        let adjacency = ImmutableAdjacency::from_weights(&weights).expect("adjacency");
        weights.set(0, 1, -0.1);
        assert!(validate_touched_edge(&weights, &adjacency, 0, 1, &cfg).is_err());
        assert!(validate_touched_edge(&weights, &adjacency, 0, 0, &cfg).is_err());
    }
}
