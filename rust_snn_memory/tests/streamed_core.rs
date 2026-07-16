// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Remanentia — Sparse streamed temporal SNN core tests

use rust_snn_memory::{
    run_streamed_episode_core, EpisodePhase, ImmutableAdjacency, ModelConfig, NetworkState,
    WeightMatrix,
};

fn config(neurons: usize, excitatory_neurons: usize) -> ModelConfig {
    assert!(neurons > excitatory_neurons);
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
        excitatory_neurons,
    }
}

fn matrix_three() -> WeightMatrix {
    WeightMatrix::from_flat(
        3,
        2,
        vec![0.0, 0.4, 0.0, 0.3, 0.0, 0.0, 0.0, -0.25, 0.0],
        vec![false, true, false, true, false, false, false, true, false],
    )
    .expect("valid three-neuron matrix")
}

#[test]
fn long_silence_eagerly_decays_every_trace_each_timestep() {
    let cfg = config(3, 2);
    let mut state = NetworkState::reset(3, &cfg).expect("reset");
    state.pre_trace = vec![1.0, 2.0, 3.0];
    state.post_trace = vec![4.0, 5.0, 6.0];
    let initial_pre = state.pre_trace.clone();
    let initial_post = state.post_trace.clone();
    let mut weights = matrix_three();
    let adjacency = ImmutableAdjacency::from_weights(&weights).expect("adjacency");
    assert_eq!(adjacency.outgoing(), &[vec![1], vec![0], vec![1]]);
    assert_eq!(adjacency.incoming(), &[vec![1], vec![0, 2], vec![]]);
    assert_eq!(adjacency.neurons(), 3);
    assert_eq!(adjacency.excitatory_neurons(), 2);
    assert_eq!(adjacency.topology_digest().len(), 64);
    let packets = vec![vec![0.0; 3]; 256];

    let result = run_streamed_episode_core(
        &mut state,
        &mut weights,
        &adjacency,
        &packets,
        0,
        false,
        &cfg,
    )
    .expect("silent episode");

    let pre_factor = (-cfg.dt_ms / cfg.tau_plus_ms).exp().powi(256);
    let post_factor = (-cfg.dt_ms / cfg.tau_minus_ms).exp().powi(256);
    for neuron in 0..3 {
        assert!(
            (result.final_state.pre_trace[neuron] - initial_pre[neuron] * pre_factor).abs() < 1e-15
        );
        assert!(
            (result.final_state.post_trace[neuron] - initial_post[neuron] * post_factor).abs()
                < 1e-15
        );
    }
    assert!(result.spike_indices.iter().all(Vec::is_empty));
    assert!(result
        .current_summaries
        .iter()
        .all(|summary| summary.phase == EpisodePhase::Completion));
}

#[test]
fn simultaneous_excitatory_inhibitory_currents_have_typed_target_summaries() {
    let cfg = config(3, 2);
    let mut state = NetworkState::reset(3, &cfg).expect("reset");
    state.spikes = vec![true, false, true];
    let mut weights = matrix_three();
    let adjacency = ImmutableAdjacency::from_weights(&weights).expect("adjacency");

    let result = run_streamed_episode_core(
        &mut state,
        &mut weights,
        &adjacency,
        &[vec![0.3, -0.4, 0.0]],
        1,
        false,
        &cfg,
    )
    .expect("mixed-current episode");

    let summary = &result.current_summaries[0];
    assert_eq!(summary.phase, EpisodePhase::Cue);
    assert_eq!(summary.phase.as_str(), "cue");
    assert!((summary.external.signed_sum + 0.1).abs() < 1e-15);
    assert_eq!(summary.external.l1, 0.7);
    assert_eq!(summary.external.l2_norm, 0.5);
    assert_eq!(summary.external.nonzero_count, 2);
    assert_eq!(summary.recurrent_excitatory.signed_sum, 0.4);
    assert_eq!(summary.recurrent_excitatory.l2_norm, 0.4);
    assert_eq!(summary.recurrent_excitatory.nonzero_count, 1);
    assert_eq!(summary.recurrent_inhibitory.signed_sum, -0.25);
    assert_eq!(summary.recurrent_inhibitory.l1, 0.25);
    assert_eq!(summary.recurrent_inhibitory.nonzero_count, 1);
    assert!((summary.recurrent_net.signed_sum - 0.15).abs() < 1e-15);
    assert!((summary.recurrent_net.l1 - 0.15).abs() < 1e-15);
    assert!((summary.recurrent_net.l2_norm - 0.15).abs() < 1e-15);
    assert_eq!(summary.recurrent_net.nonzero_count, 1);
}

#[test]
fn simultaneous_pre_post_uses_one_combined_delta_then_one_ceiling_clamp() {
    let mut cfg = config(3, 2);
    cfg.a_plus = 0.5;
    cfg.a_minus = 0.25;
    let mut state = NetworkState::reset(3, &cfg).expect("reset");
    state.pre_trace[0] = 1.0;
    state.post_trace[1] = 0.5;
    let mut weights = WeightMatrix::from_flat(
        3,
        2,
        vec![0.0, 0.99, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        vec![false, true, false, false, false, false, false, false, false],
    )
    .expect("near-ceiling edge");
    let adjacency = ImmutableAdjacency::from_weights(&weights).expect("adjacency");

    let result = run_streamed_episode_core(
        &mut state,
        &mut weights,
        &adjacency,
        &[vec![1.0, 1.0, 0.0]],
        1,
        true,
        &cfg,
    )
    .expect("simultaneous plastic step");

    assert_eq!(result.spike_indices, vec![vec![0, 1]]);
    assert_eq!(result.final_weights.get(0, 1), 1.0);
    assert_eq!(result.final_weights.get(0, 0), 0.0);
    assert_eq!(result.final_weights.get(1, 1), 0.0);
}

#[test]
fn connected_zero_edge_repotentiates_without_topology_change() {
    let cfg = config(3, 2);
    let mut state = NetworkState::reset(3, &cfg).expect("reset");
    state.pre_trace[0] = 1.0;
    let mut weights = WeightMatrix::from_flat(
        3,
        2,
        vec![0.0; 9],
        vec![false, true, false, false, false, false, false, false, false],
    )
    .expect("connected-zero edge");
    let adjacency = ImmutableAdjacency::from_weights(&weights).expect("adjacency");
    let original_topology = adjacency.topology().to_vec();

    let result = run_streamed_episode_core(
        &mut state,
        &mut weights,
        &adjacency,
        &[vec![0.0, 1.0, 0.0]],
        1,
        true,
        &cfg,
    )
    .expect("re-potentiation step");

    let expected = cfg.a_plus * (-cfg.dt_ms / cfg.tau_plus_ms).exp();
    assert_eq!(result.final_weights.get(0, 1), expected);
    assert_eq!(adjacency.topology(), original_topology);
}

#[test]
fn threshold_near_recurrence_uses_ascending_presynaptic_order() {
    let mut cfg = config(4, 3);
    cfg.excitatory_weight_max = 1.0e16;
    let mut state = NetworkState::reset(4, &cfg).expect("reset");
    state.spikes = vec![true, true, false, true];
    let mut values = vec![0.0; 16];
    let mut topology = vec![false; 16];
    values[2] = 1.0e16;
    values[6] = 1.0;
    values[14] = -1.0e16;
    topology[2] = true;
    topology[6] = true;
    topology[14] = true;
    let mut weights = WeightMatrix::from_flat(4, 3, values, topology).expect("ordered matrix");
    let adjacency = ImmutableAdjacency::from_weights(&weights).expect("adjacency");
    let mut packet = vec![0.0; 4];
    packet[2] = f64::from_bits(1.0_f64.to_bits() - 1);

    let result = run_streamed_episode_core(
        &mut state,
        &mut weights,
        &adjacency,
        &[packet],
        1,
        false,
        &cfg,
    )
    .expect("threshold-near episode");

    assert_eq!(result.current_summaries[0].recurrent_net.signed_sum, 0.0);
    assert!(!result.spike_indices[0].contains(&2));
}

#[test]
fn canonical_net_is_not_reassociated_from_separate_excitatory_inhibitory_sums() {
    let mut cfg = config(5, 1);
    cfg.excitatory_weight_max = 1.0e16;
    let mut state = NetworkState::reset(5, &cfg).expect("reset");
    state.spikes = vec![true, true, true, false, false];
    let mut values = vec![0.0; 25];
    let mut topology = vec![false; 25];
    values[4] = 1.0e16;
    values[9] = -1.0e16;
    values[14] = -1.0;
    topology[4] = true;
    topology[9] = true;
    topology[14] = true;
    let mut weights = WeightMatrix::from_flat(5, 1, values, topology).expect("ordered matrix");
    let adjacency = ImmutableAdjacency::from_weights(&weights).expect("adjacency");
    let mut packet = vec![0.0; 5];
    packet[4] = 1.0;

    let result = run_streamed_episode_core(
        &mut state,
        &mut weights,
        &adjacency,
        &[packet],
        1,
        false,
        &cfg,
    )
    .expect("association-sensitive episode");

    let summary = &result.current_summaries[0];
    assert_eq!(summary.recurrent_excitatory.signed_sum, 1.0e16);
    assert_eq!(summary.recurrent_inhibitory.signed_sum, -1.0e16);
    assert_eq!(summary.recurrent_net.signed_sum, -1.0);
    assert!(!result.spike_indices[0].contains(&4));
}

#[test]
fn membrane_recurrence_preserves_oracle_left_association() {
    let config = ModelConfig {
        dt_ms: 1.0,
        tau_m_ms: 1.0e-16,
        v_rest_mv: 0.0,
        v_reset_mv: -2.0,
        v_threshold_mv: 10.0,
        refractory_ms: 0.0,
        tau_plus_ms: 20.0,
        tau_minus_ms: 20.0,
        a_plus: 0.005,
        a_minus: 0.006,
        excitatory_weight_max: 1.0,
        excitatory_neurons: 1,
    };
    let mut state = NetworkState {
        voltage_mv: vec![-1.0, 0.0],
        refractory_steps: vec![0, 0],
        spikes: vec![false, true],
        pre_trace: vec![0.0, 0.0],
        post_trace: vec![0.0, 0.0],
    };
    let mut weights = WeightMatrix::from_flat(
        2,
        1,
        vec![0.0, 0.0, -1.0, 0.0],
        vec![false, false, true, false],
    )
    .expect("valid inhibitory edge");
    let adjacency = ImmutableAdjacency::from_weights(&weights).expect("adjacency");
    let result = run_streamed_episode_core(
        &mut state,
        &mut weights,
        &adjacency,
        &[vec![-1.0e16, 0.0]],
        1,
        false,
        &config,
    )
    .expect("adversarial finite episode");
    assert_eq!(result.final_state.voltage_mv[0], -2.0);
    assert_eq!(result.state_digests.len(), 1);
}

#[test]
fn finite_oracle_recurrence_is_not_rejected_by_grouped_current_overflow() {
    let config = ModelConfig {
        dt_ms: 0.5,
        tau_m_ms: 1.0,
        v_rest_mv: 0.0,
        v_reset_mv: -1.0,
        v_threshold_mv: 1.7e308,
        refractory_ms: 0.0,
        tau_plus_ms: 20.0,
        tau_minus_ms: 20.0,
        a_plus: 0.005,
        a_minus: 0.006,
        excitatory_weight_max: 1.0e308,
        excitatory_neurons: 1,
    };
    let mut state = NetworkState {
        voltage_mv: vec![0.0, 1.0e308],
        refractory_steps: vec![0, 0],
        spikes: vec![true, false],
        pre_trace: vec![0.0, 0.0],
        post_trace: vec![0.0, 0.0],
    };
    let mut weights = WeightMatrix::from_flat(
        2,
        1,
        vec![0.0, 1.0e308, 0.0, 0.0],
        vec![false, true, false, false],
    )
    .expect("valid extreme excitatory edge");
    let adjacency = ImmutableAdjacency::from_weights(&weights).expect("adjacency");
    let result = run_streamed_episode_core(
        &mut state,
        &mut weights,
        &adjacency,
        &[vec![0.0, 1.0e308]],
        1,
        false,
        &config,
    )
    .expect("ordered finite recurrence must be accepted");
    assert_eq!(result.final_state.voltage_mv[1], 1.5e308);
}

#[test]
fn cue_boundary_supports_all_completion_and_all_cue() {
    let cfg = config(3, 2);
    let packets = vec![vec![0.0; 3]; 2];
    let weights = matrix_three();
    let adjacency = ImmutableAdjacency::from_weights(&weights).expect("adjacency");

    let mut completion_state = NetworkState::reset(3, &cfg).expect("reset");
    let mut completion_weights = weights.clone();
    let completion = run_streamed_episode_core(
        &mut completion_state,
        &mut completion_weights,
        &adjacency,
        &packets,
        0,
        false,
        &cfg,
    )
    .expect("all completion");
    assert!(completion
        .current_summaries
        .iter()
        .all(|summary| summary.phase.as_str() == "completion"));

    let mut cue_state = NetworkState::reset(3, &cfg).expect("reset");
    let mut cue_weights = weights;
    let cue = run_streamed_episode_core(
        &mut cue_state,
        &mut cue_weights,
        &adjacency,
        &packets,
        packets.len(),
        false,
        &cfg,
    )
    .expect("all cue");
    assert!(cue
        .current_summaries
        .iter()
        .all(|summary| summary.phase == EpisodePhase::Cue));
}

#[test]
fn streamed_replay_is_bitwise_deterministic() {
    let cfg = config(3, 2);
    let state = NetworkState::reset(3, &cfg).expect("reset");
    let weights = matrix_three();
    let adjacency = ImmutableAdjacency::from_weights(&weights).expect("adjacency");
    let packets = vec![vec![1.0, 0.0, 0.0], vec![0.0, 0.6, 0.0], vec![0.0; 3]];

    let mut left_state = state.clone();
    let mut left_weights = weights.clone();
    let left = run_streamed_episode_core(
        &mut left_state,
        &mut left_weights,
        &adjacency,
        &packets,
        2,
        true,
        &cfg,
    )
    .expect("left replay");
    let mut right_state = state;
    let mut right_weights = weights;
    let right = run_streamed_episode_core(
        &mut right_state,
        &mut right_weights,
        &adjacency,
        &packets,
        2,
        true,
        &cfg,
    )
    .expect("right replay");

    assert_eq!(left, right);
}

#[test]
fn streamed_result_keeps_spikes_sparse_without_dense_state_history() {
    let cfg = config(128, 64);
    let mut state = NetworkState::reset(128, &cfg).expect("reset");
    let mut weights =
        WeightMatrix::from_flat(128, 64, vec![0.0; 128 * 128], vec![false; 128 * 128])
            .expect("empty topology");
    let adjacency = ImmutableAdjacency::from_weights(&weights).expect("adjacency");
    let mut packets = vec![vec![0.0; 128]; 16];
    packets[0][0] = 1.0;

    let result = run_streamed_episode_core(
        &mut state,
        &mut weights,
        &adjacency,
        &packets,
        1,
        false,
        &cfg,
    )
    .expect("sparse episode");

    assert_eq!(result.spike_indices.len(), packets.len());
    assert_eq!(result.spike_indices.iter().flatten().count(), 1);
    assert_eq!(result.current_summaries.len(), packets.len());
}

#[test]
fn full_pre_episode_validation_rejects_state_weights_packets_and_adjacency() {
    let cfg = config(3, 2);
    let base_weights = matrix_three();
    let adjacency = ImmutableAdjacency::from_weights(&base_weights).expect("adjacency");

    let mut bad_state = NetworkState::reset(3, &cfg).expect("reset");
    bad_state.pre_trace[0] = f64::NAN;
    let mut weights = base_weights.clone();
    assert!(run_streamed_episode_core(
        &mut bad_state,
        &mut weights,
        &adjacency,
        &[vec![0.0; 3]],
        0,
        false,
        &cfg,
    )
    .is_err());

    let mut state = NetworkState::reset(3, &cfg).expect("reset");
    let mut bad_weights = base_weights.clone();
    bad_weights.values[1] = f64::NAN;
    assert!(run_streamed_episode_core(
        &mut state,
        &mut bad_weights,
        &adjacency,
        &[vec![0.0; 3]],
        0,
        false,
        &cfg,
    )
    .is_err());

    let mut state = NetworkState::reset(3, &cfg).expect("reset");
    let mut weights = base_weights.clone();
    assert!(run_streamed_episode_core(
        &mut state,
        &mut weights,
        &adjacency,
        &[vec![f64::INFINITY, 0.0, 0.0]],
        0,
        false,
        &cfg,
    )
    .is_err());

    let other =
        WeightMatrix::from_flat(3, 2, vec![0.0; 9], vec![false; 9]).expect("different topology");
    let wrong_adjacency = ImmutableAdjacency::from_weights(&other).expect("adjacency");
    let mut state = NetworkState::reset(3, &cfg).expect("reset");
    let mut weights = base_weights;
    assert!(run_streamed_episode_core(
        &mut state,
        &mut weights,
        &wrong_adjacency,
        &[vec![0.0; 3]],
        0,
        false,
        &cfg,
    )
    .is_err());
}

#[test]
fn current_overflow_is_rejected_instead_of_emitted_as_infinite_evidence() {
    let mut cfg = config(4, 3);
    cfg.excitatory_weight_max = f64::MAX;
    let mut values = vec![0.0; 16];
    let mut topology = vec![false; 16];
    values[2] = f64::MAX;
    values[6] = f64::MAX;
    topology[2] = true;
    topology[6] = true;
    let mut weights = WeightMatrix::from_flat(4, 3, values, topology).expect("finite weights");
    let adjacency = ImmutableAdjacency::from_weights(&weights).expect("adjacency");
    let mut state = NetworkState::reset(4, &cfg).expect("reset");
    state.spikes[0] = true;
    state.spikes[1] = true;

    assert!(run_streamed_episode_core(
        &mut state,
        &mut weights,
        &adjacency,
        &[vec![0.0; 4]],
        0,
        false,
        &cfg,
    )
    .is_err());
}

#[test]
fn finite_external_and_recurrent_entries_that_overflow_total_are_rejected() {
    let mut cfg = config(3, 2);
    cfg.excitatory_weight_max = f64::MAX;
    let mut state = NetworkState::reset(3, &cfg).expect("reset");
    state.spikes[0] = true;
    let mut weights = WeightMatrix::from_flat(
        3,
        2,
        vec![0.0, f64::MAX, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        vec![false, true, false, false, false, false, false, false, false],
    )
    .expect("finite maximum weight");
    let adjacency = ImmutableAdjacency::from_weights(&weights).expect("adjacency");

    assert!(run_streamed_episode_core(
        &mut state,
        &mut weights,
        &adjacency,
        &[vec![0.0, f64::MAX, 0.0]],
        0,
        false,
        &cfg,
    )
    .is_err());
}

#[test]
fn finite_packet_entries_with_overflowing_summary_are_rejected() {
    let cfg = config(3, 2);
    let mut state = NetworkState::reset(3, &cfg).expect("reset");
    let mut weights = matrix_three();
    let adjacency = ImmutableAdjacency::from_weights(&weights).expect("adjacency");

    assert!(run_streamed_episode_core(
        &mut state,
        &mut weights,
        &adjacency,
        &[vec![f64::MAX, f64::MAX, 0.0]],
        0,
        false,
        &cfg,
    )
    .is_err());
}

#[test]
fn nonfinite_combined_plasticity_delta_is_rejected() {
    let mut cfg = config(3, 2);
    cfg.a_plus = f64::MAX;
    let mut state = NetworkState::reset(3, &cfg).expect("reset");
    state.pre_trace[0] = f64::MAX;
    let mut weights = WeightMatrix::from_flat(
        3,
        2,
        vec![0.0; 9],
        vec![false, true, false, false, false, false, false, false, false],
    )
    .expect("connected edge");
    let adjacency = ImmutableAdjacency::from_weights(&weights).expect("adjacency");

    assert!(run_streamed_episode_core(
        &mut state,
        &mut weights,
        &adjacency,
        &[vec![0.0, 1.0, 0.0]],
        1,
        true,
        &cfg,
    )
    .is_err());
}

#[test]
fn ordered_hypot_l2_stays_finite_when_naive_squares_would_overflow() {
    let cfg = config(3, 2);
    let mut state = NetworkState::reset(3, &cfg).expect("reset");
    let mut weights = matrix_three();
    let adjacency = ImmutableAdjacency::from_weights(&weights).expect("adjacency");
    let large = f64::MAX / 4.0;

    let result = run_streamed_episode_core(
        &mut state,
        &mut weights,
        &adjacency,
        &[vec![large, large, 0.0]],
        1,
        false,
        &cfg,
    )
    .expect("stable summary");

    let summary = &result.current_summaries[0].external;
    assert!(summary.l2_norm.is_finite());
    assert!(summary.l2_norm > large);
}

#[test]
fn full_post_episode_state_validation_rejects_generated_nonfinite_voltage() {
    let mut cfg = config(3, 2);
    cfg.dt_ms = 2.0;
    let mut state = NetworkState::reset(3, &cfg).expect("reset");
    let mut weights = matrix_three();
    let adjacency = ImmutableAdjacency::from_weights(&weights).expect("adjacency");

    assert!(run_streamed_episode_core(
        &mut state,
        &mut weights,
        &adjacency,
        &[vec![-f64::MAX, 0.0, 0.0]],
        0,
        false,
        &cfg,
    )
    .is_err());
}

#[test]
fn cue_boundary_and_empty_episode_contracts_fail_closed() {
    let cfg = config(3, 2);
    let mut state = NetworkState::reset(3, &cfg).expect("reset");
    let mut weights = matrix_three();
    let adjacency = ImmutableAdjacency::from_weights(&weights).expect("adjacency");
    assert!(run_streamed_episode_core(
        &mut state,
        &mut weights,
        &adjacency,
        &[vec![0.0; 3]],
        2,
        false,
        &cfg,
    )
    .is_err());
    assert!(
        run_streamed_episode_core(&mut state, &mut weights, &adjacency, &[], 0, false, &cfg,)
            .is_err()
    );
    assert!(run_streamed_episode_core(
        &mut state,
        &mut weights,
        &adjacency,
        &[vec![0.0; 2]],
        0,
        false,
        &cfg,
    )
    .is_err());

    let mut larger_state = NetworkState::reset(4, &cfg).expect("larger reset");
    assert!(run_streamed_episode_core(
        &mut larger_state,
        &mut weights,
        &adjacency,
        &[vec![0.0; 4]],
        0,
        false,
        &cfg,
    )
    .is_err());
}
