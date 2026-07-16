// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Remanentia — Temporal SNN public Rust API tests

use rust_snn_memory::{
    probe_episode_core, run_episode_core, step_lif_core, step_stdp_core, EngineError, ModelConfig,
    NetworkState, WeightMatrix,
};

fn config() -> ModelConfig {
    ModelConfig {
        dt_ms: 1.0,
        tau_m_ms: 1.0,
        v_rest_mv: 0.0,
        v_reset_mv: 0.0,
        v_threshold_mv: 1.0,
        refractory_ms: 2.0,
        tau_plus_ms: 20.0,
        tau_minus_ms: 20.0,
        a_plus: 0.005,
        a_minus: 0.006,
        excitatory_weight_max: 1.0,
        excitatory_neurons: 2,
    }
}

fn weights() -> WeightMatrix {
    WeightMatrix::from_flat(
        3,
        2,
        vec![0.0, 0.4, 0.3, 0.2, 0.0, 0.1, -0.5, -0.25, 0.0],
        vec![false, true, true, true, false, true, true, true, false],
    )
    .expect("valid Dale-compliant fixture")
}

#[test]
fn step_lif_emits_spike_and_enforces_refractory_period() {
    let cfg = config();
    let mut state = NetworkState::reset(3, &cfg).expect("valid reset state");

    let first = step_lif_core(&mut state, &[1.0, 0.0, 0.0], &cfg).expect("first step");
    assert_eq!(first.spike_indices, vec![0]);
    assert_eq!(first.voltage_mv[0], cfg.v_reset_mv);
    assert_eq!(first.refractory_steps[0], 2);

    let second = step_lif_core(&mut state, &[10.0, 0.0, 0.0], &cfg).expect("second step");
    assert!(second.spike_indices.is_empty());
    assert_eq!(second.refractory_steps[0], 1);
}

#[test]
fn stdp_uses_decayed_old_traces_for_causal_ltp_and_reverse_ltd() {
    let cfg = config();
    let mut causal = weights();
    let original_01 = causal.get(0, 1);
    let mut pre_trace = vec![1.0, 0.0, 0.0];
    let mut post_trace = vec![0.0; 3];

    step_stdp_core(
        &mut causal,
        &mut pre_trace,
        &mut post_trace,
        &[false, true, false],
        &cfg,
    )
    .expect("valid causal STDP step");

    let decay = (-cfg.dt_ms / cfg.tau_plus_ms).exp();
    assert_eq!(causal.get(0, 1), original_01 + cfg.a_plus * decay);

    let mut reverse = weights();
    let original_10 = reverse.get(1, 0);
    let mut pre_trace = vec![0.0; 3];
    let mut post_trace = vec![1.0, 0.0, 0.0];
    step_stdp_core(
        &mut reverse,
        &mut pre_trace,
        &mut post_trace,
        &[false, true, false],
        &cfg,
    )
    .expect("valid reverse STDP step");

    assert_eq!(reverse.get(1, 0), original_10 - cfg.a_minus * decay);
    assert_eq!(reverse.get(2, 0), -0.5, "inhibitory rows remain fixed");
    assert_eq!(reverse.get(0, 0), 0.0, "diagonal remains zero");
}

#[test]
fn episode_propagates_row_pre_weight_on_following_timestep() {
    let cfg = config();
    let mut state = NetworkState::reset(3, &cfg).expect("valid reset state");
    let mut matrix = weights();
    let packets = vec![vec![1.0, 0.0, 0.0], vec![0.0, 0.6, 0.0]];

    let result =
        run_episode_core(&mut state, &mut matrix, &packets, false, &cfg).expect("valid episode");

    assert_eq!(result.spike_indices[0], vec![0]);
    assert_eq!(result.spike_indices[1], vec![1]);
}

#[test]
fn probe_preserves_frozen_checkpoint_and_disables_plasticity() {
    let cfg = config();
    let state = NetworkState::reset(3, &cfg).expect("valid reset state");
    let matrix = weights();
    let original_state = state.clone();
    let original_weights = matrix.clone();

    let result = probe_episode_core(&state, &matrix, &[vec![1.0, 0.0, 0.0], vec![0.0; 3]], &cfg)
        .expect("valid frozen probe");

    assert_eq!(state, original_state);
    assert_eq!(matrix, original_weights);
    assert_eq!(result.final_weights, original_weights);
}

#[test]
fn contracts_reject_shape_nonfinite_and_dale_violations() {
    assert!(WeightMatrix::from_flat(2, 1, vec![0.0, 1.0], vec![false; 4]).is_err());
    assert!(WeightMatrix::from_flat(
        2,
        1,
        vec![0.0, -1.0, -1.0, 0.0],
        vec![false, true, true, false],
    )
    .is_err());

    let cfg = config();
    let mut state = NetworkState::reset(3, &cfg).expect("valid reset state");
    assert!(step_lif_core(&mut state, &[f64::NAN, 0.0, 0.0], &cfg).is_err());
}

#[test]
fn model_contract_rejects_invalid_configuration_and_weight_bound() {
    let mut cfg = config();
    cfg.dt_ms = 0.0;
    assert!(NetworkState::reset(3, &cfg).is_err());

    let cfg = config();
    let oversized = WeightMatrix::from_flat(
        3,
        2,
        vec![0.0, 2.1, 0.3, 0.2, 0.0, 0.1, -0.5, -0.25, 0.0],
        vec![false, true, true, true, false, true, true, true, false],
    )
    .expect("Dale-compliant matrix can be checked against a model separately");
    assert!(oversized.validate_for_config(&cfg).is_err());

    let inhibitory_oversized = WeightMatrix::from_flat(
        3,
        2,
        vec![0.0, 0.4, 0.3, 0.2, 0.0, 0.1, -1.1, -0.25, 0.0],
        vec![false, true, true, true, false, true, true, true, false],
    )
    .expect("Dale-compliant inhibitory magnitude is checked against the model");
    assert!(inhibitory_oversized.validate_for_config(&cfg).is_err());

    let mut fractional = cfg.clone();
    fractional.refractory_ms = 2.5;
    assert_eq!(fractional.refractory_steps(), 3);
    fractional.refractory_ms = 0.4;
    assert_eq!(fractional.refractory_steps(), 1);
    fractional.refractory_ms = 0.0;
    assert_eq!(fractional.refractory_steps(), 0);
}

#[test]
fn simultaneous_spikes_with_empty_traces_preserve_weights() {
    let cfg = config();
    let mut matrix = weights();
    let original = matrix.clone();
    let mut pre_trace = vec![0.0; 3];
    let mut post_trace = vec![0.0; 3];

    step_stdp_core(
        &mut matrix,
        &mut pre_trace,
        &mut post_trace,
        &[true, true, false],
        &cfg,
    )
    .expect("simultaneous step");

    assert_eq!(matrix, original);
    assert_eq!(pre_trace, vec![1.0, 1.0, 0.0]);
    assert_eq!(post_trace, vec![1.0, 1.0, 0.0]);
}

#[test]
fn connected_zero_weight_can_potentiate_without_changing_topology() {
    let cfg = config();
    let mut matrix = WeightMatrix::from_flat(
        3,
        2,
        vec![0.0, 0.0, 0.0, 0.2, 0.0, 0.0, -0.5, 0.0, 0.0],
        vec![false, true, false, true, false, false, true, false, false],
    )
    .expect("connected zero-weight synapse is valid");
    let original_topology = matrix.topology().to_vec();
    let mut pre_trace = vec![1.0, 0.0, 0.0];
    let mut post_trace = vec![0.0; 3];

    step_stdp_core(
        &mut matrix,
        &mut pre_trace,
        &mut post_trace,
        &[false, true, false],
        &cfg,
    )
    .expect("zero-weight connected synapse can recover");

    let decay = (-cfg.dt_ms / cfg.tau_plus_ms).exp();
    assert_eq!(matrix.get(0, 1), cfg.a_plus * decay);
    assert_eq!(matrix.topology(), original_topology);
}

#[test]
fn configuration_contract_rejects_each_invalid_parameter_class() {
    let mut cfg = config();
    cfg.a_plus = f64::NAN;
    assert!(cfg.validate(3).is_err());

    let mut cfg = config();
    cfg.a_minus = -0.1;
    assert!(cfg.validate(3).is_err());

    let mut cfg = config();
    cfg.v_rest_mv = cfg.v_threshold_mv;
    assert!(cfg.validate(3).is_err());

    let cfg = config();
    assert!(cfg.validate(0).is_err());

    let mut cfg = config();
    cfg.refractory_ms = f64::from(u32::MAX);
    cfg.dt_ms = 1.0;
    assert!(cfg.validate(3).is_ok());
    assert_eq!(cfg.refractory_steps(), u32::MAX);
    cfg.refractory_ms = f64::from(u32::MAX) + 1.0;
    assert!(cfg.validate(3).is_err());
    cfg.refractory_ms = 1.0e308;
    cfg.dt_ms = 1.0e-308;
    assert!(cfg.validate(3).is_err());
    assert_eq!(
        EngineError::Shape("mismatch").to_string(),
        "shape error: mismatch"
    );
    assert_eq!(
        EngineError::InvalidConfig("parameter").to_string(),
        "invalid configuration: parameter"
    );
}

#[test]
fn model_contract_rejects_empty_excitatory_or_inhibitory_population() {
    let mut cfg = config();
    cfg.excitatory_neurons = 0;
    assert!(cfg.validate(3).is_err());

    cfg.excitatory_neurons = 3;
    assert!(cfg.validate(3).is_err());

    assert!(WeightMatrix::from_flat(2, 0, vec![0.0; 4], vec![false; 4]).is_err());
    assert!(WeightMatrix::from_flat(2, 2, vec![0.0; 4], vec![false; 4]).is_err());
}

#[test]
fn state_and_weight_contracts_reject_corrupt_persisted_values() {
    let cfg = config();
    let malformed_state = NetworkState {
        voltage_mv: vec![0.0; 3],
        refractory_steps: vec![0; 2],
        spikes: vec![false; 3],
        pre_trace: vec![0.0; 3],
        post_trace: vec![0.0; 3],
    };
    assert!(malformed_state.validate().is_err());

    let mut nonfinite_state = NetworkState::reset(3, &cfg).expect("valid state");
    nonfinite_state.post_trace[1] = f64::INFINITY;
    assert!(nonfinite_state.validate().is_err());

    let topology = vec![false, true, true, false];
    assert!(
        WeightMatrix::from_flat(2, 1, vec![0.0, f64::NAN, -0.1, 0.0], topology.clone(),).is_err()
    );
    assert!(WeightMatrix::from_flat(2, 1, vec![0.0, 0.2, -0.1, 0.0], vec![false; 4],).is_err());
    assert!(WeightMatrix::from_flat(
        2,
        1,
        vec![0.0, 0.2, -0.1, 0.0],
        vec![true, true, true, false],
    )
    .is_err());
    assert!(WeightMatrix::from_flat(2, 1, vec![0.1, 0.2, -0.1, 0.0], topology,).is_err());

    let mut mismatched_cfg = cfg.clone();
    mismatched_cfg.excitatory_neurons = 1;
    assert!(weights().validate_for_config(&mismatched_cfg).is_err());
}

#[test]
fn timestep_and_episode_contracts_reject_mismatched_inputs() {
    let cfg = config();
    let mut state = NetworkState::reset(3, &cfg).expect("valid state");
    assert!(step_lif_core(&mut state, &[0.0; 2], &cfg).is_err());

    let mut matrix = weights();
    assert!(run_episode_core(&mut state, &mut matrix, &[], false, &cfg).is_err());
    assert!(step_stdp_core(&mut matrix, &mut [0.0; 2], &mut [0.0; 3], &[false; 3], &cfg,).is_err());
    assert!(step_stdp_core(
        &mut matrix,
        &mut [f64::INFINITY, 0.0, 0.0],
        &mut [0.0; 3],
        &[false; 3],
        &cfg,
    )
    .is_err());

    let mut wrong_size_matrix = WeightMatrix::from_flat(
        2,
        1,
        vec![0.0, 0.2, -0.1, 0.0],
        vec![false, true, true, false],
    )
    .expect("valid 2x2 matrix");
    assert!(run_episode_core(&mut state, &mut wrong_size_matrix, &[], false, &cfg,).is_err());

    let mut matrix = weights();
    assert!(run_episode_core(&mut state, &mut matrix, &[vec![0.0; 2]], false, &cfg,).is_err());
}

#[test]
fn plastic_episode_updates_weights_through_the_public_runner() {
    let cfg = config();
    let mut state = NetworkState::reset(3, &cfg).expect("valid state");
    state.pre_trace[0] = 1.0;
    let mut matrix = weights();
    let original = matrix.get(0, 1);

    let result = run_episode_core(&mut state, &mut matrix, &[vec![0.0, 1.0, 0.0]], true, &cfg)
        .expect("plastic episode");

    assert_eq!(result.spike_indices, vec![vec![1]]);
    let decay = (-cfg.dt_ms / cfg.tau_plus_ms).exp();
    assert_eq!(
        result.final_weights.get(0, 1),
        original + cfg.a_plus * decay
    );
}
