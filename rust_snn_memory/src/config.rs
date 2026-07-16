// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Remanentia — Temporal SNN model configuration

use pyo3::prelude::*;

use crate::{EngineError, EngineResult};

/// Numerical and plasticity parameters for one temporal SNN network.
#[pyclass(get_all)]
#[derive(Clone, Debug, PartialEq)]
pub struct ModelConfig {
    /// Simulation timestep in milliseconds.
    pub dt_ms: f64,
    /// Membrane time constant in milliseconds.
    pub tau_m_ms: f64,
    /// Resting membrane potential in millivolts.
    pub v_rest_mv: f64,
    /// Reset membrane potential in millivolts.
    pub v_reset_mv: f64,
    /// Spike threshold in millivolts.
    pub v_threshold_mv: f64,
    /// Absolute refractory duration in milliseconds.
    pub refractory_ms: f64,
    /// Presynaptic trace decay constant in milliseconds.
    pub tau_plus_ms: f64,
    /// Postsynaptic trace decay constant in milliseconds.
    pub tau_minus_ms: f64,
    /// Causal potentiation amplitude.
    pub a_plus: f64,
    /// Anti-causal depression amplitude.
    pub a_minus: f64,
    /// Maximum magnitude of an excitatory synapse.
    pub excitatory_weight_max: f64,
    /// Number of excitatory neurons at the beginning of the population.
    pub excitatory_neurons: usize,
}

impl ModelConfig {
    /// Validate finite values, positive time constants, and threshold ordering.
    pub fn validate(&self, neurons: usize) -> EngineResult<()> {
        let finite = [
            self.dt_ms,
            self.tau_m_ms,
            self.v_rest_mv,
            self.v_reset_mv,
            self.v_threshold_mv,
            self.refractory_ms,
            self.tau_plus_ms,
            self.tau_minus_ms,
            self.a_plus,
            self.a_minus,
            self.excitatory_weight_max,
        ];
        if finite.iter().any(|value| !value.is_finite()) {
            return Err(EngineError::InvalidConfig(
                "all numeric parameters must be finite",
            ));
        }
        if self.dt_ms <= 0.0
            || self.tau_m_ms <= 0.0
            || self.tau_plus_ms <= 0.0
            || self.tau_minus_ms <= 0.0
        {
            return Err(EngineError::InvalidConfig(
                "dt and time constants must be positive",
            ));
        }
        if self.refractory_ms < 0.0
            || self.a_plus < 0.0
            || self.a_minus < 0.0
            || self.excitatory_weight_max <= 0.0
        {
            return Err(EngineError::InvalidConfig(
                "refractory duration and learning amplitudes must be non-negative, and weight maximum positive",
            ));
        }
        let refractory_ratio = self.refractory_ms / self.dt_ms;
        if !refractory_ratio.is_finite() || refractory_ratio.ceil() > f64::from(u32::MAX) {
            return Err(EngineError::InvalidConfig(
                "refractory duration exceeds the u32 timestep domain",
            ));
        }
        if self.v_reset_mv >= self.v_threshold_mv || self.v_rest_mv >= self.v_threshold_mv {
            return Err(EngineError::InvalidConfig(
                "reset and resting potentials must be below threshold",
            ));
        }
        if neurons < 2 || self.excitatory_neurons == 0 || self.excitatory_neurons >= neurons {
            return Err(EngineError::InvalidConfig(
                "population must contain both excitatory and inhibitory neurons",
            ));
        }
        Ok(())
    }

    /// Convert the refractory duration to an integral number of timesteps.
    pub fn refractory_steps(&self) -> u32 {
        if self.refractory_ms <= 0.0 {
            0
        } else {
            (self.refractory_ms / self.dt_ms).ceil() as u32
        }
    }
}

#[pymethods]
impl ModelConfig {
    /// Construct the declared LIF/STDP model configuration.
    #[new]
    #[pyo3(signature = (
        excitatory_neurons,
        dt_ms=1.0,
        tau_m_ms=20.0,
        v_rest_mv=-65.0,
        v_reset_mv=-70.0,
        v_threshold_mv=-55.0,
        refractory_ms=2.0,
        tau_plus_ms=20.0,
        tau_minus_ms=20.0,
        a_plus=0.005,
        a_minus=0.006,
        excitatory_weight_max=1.0
    ))]
    #[allow(clippy::too_many_arguments)]
    fn py_new(
        excitatory_neurons: usize,
        dt_ms: f64,
        tau_m_ms: f64,
        v_rest_mv: f64,
        v_reset_mv: f64,
        v_threshold_mv: f64,
        refractory_ms: f64,
        tau_plus_ms: f64,
        tau_minus_ms: f64,
        a_plus: f64,
        a_minus: f64,
        excitatory_weight_max: f64,
    ) -> Self {
        Self {
            dt_ms,
            tau_m_ms,
            v_rest_mv,
            v_reset_mv,
            v_threshold_mv,
            refractory_ms,
            tau_plus_ms,
            tau_minus_ms,
            a_plus,
            a_minus,
            excitatory_weight_max,
            excitatory_neurons,
        }
    }
}
