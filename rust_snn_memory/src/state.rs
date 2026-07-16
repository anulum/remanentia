// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Remanentia — Temporal SNN state and weight contracts

use pyo3::prelude::*;

use crate::{EngineError, EngineResult, ModelConfig};

/// Mutable LIF and trace state for one network.
#[pyclass(get_all)]
#[derive(Clone, Debug, PartialEq)]
pub struct NetworkState {
    /// Membrane potential per neuron in millivolts.
    pub voltage_mv: Vec<f64>,
    /// Remaining refractory timesteps per neuron.
    pub refractory_steps: Vec<u32>,
    /// Binary spikes emitted by the preceding timestep.
    pub spikes: Vec<bool>,
    /// Decaying presynaptic traces.
    pub pre_trace: Vec<f64>,
    /// Decaying postsynaptic traces.
    pub post_trace: Vec<f64>,
}

impl NetworkState {
    /// Create a reset network state at the declared resting potential.
    pub fn reset(neurons: usize, config: &ModelConfig) -> EngineResult<Self> {
        config.validate(neurons)?;
        Ok(Self {
            voltage_mv: vec![config.v_rest_mv; neurons],
            refractory_steps: vec![0; neurons],
            spikes: vec![false; neurons],
            pre_trace: vec![0.0; neurons],
            post_trace: vec![0.0; neurons],
        })
    }

    /// Return the population size represented by the state.
    pub fn neurons(&self) -> usize {
        self.voltage_mv.len()
    }

    /// Validate vector lengths, finite state, and binary spike values.
    pub fn validate(&self) -> EngineResult<()> {
        let neurons = self.neurons();
        if neurons == 0
            || self.refractory_steps.len() != neurons
            || self.spikes.len() != neurons
            || self.pre_trace.len() != neurons
            || self.post_trace.len() != neurons
        {
            return Err(EngineError::Shape(
                "state vectors must have equal non-zero length",
            ));
        }
        if self
            .voltage_mv
            .iter()
            .chain(self.pre_trace.iter())
            .chain(self.post_trace.iter())
            .any(|value| !value.is_finite())
        {
            return Err(EngineError::NonFinite("network state"));
        }
        Ok(())
    }
}

#[pymethods]
impl NetworkState {
    /// Construct reset state for a population.
    #[new]
    fn py_new(neurons: usize, config: &ModelConfig) -> PyResult<Self> {
        Self::reset(neurons, config).map_err(Into::into)
    }
}

/// Dense row-pre/column-post recurrent weight matrix.
#[pyclass(get_all)]
#[derive(Clone, Debug, PartialEq)]
pub struct WeightMatrix {
    /// Matrix dimension.
    pub neurons: usize,
    /// Excitatory population size used by Dale and plasticity checks.
    pub excitatory_neurons: usize,
    /// Row-major values, indexed as `pre * neurons + post`.
    pub values: Vec<f64>,
    /// Immutable row-major connectivity mask, independent of current weight value.
    topology: Vec<bool>,
}

impl WeightMatrix {
    /// Construct and validate a row-pre/column-post matrix.
    pub fn from_flat(
        neurons: usize,
        excitatory_neurons: usize,
        values: Vec<f64>,
        topology: Vec<bool>,
    ) -> EngineResult<Self> {
        let matrix = Self {
            neurons,
            excitatory_neurons,
            values,
            topology,
        };
        matrix.validate()?;
        Ok(matrix)
    }

    /// Return a synapse value using the declared `W[pre, post]` convention.
    pub fn get(&self, pre: usize, post: usize) -> f64 {
        self.values[pre * self.neurons + post]
    }

    /// Set a synapse value using the declared `W[pre, post]` convention.
    pub fn set(&mut self, pre: usize, post: usize, value: f64) {
        self.values[pre * self.neurons + post] = value;
    }

    /// Return whether the immutable topology contains a synapse.
    pub fn is_connected(&self, pre: usize, post: usize) -> bool {
        self.topology[pre * self.neurons + post]
    }

    /// Borrow the immutable row-major connectivity mask.
    pub fn topology(&self) -> &[bool] {
        &self.topology
    }

    /// Validate shape, finiteness, zero diagonal, and Dale signs.
    pub fn validate(&self) -> EngineResult<()> {
        if self.neurons < 2
            || self.excitatory_neurons == 0
            || self.excitatory_neurons >= self.neurons
            || self.values.len() != self.neurons * self.neurons
            || self.topology.len() != self.neurons * self.neurons
        {
            return Err(EngineError::Shape(
                "weight values must form a square matrix with excitatory and inhibitory populations",
            ));
        }
        if self.values.iter().any(|value| !value.is_finite()) {
            return Err(EngineError::NonFinite("weight matrix"));
        }
        for pre in 0..self.neurons {
            for post in 0..self.neurons {
                let value = self.get(pre, post);
                let connected = self.is_connected(pre, post);
                if pre == post && (value != 0.0 || connected) {
                    return Err(EngineError::InvalidState(
                        "self-connections must have zero weight and absent topology",
                    ));
                }
                if !connected && value != 0.0 {
                    return Err(EngineError::InvalidState(
                        "weights outside the immutable topology must be zero",
                    ));
                }
                if (pre < self.excitatory_neurons && value < 0.0)
                    || (pre >= self.excitatory_neurons && value > 0.0)
                {
                    return Err(EngineError::InvalidState(
                        "presynaptic rows must obey Dale signs",
                    ));
                }
            }
        }
        Ok(())
    }

    /// Validate matrix population metadata and excitatory bounds against a model.
    pub fn validate_for_config(&self, config: &ModelConfig) -> EngineResult<()> {
        self.validate()?;
        config.validate(self.neurons)?;
        if self.excitatory_neurons != config.excitatory_neurons {
            return Err(EngineError::Shape(
                "weight and configuration excitatory populations must match",
            ));
        }
        for pre in 0..self.excitatory_neurons {
            for post in 0..self.neurons {
                if self.get(pre, post) > config.excitatory_weight_max {
                    return Err(EngineError::InvalidState(
                        "excitatory weights must not exceed the configured bound",
                    ));
                }
            }
        }
        for pre in self.excitatory_neurons..self.neurons {
            for post in 0..self.neurons {
                if self.get(pre, post) < -config.excitatory_weight_max {
                    return Err(EngineError::InvalidState(
                        "inhibitory weights must not exceed the configured magnitude",
                    ));
                }
            }
        }
        Ok(())
    }
}

#[pymethods]
impl WeightMatrix {
    /// Construct a validated dense row-pre/column-post matrix.
    #[new]
    fn py_new(
        neurons: usize,
        excitatory_neurons: usize,
        values: Vec<f64>,
        topology: Vec<bool>,
    ) -> PyResult<Self> {
        Self::from_flat(neurons, excitatory_neurons, values, topology).map_err(Into::into)
    }
}
