// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Remanentia — Temporal SNN memory Python extension

//! Deterministic row-pre/column-post LIF and online STDP primitives.

#![deny(missing_docs)]

mod config;
mod engine;
mod python_stream;
mod state;
mod streamed;

use std::error::Error;
use std::fmt::{Display, Formatter};

pub use config::ModelConfig;
pub use engine::{
    probe_episode_core, run_episode_core, step_lif_core, step_stdp_core, EpisodeResult, StepResult,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
pub use state::{NetworkState, WeightMatrix};
pub use streamed::{
    run_streamed_episode_core, state_digest, CurrentSummary, CurrentVectorSummary, EpisodePhase,
    ImmutableAdjacency, StreamedEpisodeResult,
};

/// Domain error returned when an SNN contract is invalid.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EngineError {
    /// Inconsistent vector or matrix dimensions.
    Shape(&'static str),
    /// A non-finite value was supplied to a numerical surface.
    NonFinite(&'static str),
    /// State violates a declared network invariant.
    InvalidState(&'static str),
    /// Configuration values are internally inconsistent.
    InvalidConfig(&'static str),
}

impl Display for EngineError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        let (kind, detail) = match self {
            Self::Shape(detail) => ("shape error", detail),
            Self::NonFinite(detail) => ("non-finite value", detail),
            Self::InvalidState(detail) => ("invalid state", detail),
            Self::InvalidConfig(detail) => ("invalid configuration", detail),
        };
        write!(formatter, "{kind}: {detail}")
    }
}

impl Error for EngineError {}

impl From<EngineError> for PyErr {
    fn from(error: EngineError) -> Self {
        PyValueError::new_err(error.to_string())
    }
}

/// Result type shared by Rust engine primitives.
pub type EngineResult<T> = Result<T, EngineError>;

/// Advance LIF membrane state using externally prepared total input current.
#[pyfunction(name = "step_lif")]
fn py_step_lif(
    state: &mut NetworkState,
    input_current: Vec<f64>,
    config: &ModelConfig,
) -> PyResult<StepResult> {
    step_lif_core(state, &input_current, config).map_err(Into::into)
}

/// Apply an online STDP update to E-to-E synapses and advance traces.
#[pyfunction(name = "step_stdp")]
fn py_step_stdp(
    weights: &WeightMatrix,
    pre_trace: Vec<f64>,
    post_trace: Vec<f64>,
    spikes: Vec<bool>,
    config: &ModelConfig,
) -> PyResult<(WeightMatrix, Vec<f64>, Vec<f64>)> {
    let mut updated_weights = weights.clone();
    let mut updated_pre_trace = pre_trace;
    let mut updated_post_trace = post_trace;
    step_stdp_core(
        &mut updated_weights,
        &mut updated_pre_trace,
        &mut updated_post_trace,
        &spikes,
        config,
    )?;
    Ok((updated_weights, updated_pre_trace, updated_post_trace))
}

/// Run one episode while explicitly selecting whether online plasticity is active.
#[pyfunction(name = "run_episode")]
fn py_run_episode(
    state: &NetworkState,
    weights: &WeightMatrix,
    spike_packets: Vec<Vec<f64>>,
    plasticity_enabled: bool,
    config: &ModelConfig,
) -> PyResult<EpisodeResult> {
    let mut episode_state = state.clone();
    let mut episode_weights = weights.clone();
    run_episode_core(
        &mut episode_state,
        &mut episode_weights,
        &spike_packets,
        plasticity_enabled,
        config,
    )
    .map_err(Into::into)
}

/// Probe a frozen checkpoint clone without modifying its state or weights.
#[pyfunction(name = "probe_episode")]
fn py_probe_episode(
    frozen_state: &NetworkState,
    frozen_weights: &WeightMatrix,
    cue_packets: Vec<Vec<f64>>,
    config: &ModelConfig,
) -> PyResult<EpisodeResult> {
    probe_episode_core(frozen_state, frozen_weights, &cue_packets, config).map_err(Into::into)
}

/// Register the temporal SNN memory extension module.
#[pymodule]
fn rust_snn_memory(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<ModelConfig>()?;
    module.add_class::<NetworkState>()?;
    module.add_class::<WeightMatrix>()?;
    module.add_class::<StepResult>()?;
    module.add_class::<EpisodeResult>()?;
    module.add_function(wrap_pyfunction!(py_step_lif, module)?)?;
    module.add_function(wrap_pyfunction!(py_step_stdp, module)?)?;
    module.add_function(wrap_pyfunction!(py_run_episode, module)?)?;
    module.add_function(wrap_pyfunction!(py_probe_episode, module)?)?;
    python_stream::register_streamed(module)?;
    Ok(())
}
