// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Remanentia — NumPy boundary for sparse streamed SNN episodes

//! Transactional PyO3 binding for the Stage-1 streamed Rust core.

use numpy::{
    Element, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::{
    run_streamed_episode_core, CurrentSummary, EpisodePhase, ImmutableAdjacency, ModelConfig,
    NetworkState, StreamedEpisodeResult, WeightMatrix,
};

const STREAMED_API_VERSION: u32 = 2;
const TOPOLOGY_DIGEST_FRAMING_VERSION: u32 = 2;
const STATE_DIGEST_FRAMING_VERSION: u32 = 2;

/// Read-only NumPy-backed streamed result exposed to Python.
#[pyclass(name = "StreamedEpisodeV2", frozen)]
pub struct PyStreamedEpisodeV2 {
    spike_offsets: Py<PyArray1<u64>>,
    spike_indices: Py<PyArray1<u64>>,
    phases: Py<PyArray1<u8>>,
    current_values: Py<PyArray2<f64>>,
    current_nonzero_counts: Py<PyArray2<u64>>,
    state_digests: Vec<String>,
    final_voltage_mv: Py<PyArray1<f64>>,
    final_refractory_steps: Py<PyArray1<u32>>,
    final_spikes: Py<PyArray1<bool>>,
    final_pre_trace: Py<PyArray1<f64>>,
    final_post_trace: Py<PyArray1<f64>>,
    final_weights: Py<PyArray2<f64>>,
    topology: Py<PyArray2<bool>>,
    outgoing_offsets: Py<PyArray1<u64>>,
    outgoing_indices: Py<PyArray1<u64>>,
    incoming_offsets: Py<PyArray1<u64>>,
    incoming_indices: Py<PyArray1<u64>>,
    topology_digest: String,
}

#[pymethods]
impl PyStreamedEpisodeV2 {
    /// CSR timestep offsets for sparse spike indices.
    #[getter]
    fn spike_offsets(&self, py: Python<'_>) -> Py<PyArray1<u64>> {
        self.spike_offsets.clone_ref(py)
    }

    /// Flattened sparse spike neuron indices.
    #[getter]
    fn spike_indices(&self, py: Python<'_>) -> Py<PyArray1<u64>> {
        self.spike_indices.clone_ref(py)
    }

    /// Per-timestep phases: 0 cue, 1 completion.
    #[getter]
    fn phases(&self, py: Python<'_>) -> Py<PyArray1<u8>> {
        self.phases.clone_ref(py)
    }

    /// f64 columns ext/E/I/net x signed-sum/L1/Euclidean-L2.
    #[getter]
    fn current_values(&self, py: Python<'_>) -> Py<PyArray2<f64>> {
        self.current_values.clone_ref(py)
    }

    /// u64 columns ext/E/I/net nonzero target-neuron counts.
    #[getter]
    fn current_nonzero_counts(&self, py: Python<'_>) -> Py<PyArray2<u64>> {
        self.current_nonzero_counts.clone_ref(py)
    }

    /// Immutable compact state digests, one after every timestep.
    #[getter]
    fn state_digests(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        Ok(PyTuple::new(py, &self.state_digests)?.unbind())
    }

    /// Final membrane potential.
    #[getter]
    fn final_voltage_mv(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        self.final_voltage_mv.clone_ref(py)
    }

    /// Final refractory counters.
    #[getter]
    fn final_refractory_steps(&self, py: Python<'_>) -> Py<PyArray1<u32>> {
        self.final_refractory_steps.clone_ref(py)
    }

    /// Final spike vector.
    #[getter]
    fn final_spikes(&self, py: Python<'_>) -> Py<PyArray1<bool>> {
        self.final_spikes.clone_ref(py)
    }

    /// Final presynaptic traces.
    #[getter]
    fn final_pre_trace(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        self.final_pre_trace.clone_ref(py)
    }

    /// Final postsynaptic traces.
    #[getter]
    fn final_post_trace(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        self.final_post_trace.clone_ref(py)
    }

    /// Final dense row-pre/column-post weights.
    #[getter]
    fn final_weights(&self, py: Python<'_>) -> Py<PyArray2<f64>> {
        self.final_weights.clone_ref(py)
    }

    /// Immutable dense topology mask.
    #[getter]
    fn topology(&self, py: Python<'_>) -> Py<PyArray2<bool>> {
        self.topology.clone_ref(py)
    }

    /// Immutable outgoing CSR offsets.
    #[getter]
    fn outgoing_offsets(&self, py: Python<'_>) -> Py<PyArray1<u64>> {
        self.outgoing_offsets.clone_ref(py)
    }

    /// Immutable outgoing CSR indices.
    #[getter]
    fn outgoing_indices(&self, py: Python<'_>) -> Py<PyArray1<u64>> {
        self.outgoing_indices.clone_ref(py)
    }

    /// Immutable incoming CSR offsets.
    #[getter]
    fn incoming_offsets(&self, py: Python<'_>) -> Py<PyArray1<u64>> {
        self.incoming_offsets.clone_ref(py)
    }

    /// Immutable incoming CSR indices.
    #[getter]
    fn incoming_indices(&self, py: Python<'_>) -> Py<PyArray1<u64>> {
        self.incoming_indices.clone_ref(py)
    }

    /// Authenticated lowercase topology SHA-256.
    #[getter]
    fn topology_digest(&self) -> &str {
        &self.topology_digest
    }
}

fn contiguous_1<T: Element + Copy>(
    array: &PyReadonlyArray1<'_, T>,
    label: &'static str,
) -> PyResult<Vec<T>> {
    if !array.is_c_contiguous() {
        return Err(PyValueError::new_err(format!(
            "{label} must be C-contiguous"
        )));
    }
    Ok(array
        .as_slice()
        .map_err(|_| PyValueError::new_err(format!("{label} must be contiguous")))?
        .to_vec())
}

fn contiguous_2<T: Element + Copy>(
    array: &PyReadonlyArray2<'_, T>,
    label: &'static str,
) -> PyResult<Vec<T>> {
    if !array.is_c_contiguous() {
        return Err(PyValueError::new_err(format!(
            "{label} must be C-contiguous"
        )));
    }
    Ok(array
        .as_slice()
        .map_err(|_| PyValueError::new_err(format!("{label} must be contiguous")))?
        .to_vec())
}

fn readonly_1<T: Element>(py: Python<'_>, values: Vec<T>) -> PyResult<Py<PyArray1<T>>> {
    let owner = PyArray1::from_vec(py, values);
    let bytes = owner.call_method0("tobytes")?;
    let dtype = owner.getattr("dtype")?;
    let array = py
        .import("numpy")?
        .call_method1("frombuffer", (&bytes, &dtype))?
        .cast_into::<PyArray1<T>>()?;
    array.call_method1("setflags", (false,))?;
    Ok(array.unbind())
}

fn readonly_2<T: Element>(
    py: Python<'_>,
    values: Vec<T>,
    rows: usize,
    columns: usize,
) -> PyResult<Py<PyArray2<T>>> {
    let owner = PyArray1::from_vec(py, values);
    let bytes = owner.call_method0("tobytes")?;
    let dtype = owner.getattr("dtype")?;
    let immutable = py
        .import("numpy")?
        .call_method1("frombuffer", (&bytes, &dtype))?
        .cast_into::<PyArray1<T>>()?;
    immutable.call_method1("setflags", (false,))?;
    let array = immutable.reshape([rows, columns])?;
    array.call_method1("setflags", (false,))?;
    Ok(array.unbind())
}

fn sparse_csr(rows: &[Vec<usize>]) -> (Vec<u64>, Vec<u64>) {
    let mut offsets = Vec::with_capacity(rows.len() + 1);
    let mut indices = Vec::new();
    offsets.push(0);
    for row in rows {
        indices.extend(row.iter().map(|index| *index as u64));
        offsets.push(indices.len() as u64);
    }
    (offsets, indices)
}

fn current_arrays(summaries: &[CurrentSummary]) -> (Vec<f64>, Vec<u64>, Vec<u8>) {
    let mut values = Vec::with_capacity(summaries.len() * 12);
    let mut counts = Vec::with_capacity(summaries.len() * 4);
    let mut phases = Vec::with_capacity(summaries.len());
    for summary in summaries {
        for vector in [
            &summary.external,
            &summary.recurrent_excitatory,
            &summary.recurrent_inhibitory,
            &summary.recurrent_net,
        ] {
            values.extend([vector.signed_sum, vector.l1, vector.l2_norm]);
            counts.push(vector.nonzero_count as u64);
        }
        phases.push(match summary.phase {
            EpisodePhase::Cue => 0,
            EpisodePhase::Completion => 1,
        });
    }
    (values, counts, phases)
}

fn build_python_result(
    py: Python<'_>,
    result: StreamedEpisodeResult,
    adjacency: &ImmutableAdjacency,
) -> PyResult<PyStreamedEpisodeV2> {
    let neurons = result.final_state.neurons();
    let timesteps = result.spike_indices.len();
    let (spike_offsets, spike_indices) = sparse_csr(&result.spike_indices);
    let (outgoing_offsets, outgoing_indices) = sparse_csr(adjacency.outgoing());
    let (incoming_offsets, incoming_indices) = sparse_csr(adjacency.incoming());
    let (current_values, current_counts, phases) = current_arrays(&result.current_summaries);
    Ok(PyStreamedEpisodeV2 {
        spike_offsets: readonly_1(py, spike_offsets)?,
        spike_indices: readonly_1(py, spike_indices)?,
        phases: readonly_1(py, phases)?,
        current_values: readonly_2(py, current_values, timesteps, 12)?,
        current_nonzero_counts: readonly_2(py, current_counts, timesteps, 4)?,
        state_digests: result.state_digests,
        final_voltage_mv: readonly_1(py, result.final_state.voltage_mv)?,
        final_refractory_steps: readonly_1(py, result.final_state.refractory_steps)?,
        final_spikes: readonly_1(py, result.final_state.spikes)?,
        final_pre_trace: readonly_1(py, result.final_state.pre_trace)?,
        final_post_trace: readonly_1(py, result.final_state.post_trace)?,
        final_weights: readonly_2(py, result.final_weights.values, neurons, neurons)?,
        topology: readonly_2(py, adjacency.topology().to_vec(), neurons, neurons)?,
        outgoing_offsets: readonly_1(py, outgoing_offsets)?,
        outgoing_indices: readonly_1(py, outgoing_indices)?,
        incoming_offsets: readonly_1(py, incoming_offsets)?,
        incoming_indices: readonly_1(py, incoming_indices)?,
        topology_digest: adjacency.topology_digest().to_owned(),
    })
}

#[allow(clippy::too_many_arguments)]
#[pyfunction(name = "run_streamed_episode_v2")]
fn py_run_streamed_episode_v2(
    py: Python<'_>,
    voltage_mv: PyReadonlyArray1<'_, f64>,
    refractory_steps: PyReadonlyArray1<'_, u32>,
    spikes: PyReadonlyArray1<'_, bool>,
    pre_trace: PyReadonlyArray1<'_, f64>,
    post_trace: PyReadonlyArray1<'_, f64>,
    weights: PyReadonlyArray2<'_, f64>,
    topology: PyReadonlyArray2<'_, bool>,
    packets: PyReadonlyArray2<'_, f64>,
    cue_steps: usize,
    plasticity_enabled: bool,
    config: &ModelConfig,
) -> PyResult<PyStreamedEpisodeV2> {
    let neurons = voltage_mv.len();
    config.validate(neurons)?;
    let one_dimensional_shapes = [
        refractory_steps.len(),
        spikes.len(),
        pre_trace.len(),
        post_trace.len(),
    ];
    if one_dimensional_shapes
        .iter()
        .any(|length| *length != neurons)
        || weights.shape() != [neurons, neurons]
        || topology.shape() != [neurons, neurons]
        || packets.shape().get(1) != Some(&neurons)
    {
        return Err(PyValueError::new_err(
            "streamed arrays must agree on one population shape",
        ));
    }

    let mut rust_state = NetworkState {
        voltage_mv: contiguous_1(&voltage_mv, "voltage_mv")?,
        refractory_steps: contiguous_1(&refractory_steps, "refractory_steps")?,
        spikes: contiguous_1(&spikes, "spikes")?,
        pre_trace: contiguous_1(&pre_trace, "pre_trace")?,
        post_trace: contiguous_1(&post_trace, "post_trace")?,
    };
    let mut rust_weights = WeightMatrix::from_flat(
        neurons,
        config.excitatory_neurons,
        contiguous_2(&weights, "weights")?,
        contiguous_2(&topology, "topology")?,
    )?;
    let adjacency = ImmutableAdjacency::from_weights(&rust_weights)?;
    let packet_rows = packets.shape()[0];
    let packet_values = packets
        .as_slice()
        .map_err(|_| PyValueError::new_err("packets must be C-contiguous"))?;
    let rust_packets: Vec<Vec<f64>> = packet_values.chunks(neurons).map(<[f64]>::to_vec).collect();
    if rust_packets.len() != packet_rows {
        return Err(PyValueError::new_err("packet row count mismatch"));
    }

    let result = py.detach(|| {
        run_streamed_episode_core(
            &mut rust_state,
            &mut rust_weights,
            &adjacency,
            &rust_packets,
            cue_steps,
            plasticity_enabled,
            config,
        )
    })?;
    build_python_result(py, result, &adjacency)
}

/// Register versioned streamed surfaces without changing v1 functions/classes.
pub fn register_streamed(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("STREAMED_API_VERSION", STREAMED_API_VERSION)?;
    module.add("CRATE_VERSION", env!("CARGO_PKG_VERSION"))?;
    module.add(
        "TOPOLOGY_DIGEST_FRAMING_VERSION",
        TOPOLOGY_DIGEST_FRAMING_VERSION,
    )?;
    module.add("STATE_DIGEST_FRAMING_VERSION", STATE_DIGEST_FRAMING_VERSION)?;
    module.add_class::<PyStreamedEpisodeV2>()?;
    module.add_function(wrap_pyfunction!(py_run_streamed_episode_v2, module)?)?;
    Ok(())
}
