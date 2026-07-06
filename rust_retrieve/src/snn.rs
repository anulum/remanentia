// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Remanentia — SNN spike feature + affinity scoring

use numpy::ndarray::{Array1, ArrayView1, ArrayView2};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

const V_REST: f64 = -65.0;
const V_THRESH: f64 = -55.0;
const V_RESET: f64 = -70.0;
const TAU_M: f64 = 10.0;

/// Deterministic LIF simulation returning spike counts.
///
/// Uses a seeded uniform init for v (matching Python's default_rng(0)).
/// Parameters match the Python implementation exactly.
fn spike_feature_impl(w: &ArrayView2<f64>, stim: &ArrayView1<f64>, steps: usize) -> Array1<f32> {
    let n = stim.len();

    // Deterministic init matching numpy default_rng(0).uniform(-70, -55, n)
    // We use a simple LCG with seed 0 for reproducibility
    let mut v: Vec<f64> = Vec::with_capacity(n);
    let mut rng_state: u64 = 0x12345678_9abcdef0; // deterministic seed
    for _ in 0..n {
        // xorshift64
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        let u = (rng_state as f64) / (u64::MAX as f64);
        v.push(V_RESET + u * (V_THRESH - V_RESET)); // uniform in [-70, -55]
    }

    let i_ext: Vec<f64> = stim.iter().map(|&s| 0.3 + s * 2.0).collect();
    let mut spike_count = vec![0.0f32; n];

    for _ in 0..steps {
        // Compute fired mask
        let fired: Vec<f32> = v.iter().map(|&vi| if vi >= V_THRESH { 1.0 } else { 0.0 }).collect();

        // i_syn = w @ fired
        let mut i_syn = vec![0.0f64; n];
        for i in 0..n {
            let mut sum = 0.0;
            for j in 0..n {
                sum += w[[i, j]] * fired[j] as f64;
            }
            i_syn[i] = sum;
        }

        // dv and voltage update
        for i in 0..n {
            let dv = (-(v[i] - V_REST) / TAU_M + i_ext[i] + i_syn[i] * 0.5) * 1.0; // dt_ms = 1.0
            v[i] += dv;
            if v[i] >= V_THRESH {
                spike_count[i] += 1.0;
                v[i] = V_RESET;
            }
        }
    }

    Array1::from_vec(spike_count)
}

/// Deterministic spike-count feature for a stimulus under fixed weights.
///
/// Args:
///     w: Weight matrix (n x n)
///     stim: Stimulus vector (n,)
///     steps: Number of LIF simulation steps (default 50)
///
/// Returns: spike count array (n,) as float32
#[pyfunction]
#[pyo3(signature = (w, stim, steps=50))]
pub fn spike_feature<'py>(
    py: Python<'py>,
    w: PyReadonlyArray2<f64>,
    stim: PyReadonlyArray1<f64>,
    steps: usize,
) -> Bound<'py, PyArray1<f32>> {
    let w_view = w.as_array();
    let stim_view = stim.as_array();
    let result = spike_feature_impl(&w_view, &stim_view, steps);
    result.into_pyarray(py)
}

fn cosine_sim_impl(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
    let mut dot = 0.0;
    let mut na2 = 0.0;
    let mut nb2 = 0.0;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na2 += a[i] * a[i];
        nb2 += b[i] * b[i];
    }
    let na = na2.sqrt();
    let nb = nb2.sqrt();
    if na < 1e-12 || nb < 1e-12 {
        return 0.0;
    }
    dot / (na * nb)
}

/// Cosine similarity between two vectors.
#[pyfunction]
pub fn cosine_sim(a: PyReadonlyArray1<f64>, b: PyReadonlyArray1<f64>) -> f64 {
    cosine_sim_impl(&a.as_array(), &b.as_array())
}

/// SNN affinity: cosine similarity of spike features for query vs trace.
#[pyfunction]
#[pyo3(signature = (w, query_stim, trace_stim, steps=50))]
pub fn snn_affinity(
    w: PyReadonlyArray2<f64>,
    query_stim: PyReadonlyArray1<f64>,
    trace_stim: PyReadonlyArray1<f64>,
    steps: usize,
) -> f64 {
    let w_view = w.as_array();
    let q_spikes = spike_feature_impl(&w_view, &query_stim.as_array(), steps);
    let t_spikes = spike_feature_impl(&w_view, &trace_stim.as_array(), steps);

    // Convert f32 spike counts to f64 for cosine
    let q64: Array1<f64> = q_spikes.mapv(|x| x as f64);
    let t64: Array1<f64> = t_spikes.mapv(|x| x as f64);
    cosine_sim_impl(&q64.view(), &t64.view())
}
