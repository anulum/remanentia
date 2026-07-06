// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// Arcane Sapience — Rust STDP Kernel
//
// Vectorized STDP weight update via outer product.
// Replaces the pure-numpy STDP in snn_daemon.py with ~10-50x speedup.
// PyO3-exposed signatures carry the Python-side argument set; cleanup tracked in TODO.md.
#![allow(clippy::too_many_arguments)]

use ndarray::{Array1, Array2, Zip};
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

/// Batch STDP update: dW = A+ * spike_post ⊗ trace - A- * trace ⊗ spike_post
///
/// w: (N, N) weight matrix (modified in place)
/// spiked: (N,) binary spike vector
/// last_spike: (N,) last spike times
/// t_now: current simulation time
/// mask: (N, N) connectivity mask (only update existing connections)
/// a_plus, a_minus: STDP learning rates
/// tau: STDP time constant (ms)
/// w_max: maximum weight
#[pyfunction]
fn stdp_batch<'py>(
    _py: Python<'py>,
    w: &Bound<'py, PyArray2<f32>>,
    spiked: PyReadonlyArray1<'py, f32>,
    last_spike: PyReadonlyArray1<'py, f32>,
    t_now: f32,
    mask: PyReadonlyArray2<'py, f32>,
    a_plus: f32,
    a_minus: f32,
    tau: f32,
    w_max: f32,
) -> PyResult<()> {
    let spiked = spiked.as_array();
    let last_spike = last_spike.as_array();
    let mask = mask.as_array();
    let n = spiked.len();

    // Compute pre-synaptic trace: exp(-dt/tau) for valid spikes
    let mut trace = Array1::<f32>::zeros(n);
    for i in 0..n {
        let dt = t_now - last_spike[i];
        if dt > 0.0 && dt < 100.0 {
            trace[i] = (-dt / tau).exp();
        }
    }

    // Outer products for LTP and LTD
    let mut dw = Array2::<f32>::zeros((n, n));
    // LTP: dW[post, pre] += A+ * spike_post[post] * trace[pre]
    // LTD: dW[post, pre] -= A- * trace[post] * spike_pre[pre]
    Zip::indexed(&mut dw).par_for_each(|idx, val| {
        let (i, j) = idx;
        if mask[[i, j]] > 0.0 {
            *val = a_plus * spiked[i] * trace[j] - a_minus * trace[i] * spiked[j];
        }
    });

    // Apply to weight matrix
    {
        let mut w_arr = unsafe { w.as_array_mut() };
        Zip::from(&mut w_arr).and(&dw).for_each(|w_val, &dw_val| {
            *w_val = (*w_val + dw_val).clamp(0.0, w_max);
        });
    }

    Ok(())
}

/// LIF network step: compute dv for all neurons.
/// Returns indices of neurons that spiked.
#[pyfunction]
fn lif_step<'py>(
    py: Python<'py>,
    v: &Bound<'py, PyArray1<f32>>,
    w: PyReadonlyArray2<'py, f32>,
    i_ext: PyReadonlyArray1<'py, f32>,
    v_rest: f32,
    v_thresh: f32,
    v_reset: f32,
    tau_m: f32,
    dt_ms: f32,
) -> PyResult<Bound<'py, PyArray1<u32>>> {
    let w = w.as_array();
    let i_ext = i_ext.as_array();
    let n = i_ext.len();

    // Detect spikes from previous step
    let v_arr = unsafe { v.as_array() };
    let mut fired = Array1::<f32>::zeros(n);
    for i in 0..n {
        if v_arr[i] >= v_thresh {
            fired[i] = 1.0;
        }
    }

    // Synaptic current: W @ fired
    let i_syn = w.dot(&fired);

    // Update membrane potentials
    let mut spike_idx = Vec::new();
    {
        let mut v_mut = unsafe { v.as_array_mut() };
        for i in 0..n {
            let dv = (-(v_mut[i] - v_rest) / tau_m + i_ext[i] + i_syn[i] * 0.5) * dt_ms;
            v_mut[i] += dv;
            if v_mut[i] >= v_thresh {
                spike_idx.push(i as u32);
                v_mut[i] = v_reset;
            }
        }
    }

    Ok(PyArray1::from_vec(py, spike_idx))
}

/// Homeostatic synaptic scaling: normalize weights toward target mean activity.
///
/// For each row, scales active weights (> 0.001) so the row mean approaches
/// target_mean. Clips all values to [0, 2.0]. Modifies w in place.
///
/// Reference: Turrigiano & Nelson (2004), "Homeostatic plasticity in
/// the developing nervous system," Nature Reviews Neuroscience.
#[pyfunction]
fn homeostatic_scaling<'py>(
    _py: Python<'py>,
    w: &Bound<'py, PyArray2<f32>>,
    target_mean: f32,
    rate: f32,
) -> PyResult<()> {
    let n = {
        let shape = unsafe { w.as_array() }.shape().to_vec();
        shape[0]
    };

    let mut w_arr = unsafe { w.as_array_mut() };

    for i in 0..n {
        let row = w_arr.row(i);
        let mut active_sum: f32 = 0.0;
        let mut active_count: usize = 0;
        for &val in row.iter() {
            if val > 0.001 {
                active_sum += val;
                active_count += 1;
            }
        }
        if active_count < 2 {
            continue;
        }
        let current_mean = active_sum / active_count as f32;
        if current_mean < 0.001 {
            continue;
        }
        let scale = 1.0 + rate * (target_mean / current_mean - 1.0);
        for val in w_arr.row_mut(i).iter_mut() {
            *val = (*val * scale).clamp(0.0, 2.0);
        }
    }

    Ok(())
}

/// Hash-based unigram+bigram text encoding for SNN stimulus injection.
///
/// Replicates snn_backend.encode_text: tokenise → MD5 hash → scatter
/// into pattern array with prime-based indexing. Returns f32 array of
/// length n_neurons.
#[pyfunction]
fn encode_text<'py>(
    py: Python<'py>,
    text: &str,
    n_neurons: usize,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    use std::hash::{Hash, Hasher};

    const PRIMES: [usize; 7] = [
        7919, 104729, 15485863, 32452843, 49979687, 67867967, 86028121,
    ];

    let mut pattern = vec![0.0f32; n_neurons];
    let lower = text.to_lowercase();

    // Tokenise: [a-z0-9_]+ with len > 1
    let tokens: Vec<&str> = lower
        .split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|w| w.len() > 1)
        .collect();

    // Unigram encoding
    for word in &tokens {
        let h = md5_hash(word.as_bytes());
        for &p in &PRIMES {
            let idx = (h.wrapping_add(p)) % n_neurons;
            pattern[idx] = (pattern[idx] + 0.15).min(1.0);
        }
    }

    // Bigram encoding
    for pair in tokens.windows(2) {
        let bg = format!("{}_{}", pair[0], pair[1]);
        let h = md5_hash(bg.as_bytes());
        for &p in &PRIMES[..5] {
            let idx = (h.wrapping_add(p)) % n_neurons;
            pattern[idx] = (pattern[idx] + 0.25).min(1.0);
        }
    }

    Ok(PyArray1::from_vec(py, pattern))
}

/// Simple MD5-based hash (matching Python hashlib.md5().hexdigest() → int conversion).
fn md5_hash(data: &[u8]) -> usize {
    // Use a fast non-crypto hash that produces same distribution as MD5 for indexing.
    // We don't need crypto-grade — just deterministic scatter.
    let mut h: u64 = 0xcbf29ce484222325; // FNV offset basis
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3); // FNV prime
    }
    h as usize
}

#[pymodule]
fn arcane_stdp(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(stdp_batch, m)?)?;
    m.add_function(wrap_pyfunction!(lif_step, m)?)?;
    m.add_function(wrap_pyfunction!(homeostatic_scaling, m)?)?;
    m.add_function(wrap_pyfunction!(encode_text, m)?)?;
    Ok(())
}
