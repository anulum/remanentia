// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Remanentia — Hash encoding for neuron activation patterns

use md5::{Digest, Md5};
use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;
use rustc_hash::FxHashSet;

use crate::tokenize;

/// Hash-encode text into a neuron activation pattern.
///
/// Each word hashed via MD5 + 7 prime offsets → activation slots.
/// Bigrams get higher weight (0.25 vs 0.15). All values clamped to [0, 1].
///
/// Matches Python `_encode()` in retrieve.py exactly.
#[pyfunction]
pub fn hash_encode(
    text: &str,
    n_neurons: usize,
    hash_primes: Vec<u64>,
    stopwords: FxHashSet<String>,
) -> PyResult<Py<PyArray1<f64>>> {
    let tokens = tokenize::tokenize(text, stopwords);
    let mut pattern = vec![0.0f64; n_neurons];

    // Unigrams: weight 0.15, all primes
    for word in &tokens {
        let h = md5_to_u128(word);
        for &p in &hash_primes {
            let idx = ((h.wrapping_add(p as u128)) % (n_neurons as u128)) as usize;
            pattern[idx] = (pattern[idx] + 0.15).min(1.0);
        }
    }

    // Bigrams: weight 0.25, first 5 primes
    let bgs = tokenize::bigrams(tokens);
    let n_bg_primes = hash_primes.len().min(5);
    for bg in &bgs {
        let h = md5_to_u128(bg);
        for &p in &hash_primes[..n_bg_primes] {
            let idx = ((h.wrapping_add(p as u128)) % (n_neurons as u128)) as usize;
            pattern[idx] = (pattern[idx] + 0.25).min(1.0);
        }
    }

    Python::attach(|py| {
        let arr = Array1::from_vec(pattern);
        Ok(arr.into_pyarray(py).into())
    })
}

fn md5_to_u128(text: &str) -> u128 {
    let mut hasher = Md5::new();
    hasher.update(text.as_bytes());
    let result = hasher.finalize();
    u128::from_be_bytes(result.into())
}
