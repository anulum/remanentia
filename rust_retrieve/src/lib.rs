// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Remanentia — Rust Retrieval Pipeline (hot-path acceleration)

use pyo3::prelude::*;

mod encode;
mod fusion;
mod snn;
mod tfidf;
mod tokenize;

#[pymodule]
fn remanentia_retrieve(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // tokenize.rs
    m.add_function(wrap_pyfunction!(tokenize::tokenize, m)?)?;
    m.add_function(wrap_pyfunction!(tokenize::stem, m)?)?;
    m.add_function(wrap_pyfunction!(tokenize::expand_query, m)?)?;
    m.add_function(wrap_pyfunction!(tokenize::bigrams, m)?)?;

    // tfidf.rs
    m.add_function(wrap_pyfunction!(tfidf::build_idf, m)?)?;
    m.add_function(wrap_pyfunction!(tfidf::tfidf_score, m)?)?;
    m.add_function(wrap_pyfunction!(tfidf::filename_bonus, m)?)?;

    // snn.rs
    m.add_function(wrap_pyfunction!(snn::spike_feature, m)?)?;
    m.add_function(wrap_pyfunction!(snn::snn_affinity, m)?)?;
    m.add_function(wrap_pyfunction!(snn::cosine_sim, m)?)?;

    // encode.rs
    m.add_function(wrap_pyfunction!(encode::hash_encode, m)?)?;

    // fusion.rs
    m.add_function(wrap_pyfunction!(fusion::reciprocal_rank_fusion, m)?)?;
    m.add_function(wrap_pyfunction!(fusion::entity_graph_score, m)?)?;

    Ok(())
}
