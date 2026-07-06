// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Remanentia — TF-IDF scoring with bigrams and filename boost

use pyo3::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::HashMap;

use crate::tokenize;

/// Build IDF map from document corpus.
///
/// IDF(term) = ln(1 + N / (1 + df(term)))
#[pyfunction]
pub fn build_idf(
    docs: HashMap<String, String>,
    stopwords: FxHashSet<String>,
) -> HashMap<String, f64> {
    let n_docs = docs.len() as f64;
    let mut df: FxHashMap<String, u32> = FxHashMap::default();

    for (name, text) in &docs {
        let combined = format!("{} {}", text, name.replace('-', " ").replace('_', " "));
        let tokens = tokenize::tokenize(&combined, stopwords.clone());
        let bgs = tokenize::bigrams(tokens.clone());
        let mut terms: FxHashSet<String> = tokens.into_iter().collect();
        terms.extend(bgs);
        for t in terms {
            *df.entry(t).or_insert(0) += 1;
        }
    }

    df.into_iter()
        .map(|(t, count)| {
            let idf = (1.0 + n_docs / (1.0 + count as f64)).ln();
            (t, idf)
        })
        .collect()
}

/// TF-IDF with sublinear TF, bigrams, and filename 3x boost.
///
/// score = sum_{t in query_terms & doc_terms} (1 + ln(tf)) * idf(t)
/// Normalised by query term count.
#[pyfunction]
pub fn tfidf_score(
    query: &str,
    doc_name: &str,
    doc_text: &str,
    idf: HashMap<String, f64>,
    stopwords: FxHashSet<String>,
) -> f64 {
    let q_tokens = tokenize::tokenize(query, stopwords.clone());
    if q_tokens.is_empty() {
        return 0.0;
    }
    let q_bigrams = tokenize::bigrams(q_tokens.clone());
    let mut q_terms: FxHashSet<String> = q_tokens.into_iter().collect();
    let n_q = q_terms.len() + q_bigrams.len();
    q_terms.extend(q_bigrams);

    let name_clean = doc_name.replace('-', " ").replace('_', " ");
    let name_tokens = tokenize::tokenize(&name_clean, stopwords.clone());
    let doc_tokens = tokenize::tokenize(doc_text, stopwords);

    // Build TF map: doc tokens + name tokens x3
    let mut doc_tf: FxHashMap<String, u32> = FxHashMap::default();
    for t in &doc_tokens {
        *doc_tf.entry(t.clone()).or_insert(0) += 1;
    }
    for t in &name_tokens {
        *doc_tf.entry(t.clone()).or_insert(0) += 3;
    }
    // Bigrams
    for bg in tokenize::bigrams(doc_tokens) {
        *doc_tf.entry(bg).or_insert(0) += 1;
    }
    for bg in tokenize::bigrams(name_tokens) {
        *doc_tf.entry(bg).or_insert(0) += 3;
    }

    let mut score = 0.0;
    for t in &q_terms {
        if let Some(&tf) = doc_tf.get(t) {
            let sublinear_tf = 1.0 + (tf as f64).ln();
            score += sublinear_tf * idf.get(t).copied().unwrap_or(0.0);
        }
    }

    score / n_q as f64
}

/// IDF-weighted filename overlap score.
///
/// Returns sum of IDF values for query tokens found in the filename,
/// normalised by the total query IDF.
#[pyfunction]
pub fn filename_bonus(
    query: &str,
    name_lower: &str,
    idf: HashMap<String, f64>,
    stopwords: FxHashSet<String>,
) -> f64 {
    let q_tokens = tokenize::tokenize(query, stopwords);
    if q_tokens.is_empty() {
        return 0.0;
    }

    let mut matched = 0.0;
    let mut total = 0.0;
    for t in &q_tokens {
        let w = idf.get(t).copied().unwrap_or(1.0);
        total += w;
        if name_lower.contains(t.as_str()) {
            matched += w;
        }
    }

    if total < 1e-12 {
        0.0
    } else {
        matched / total
    }
}
