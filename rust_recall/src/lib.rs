// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Remanentia — Rust-accelerated recall tokenisation and scoring

use pyo3::prelude::*;
use regex::Regex;
use std::collections::HashSet;
use std::sync::LazyLock;

static RE_WORD: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\w+").unwrap());
static RE_WORD4: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\w{4,}").unwrap());

/// Tokenise text into lowercase word set (equivalent to `re.findall(r"\w+", text.lower())`).
#[pyfunction]
fn tokenize_words(text: &str) -> HashSet<String> {
    let lower = text.to_lowercase();
    RE_WORD
        .find_iter(&lower)
        .map(|m| m.as_str().to_owned())
        .collect()
}

/// Tokenise text into lowercase words with minimum length filter.
/// Equivalent to `re.findall(r"\w{min_len,}", text.lower())`.
#[pyfunction]
fn tokenize_words_min(text: &str, min_len: usize) -> HashSet<String> {
    let lower = text.to_lowercase();
    if min_len <= 4 {
        // Use precompiled regex for common case
        RE_WORD4
            .find_iter(&lower)
            .filter(|m| m.as_str().len() >= min_len)
            .map(|m| m.as_str().to_owned())
            .collect()
    } else {
        RE_WORD
            .find_iter(&lower)
            .filter(|m| m.as_str().len() >= min_len)
            .map(|m| m.as_str().to_owned())
            .collect()
    }
}

/// Compute token overlap score: |intersection| / max(|query|, 1).
#[pyfunction]
fn token_overlap_score(query_tokens: HashSet<String>, doc_tokens: HashSet<String>) -> f64 {
    let overlap = query_tokens.intersection(&doc_tokens).count();
    let divisor = query_tokens.len().max(1);
    overlap as f64 / divisor as f64
}

/// Assess novelty of a query relative to known entity tokens.
///
/// Returns fraction of query tokens (len >= 4) not found in known_tokens.
/// Equivalent to Python `_assess_novelty()`.
#[pyfunction]
fn assess_novelty(query: &str, known_tokens: HashSet<String>) -> f64 {
    let q_tokens = tokenize_words_min(query, 4);
    if q_tokens.is_empty() {
        return 0.0;
    }
    let unknown = q_tokens.difference(&known_tokens).count();
    unknown as f64 / q_tokens.len() as f64
}

#[pymodule]
fn remanentia_recall(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(tokenize_words, m)?)?;
    m.add_function(wrap_pyfunction!(tokenize_words_min, m)?)?;
    m.add_function(wrap_pyfunction!(token_overlap_score, m)?)?;
    m.add_function(wrap_pyfunction!(assess_novelty, m)?)?;
    Ok(())
}
