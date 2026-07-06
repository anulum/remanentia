// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Remanentia — Tokenize, stem, query expansion

use pyo3::prelude::*;
use regex::Regex;
use rustc_hash::FxHashSet;
use std::sync::LazyLock;

static RE_TOKEN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[a-z0-9_]+").unwrap());

const STEM_SUFFIXES: &[&str] = &[
    "ation", "tion", "sion", "meant", "ness", "ity", "ous", "ive",
    "ing", "ical", "ally", "able", "ible", "full", "less", "ized",
    "ise", "ize", "ed", "ly", "er", "est", "al", "es", "s",
];

/// Lowercase regex tokeniser with stopword removal.
///
/// Matches `[a-z0-9_]+` tokens, drops stopwords and single-char tokens.
#[pyfunction]
pub fn tokenize(text: &str, stopwords: FxHashSet<String>) -> Vec<String> {
    let lower = text.to_lowercase();
    RE_TOKEN
        .find_iter(&lower)
        .filter_map(|m| {
            let w = m.as_str();
            if w.len() > 1 && !stopwords.contains(w) {
                Some(w.to_owned())
            } else {
                None
            }
        })
        .collect()
}

/// Minimal suffix-stripping stemmer (25 suffixes).
#[pyfunction]
pub fn stem(word: &str) -> String {
    for &suffix in STEM_SUFFIXES {
        if word.ends_with(suffix) && word.len() - suffix.len() >= 3 {
            return word[..word.len() - suffix.len()].to_owned();
        }
    }
    word.to_owned()
}

/// Expand query with stems for broader matching.
#[pyfunction]
pub fn expand_query(query: &str, stopwords: FxHashSet<String>) -> String {
    let tokens = tokenize(query, stopwords);
    let stems: FxHashSet<String> = tokens.iter().map(|t| stem(t)).collect();
    let token_set: FxHashSet<&str> = tokens.iter().map(|s| s.as_str()).collect();
    let mut extra: Vec<String> = stems
        .into_iter()
        .filter(|s| !token_set.contains(s.as_str()))
        .collect();
    if extra.is_empty() {
        return query.to_owned();
    }
    extra.sort();
    format!("{} {}", query, extra.join(" "))
}

/// Generate bigrams from token list.
#[pyfunction]
pub fn bigrams(tokens: Vec<String>) -> Vec<String> {
    if tokens.len() < 2 {
        return Vec::new();
    }
    tokens
        .windows(2)
        .map(|pair| format!("{}_{}", pair[0], pair[1]))
        .collect()
}
