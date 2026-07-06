// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// Remanentia — Rust skill extractor helpers

use pyo3::prelude::*;
use regex::Regex;
use std::collections::HashSet;
use std::sync::LazyLock;

static RE_TOKEN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"[a-z0-9_]+").unwrap()
});

// Skill marker pairs: (trigger_pattern, action_pattern)
static SKILL_MARKERS: LazyLock<Vec<(Regex, Regex)>> = LazyLock::new(|| vec![
    (
        Regex::new(r"(?i)fix(?:ed|ing)?").unwrap(),
        Regex::new(r"(?i)(?:update|change|modify|add|remove|replace)").unwrap(),
    ),
    (
        Regex::new(r"(?i)bug|error|fail(?:ure|ed)?|broke").unwrap(),
        Regex::new(r"(?i)(?:fix|resolve|patch|workaround)").unwrap(),
    ),
    (
        Regex::new(r"(?i)chose|decision|trade.?off").unwrap(),
        Regex::new(r"(?i)(?:because|reason|rationale)").unwrap(),
    ),
    (
        Regex::new(r"(?i)pattern|approach|strategy").unwrap(),
        Regex::new(r"(?i)(?:works?|better|cleaner|faster)").unwrap(),
    ),
    (
        Regex::new(r"(?i)refactor").unwrap(),
        Regex::new(r"(?i)(?:extract|split|merge|rename|move)").unwrap(),
    ),
]);

/// Tokenize text to lowercase tokens for skill matching.
#[pyfunction]
fn tokenize_lower(text: &str) -> Vec<String> {
    let lower = text.to_lowercase();
    RE_TOKEN.find_iter(&lower).map(|m| m.as_str().to_string()).collect()
}

/// Check if a line matches any skill marker (trigger + action pair).
/// Returns true if the line describes a skill-like pattern.
#[pyfunction]
fn matches_skill_marker(text: &str) -> bool {
    let lower = text.to_lowercase();
    SKILL_MARKERS.iter().any(|(trigger, action)| {
        trigger.is_match(&lower) && action.is_match(&lower)
    })
}

/// Query skills by token overlap with query.
/// Returns indices sorted by overlap score (descending).
#[pyfunction]
fn rank_skills_by_overlap(query: &str, skill_terms: Vec<Vec<String>>) -> Vec<(usize, f64)> {
    let q_tokens: HashSet<String> = RE_TOKEN.find_iter(&query.to_lowercase())
        .map(|m| m.as_str().to_string())
        .collect();
    if q_tokens.is_empty() {
        return vec![];
    }

    let mut scored: Vec<(usize, f64)> = skill_terms.iter().enumerate()
        .map(|(i, terms)| {
            let term_set: HashSet<&str> = terms.iter().map(|s| s.as_str()).collect();
            let overlap = q_tokens.iter().filter(|t| term_set.contains(t.as_str())).count();
            (i, overlap as f64 / q_tokens.len().max(1) as f64)
        })
        .filter(|(_, score)| *score > 0.0)
        .collect();

    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored
}

#[pymodule]
fn remanentia_skill_extractor(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(tokenize_lower, m)?)?;
    m.add_function(wrap_pyfunction!(matches_skill_marker, m)?)?;
    m.add_function(wrap_pyfunction!(rank_skills_by_overlap, m)?)?;
    Ok(())
}
