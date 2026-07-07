// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Remanentia — Rust active retrieval helpers (decision point extraction)

use pyo3::prelude::*;
use regex::Regex;
use std::sync::LazyLock;

static DECISION_PATTERNS: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    vec![
    Regex::new(r"(?i)(?:going to|will|plan to|about to)\s+(?:change|modify|delete|remove|add|replace|refactor)").unwrap(),
    Regex::new(r"(?i)(?:choosing|chose|decision|decided)\s+(?:to|between|against)").unwrap(),
    Regex::new(r"(?i)(?:trade.?off|alternative|instead of|rather than)").unwrap(),
    Regex::new(r"(?i)(?:should we|should i|do we|question is)").unwrap(),
]
});

static RE_SENT_SPLIT: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[.!?\n]").unwrap());

/// Identify sentences that contain decision points.
#[pyfunction]
fn extract_decision_points(text: &str) -> Vec<String> {
    let mut points = Vec::new();
    for sentence in RE_SENT_SPLIT.split(text) {
        let stripped = sentence.trim();
        if stripped.len() < 15 {
            continue;
        }
        for pattern in DECISION_PATTERNS.iter() {
            if pattern.is_match(stripped) {
                points.push(stripped.to_string());
                break;
            }
        }
    }
    points
}

#[pymodule]
fn remanentia_active_retrieval(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(extract_decision_points, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_decision_points_flags_decision_sentences() {
        let pts =
            extract_decision_points("We are going to refactor the module. The sky is blue today.");
        assert_eq!(pts, vec!["We are going to refactor the module"]);
    }

    #[test]
    fn extract_decision_points_skips_short_and_plain_sentences() {
        // "Do it" is below the 15-char floor; the second has no decision marker.
        assert!(extract_decision_points("Do it. A plain unremarkable statement here.").is_empty());
    }

    #[test]
    fn extract_decision_points_matches_question_form() {
        assert_eq!(
            extract_decision_points("Should we choose the faster path here?").len(),
            1
        );
    }
}
