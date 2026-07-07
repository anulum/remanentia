// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Remanentia — Rust answer normaliser (hedging, polarity, list extraction)

use pyo3::prelude::*;
use regex::Regex;
use std::collections::HashSet;
use std::sync::LazyLock;

static RE_YES: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)^(yes|likely\s+yes|probably\s+yes|most\s+likely\s+yes|yeah|yep|correct|true)")
        .unwrap()
});
static RE_NO: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)^(no|likely\s+no|probably\s+no|most\s+likely\s+no|nah|nope|incorrect|false|unlikely)",
    )
    .unwrap()
});
static RE_EXPL: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"[,;.!]\s+(?:because|since|as|though|but|however|due|given|considering)").unwrap()
});
static RE_HEDGE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)^(?:I think|I believe|I would say|It seems|Based on)\s+").unwrap()
});
static RE_LIST: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\s*,\s+(?:and\s+)?|\s*,\s*|\s+and\s+|\s+or\s+").unwrap());

#[pyfunction]
fn normalize_answer(text: &str) -> String {
    let t = text.trim();
    if t.is_empty() {
        return String::new();
    }
    let t = RE_HEDGE.replace(t, "");
    let t = t.trim();
    if let Some(m) = RE_YES.find(t) {
        return m.as_str().to_lowercase();
    }
    if let Some(m) = RE_NO.find(t) {
        return m.as_str().to_lowercase();
    }
    let parts: Vec<&str> = RE_EXPL.splitn(t, 2).collect();
    let mut r = parts[0].trim().to_string();
    if r.ends_with('.') {
        r.pop();
        r = r.trim_end().to_string();
    }
    r.to_lowercase()
}

#[pyfunction]
fn extract_answer_items(text: &str) -> Vec<String> {
    let n = normalize_answer(text);
    RE_LIST
        .split(&n)
        .map(|s| s.trim().to_lowercase())
        .filter(|s| !s.is_empty())
        .collect()
}

#[pyfunction]
fn answers_match(predicted: &str, ground_truth: &str, threshold: f64) -> bool {
    let p = normalize_answer(predicted);
    let g = normalize_answer(ground_truth);
    if p.is_empty() || g.is_empty() {
        return false;
    }
    if p == g || p.contains(&g) || g.contains(&p) {
        return true;
    }
    let py = RE_YES.is_match(&p);
    let pn = RE_NO.is_match(&p);
    let gy = RE_YES.is_match(&g);
    let gn = RE_NO.is_match(&g);
    if (py && gy) || (pn && gn) {
        return true;
    }
    if (py && gn) || (pn && gy) {
        return false;
    }
    let pi: HashSet<String> = extract_answer_items(predicted).into_iter().collect();
    let gi: HashSet<String> = extract_answer_items(ground_truth).into_iter().collect();
    if !pi.is_empty() && !gi.is_empty() {
        let ov = pi.intersection(&gi).count();
        if ov > 0 && (ov as f64 / gi.len().max(1) as f64) >= threshold {
            return true;
        }
    }
    false
}

#[pymodule]
fn remanentia_answer_normalizer(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(normalize_answer, m)?)?;
    m.add_function(wrap_pyfunction!(extract_answer_items, m)?)?;
    m.add_function(wrap_pyfunction!(answers_match, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_answer_strips_hedge_and_detects_polarity() {
        assert_eq!(normalize_answer(""), "");
        assert_eq!(normalize_answer("   "), "");
        // Hedge prefix removed, then the yes-marker matches.
        assert_eq!(normalize_answer("I think yes, it works"), "yes");
        assert_eq!(normalize_answer("No, because it failed"), "no");
    }

    #[test]
    fn normalize_answer_trims_explanation_and_trailing_dot() {
        assert_eq!(
            normalize_answer("Paris, because it is the capital"),
            "paris"
        );
        assert_eq!(normalize_answer("The Eiffel Tower."), "the eiffel tower");
    }

    #[test]
    fn extract_answer_items_splits_lists() {
        assert_eq!(
            extract_answer_items("apples, oranges and bananas"),
            vec!["apples", "oranges", "bananas"]
        );
    }

    #[test]
    fn answers_match_handles_equality_polarity_and_overlap() {
        // Exact (after normalisation).
        assert!(answers_match("Paris", "paris", 0.5));
        // Same polarity via synonyms.
        assert!(answers_match("yes", "correct", 0.5));
        // Opposite polarity is an explicit mismatch.
        assert!(!answers_match("yes", "no", 0.5));
        // List overlap at/above threshold.
        assert!(answers_match(
            "apples and oranges",
            "oranges and grapes",
            0.5
        ));
        // An empty side never matches.
        assert!(!answers_match("", "paris", 0.5));
    }
}
