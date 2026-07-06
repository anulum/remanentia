// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// Remanentia — Rust-accelerated aggregation pre-compute

//! Rust fast path for the cross-session aggregation precompute.
//!
//! Mirrors `aggregate_precompute.py`:
//!
//! 1. `is_sum_question(q)` detects total/combined/altogether cues.
//! 2. `extract_numeric_facts(text)` finds `<number> <unit>` anchors,
//!    walks back up to 80 chars for the nearest 3+ letter proper-noun
//!    label, returns `(label, value, unit, raw)` triples.
//! 3. `precompute_sum(question, text)` groups by unit, picks the
//!    dominant unit, returns a formatted `COMPUTED TOTAL:` line plus
//!    the fact list.
//!
//! Currency regex is preserved verbatim (dollar / pound / euro / yen
//! symbols + CHF / USD / EUR / GBP ISO codes). Labels must start with
//! an uppercase letter followed by at least two lowercase letters so
//! "It was 42" does not get redacted.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use regex::Regex;
use std::sync::LazyLock;

// ─── Patterns ────────────────────────────────────────────────────────

static SUM_Q: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)\b(?:total(?:\s+(?:number|amount|cost|money|sum|views|reach|hours|days|page|pages))?|combined|altogether|how much total|what(?:\s+is|\s+was)?\s+the\s+total|adding\s+(?:up|together))\b",
    )
    .unwrap()
});

static COUNT_Q: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)\bhow many (?:different |unique |distinct )?\w+").unwrap());

const UNITS_PATTERN: &str = r"views|people|followers|hours|days|dollars|USD|CHF|EUR|GBP|pages|episodes|goals|assists|points";

static NUM_UNIT: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(&format!(
        r"(?P<num>[\d,]+(?:\.\d+)?)\s+(?P<unit>{UNITS_PATTERN})\b"
    ))
    .unwrap()
});

static LABEL: LazyLock<Regex> = LazyLock::new(|| {
    // Capital + 2+ lowercase letters, optional 1-2 more capitalised words.
    Regex::new(r"[A-Z][a-z]{2,}[\w+\-]*(?:[ \-][A-Z][\w+\-]*){0,2}").unwrap()
});

static CURRENCY: LazyLock<Regex> = LazyLock::new(|| {
    // Symbol-prefixed or ISO-code-prefixed amounts with optional label in front.
    Regex::new(
        r"(?i)(?:([A-Z][A-Za-z0-9.+\-]+)[:\s]+(?:spent|cost|paid|charged|was|is)?\s*)?(?:([$£€¥])\s*([\d,]+(?:\.\d+)?))|(?:(CHF|USD|EUR|GBP)\s+([\d,]+(?:\.\d+)?))",
    )
    .unwrap()
});

// ─── Core helpers ────────────────────────────────────────────────────

fn coerce_number(s: &str) -> Option<f64> {
    let cleaned: String = s
        .chars()
        .filter(|c| *c != ',' && *c != '$' && *c != '£' && *c != '€')
        .collect();
    cleaned.trim().parse::<f64>().ok()
}

/// Python-facing `NumericFact` — tuple of (label, value, unit, raw).
type RustFact = (String, f64, String, String);

fn extract_facts_inner(text: &str) -> Vec<RustFact> {
    let mut out: Vec<RustFact> = Vec::new();

    // Stage 1: <num> <unit> anchors + walk-back label.
    for cap in NUM_UNIT.captures_iter(text) {
        let whole = cap.get(0).unwrap();
        let num_str = cap.name("num").unwrap().as_str();
        let unit = cap.name("unit").unwrap().as_str().to_lowercase();
        let val = match coerce_number(num_str) {
            Some(v) => v,
            None => continue,
        };

        // Walk back up to 80 bytes for the nearest label; snap to a
        // char boundary so we never split a multi-byte sequence.
        let match_start = whole.start();
        let target = match_start.saturating_sub(80);
        let mut prefix_start = target;
        while prefix_start > 0 && !text.is_char_boundary(prefix_start) {
            prefix_start -= 1;
        }
        let prefix = &text[prefix_start..match_start];

        let mut last_label: Option<&str> = None;
        for lm in LABEL.find_iter(prefix) {
            last_label = Some(lm.as_str());
        }
        if let Some(label) = last_label {
            out.push((
                label.trim().to_string(),
                val,
                unit,
                whole.as_str().to_string(),
            ));
        }
    }

    // Stage 2: currency amounts with optional prefix label.
    for cap in CURRENCY.captures_iter(text) {
        let label = cap
            .get(1)
            .map(|m| m.as_str().trim().to_string())
            .unwrap_or_default();
        let (unit, val_str) = if let Some(sym) = cap.get(2) {
            (sym.as_str().to_string(), cap.get(3).map(|m| m.as_str()))
        } else if let Some(iso) = cap.get(4) {
            (iso.as_str().to_uppercase(), cap.get(5).map(|m| m.as_str()))
        } else {
            continue;
        };
        let val_str = match val_str {
            Some(v) => v,
            None => continue,
        };
        let val = match coerce_number(val_str) {
            Some(v) => v,
            None => continue,
        };
        let raw = cap.get(0).unwrap().as_str().to_string();
        let lbl = if label.is_empty() {
            "(unlabelled)".to_string()
        } else {
            label
        };
        out.push((lbl, val, unit.to_uppercase(), raw));
    }

    out
}

// ─── PyO3 surface ────────────────────────────────────────────────────

/// Python: `is_sum_question(q) -> bool`.
#[pyfunction]
fn is_sum_question(q: &str) -> bool {
    SUM_Q.is_match(q)
}

/// Python: `is_count_question(q) -> bool`. Sum phrasing wins over count.
#[pyfunction]
fn is_count_question(q: &str) -> bool {
    if SUM_Q.is_match(q) {
        return false;
    }
    COUNT_Q.is_match(q)
}

/// Python: `extract_numeric_facts(text) -> list[tuple[str, float, str, str]]`.
#[pyfunction]
fn extract_numeric_facts(py: Python<'_>, text: &str) -> PyResult<Py<PyList>> {
    let facts = extract_facts_inner(text);
    let lst = PyList::empty(py);
    for (label, value, unit, raw) in facts {
        let t = (label, value, unit, raw);
        lst.append(t)?;
    }
    Ok(lst.unbind())
}

/// Python: `precompute_sum(question, text) -> dict | None`.
///
/// Returns ``None`` when the question is not a sum question or when
/// we lack at least two same-unit facts to sum. Returns a dict with
/// ``kind`` / ``value`` / ``message`` / ``facts`` matching the Python
/// ``PrecomputeResult`` fields when successful.
#[pyfunction]
fn precompute_sum(py: Python<'_>, question: &str, text: &str) -> PyResult<Option<Py<PyDict>>> {
    if !SUM_Q.is_match(question) {
        return Ok(None);
    }
    let facts = extract_facts_inner(text);
    if facts.len() < 2 {
        return Ok(None);
    }

    // Group by unit, keep insertion order of first-seen units so ties
    // are resolved deterministically (Python uses ``max(by_unit.values(), key=len)``
    // which on dict-insertion-order takes the first hit of the tied max).
    use std::collections::HashMap;
    let mut by_unit: HashMap<String, Vec<&RustFact>> = HashMap::new();
    let mut unit_order: Vec<String> = Vec::new();
    for f in facts.iter() {
        if !by_unit.contains_key(&f.2) {
            unit_order.push(f.2.clone());
        }
        by_unit.entry(f.2.clone()).or_default().push(f);
    }

    let mut dominant_unit: &String = &unit_order[0];
    let mut dominant_n: usize = by_unit[dominant_unit].len();
    for u in unit_order.iter().skip(1) {
        let n = by_unit[u].len();
        if n > dominant_n {
            dominant_n = n;
            dominant_unit = u;
        }
    }
    if dominant_n < 2 {
        return Ok(None);
    }

    let dominant = &by_unit[dominant_unit];
    let total: f64 = dominant.iter().map(|f| f.1).sum();
    let all_int = dominant.iter().all(|f| f.1.fract() == 0.0);
    let total_str = if all_int {
        format!("{}", total as i64)
    } else {
        // Python uses "g" formatting. Emulate via trimming trailing zeros.
        let s = format!("{}", total);
        if s.contains('.') {
            s.trim_end_matches('0').trim_end_matches('.').to_string()
        } else {
            s
        }
    };

    // Breakdown: "Label: value unit" joined by " + ".
    let breakdown = dominant
        .iter()
        .map(|f| {
            let value_str = if f.1.fract() == 0.0 {
                format!("{}", f.1 as i64)
            } else {
                format!("{}", f.1)
            };
            if f.2.is_empty() {
                format!("{}: {}", f.0, value_str)
            } else {
                format!("{}: {} {}", f.0, value_str, f.2)
            }
        })
        .collect::<Vec<_>>()
        .join(" + ");

    let unit_str = if dominant_unit.is_empty() {
        String::new()
    } else {
        format!(" {}", dominant_unit)
    };
    let message = format!("COMPUTED TOTAL: {} = {}{}", breakdown, total_str, unit_str);

    let d = PyDict::new(py);
    d.set_item("kind", "total")?;
    d.set_item("value", total)?;
    d.set_item("message", message)?;
    let facts_py = PyList::empty(py);
    for f in dominant.iter() {
        facts_py.append((f.0.clone(), f.1, f.2.clone(), f.3.clone()))?;
    }
    d.set_item("facts", facts_py)?;
    Ok(Some(d.unbind()))
}

#[pymodule]
fn remanentia_aggregate_precompute(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(is_sum_question, m)?)?;
    m.add_function(wrap_pyfunction!(is_count_question, m)?)?;
    m.add_function(wrap_pyfunction!(extract_numeric_facts, m)?)?;
    m.add_function(wrap_pyfunction!(precompute_sum, m)?)?;
    Ok(())
}

// ─── Unit tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sum_question_total() {
        assert!(SUM_Q.is_match("What is the total number of views on my videos?"));
    }

    #[test]
    fn sum_question_combined() {
        assert!(SUM_Q.is_match("Combined reach of my ad campaigns?"));
    }

    #[test]
    fn count_question_how_many() {
        assert!(COUNT_Q.is_match("How many items did I buy?"));
    }

    #[test]
    fn colon_label_extracted() {
        let facts = extract_facts_inner("YouTube: 542 views, TikTok: 1456 views end");
        let labels: Vec<&str> = facts.iter().map(|f| f.0.as_str()).collect();
        assert!(labels.contains(&"YouTube"));
        assert!(labels.contains(&"TikTok"));
    }

    #[test]
    fn walk_back_label() {
        // Label not immediately before number — walk back should still find it.
        let facts = extract_facts_inner("Your YouTube tutorial has 542 views");
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].0, "Your YouTube");
        assert_eq!(facts[0].1, 542.0);
    }

    #[test]
    fn r11_youtube_tiktok_regression() {
        let facts = extract_facts_inner(
            "You said your YouTube tutorial on social media analytics has 542 views. \
             Your TikTok video of Luna playing has 1456 views on the account.",
        );
        assert_eq!(facts.len(), 2);
        let total: f64 = facts.iter().map(|f| f.1).sum();
        assert_eq!(total, 1998.0);
    }

    #[test]
    fn short_label_rejected() {
        // "YT" (2 letters) fails the [a-z]{2,} tail requirement.
        let facts = extract_facts_inner("YT: 1 views. TT: 2 views.");
        assert!(facts.is_empty());
    }

    #[test]
    fn label_must_start_uppercase() {
        let facts = extract_facts_inner("it was 42 degrees");
        assert!(facts.is_empty());
    }

    #[test]
    fn utf8_safety_yen() {
        // Walk-back must not panic on multi-byte chars.
        let text = "Tokyo ¥500 YouTube: 100 views";
        let _ = extract_facts_inner(text);
    }

    #[test]
    fn coerce_number_basics() {
        assert_eq!(coerce_number("1,200"), Some(1200.0));
        assert_eq!(coerce_number("1.5"), Some(1.5));
        assert_eq!(coerce_number("$100"), Some(100.0));
        assert_eq!(coerce_number("garbage"), None);
    }
}
