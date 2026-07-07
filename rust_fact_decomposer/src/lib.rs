// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Remanentia — Rust fact decomposer helpers (sentence split + 9-type classify)
// PyO3-exposed signatures carry the Python-side argument set; cleanup tracked in TODO.md.
#![allow(clippy::too_many_arguments)]

use pyo3::prelude::*;
use regex::Regex;
use std::sync::LazyLock;

// Priority 8: state-change verbs
static RE_CHANGE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
    r"(?i)\b(started|began|switched|changed|moved|left|quit|joined|got|bought|sold|upgraded|downgraded|replaced|updated|decided|chose|picked|adopted|dropped|stopped|finished|married|divorced|graduated|retired|hired|fired)\b"
).unwrap()
});

// Priority 7: preference
static RE_PREF: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
    r"(?i)\b(I (?:like|love|prefer|enjoy|hate|dislike|want|need|always|never|usually|favorite|favourite)|my (?:favorite|favourite|go-to|preferred))\b"
).unwrap()
});

// Priority 6: plan/future
static RE_PLAN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
    r"(?i)\b(I (?:plan to|will|am going to|intend to|might|may|hope to|want to|'m planning)|going to|planning to|scheduled for|booked for|appointment|reservation)\b"
).unwrap()
});

// Priority 1: decision (checked first — highest priority)
static RE_DECISION: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
    r"(?i)\b(we (?:decided|chose|picked|selected|agreed|concluded|resolved)|decision was|the verdict|final (?:choice|answer|call)|consensus)\b"
).unwrap()
});

// Priority 2: correction
static RE_CORRECTION: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
    r"(?i)\b(actually|correction|was wrong|mistake|misunderstood|should have been|turned out|in fact|not true|incorrect|clarification|I was wrong|we were wrong)\b"
).unwrap()
});

// Priority 3: principle/rule
static RE_PRINCIPLE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
    r"(?i)\b(always (?:do|use|check|ensure|verify)|never (?:do|use|skip|delete)|rule(?:s)? (?:is|are|of)|principle|best practice|guideline|must always|must never|invariant|axiom|law of)\b"
).unwrap()
});

// Priority 4: commitment/deadline
static RE_COMMITMENT: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
    r"(?i)\b(I (?:promise|commit|guarantee|owe|agreed to)|deadline|due (?:by|on|date)|deliverable|committed to|obligation|must (?:deliver|finish|complete) by)\b"
).unwrap()
});

// Priority 5: skill/procedure
static RE_SKILL: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
    r"(?i)\b(to (?:do this|fix this|run this|build this|deploy|install|configure|set up)|step (?:1|2|3|one|two)|the (?:command|procedure|workflow|recipe|process) is|how to|run the following|execute|you need to)\b"
).unwrap()
});

/// Classify sentence into one of 9 fact types.
///
/// Priority order: decision > correction > principle > commitment > skill >
/// plan > preference > state > event (default).
#[pyfunction]
fn classify_fact_type(sentence: &str) -> String {
    if RE_DECISION.is_match(sentence) {
        "decision".into()
    } else if RE_CORRECTION.is_match(sentence) {
        "correction".into()
    } else if RE_PRINCIPLE.is_match(sentence) {
        "principle".into()
    } else if RE_COMMITMENT.is_match(sentence) {
        "commitment".into()
    } else if RE_SKILL.is_match(sentence) {
        "skill".into()
    } else if RE_PLAN.is_match(sentence) {
        "plan".into()
    } else if RE_PREF.is_match(sentence) {
        "preference".into()
    } else if RE_CHANGE.is_match(sentence) {
        "state".into()
    } else {
        "event".into()
    }
}

/// Split text into sentences (at ". " followed by uppercase).
#[pyfunction]
fn split_sentences(text: &str) -> Vec<String> {
    let mut sents = Vec::new();
    let mut start = 0;
    let bytes = text.as_bytes();
    for i in 0..bytes.len().saturating_sub(2) {
        if (bytes[i] == b'.' || bytes[i] == b'!' || bytes[i] == b'?')
            && bytes[i + 1] == b' '
            && bytes[i + 2].is_ascii_uppercase()
        {
            let s = text[start..=i].trim();
            if s.len() >= 10 {
                sents.push(s.to_string());
            }
            start = i + 2;
        }
    }
    let tail = text[start..].trim();
    if tail.len() >= 10 {
        sents.push(tail.to_string());
    }
    if sents.is_empty() && text.trim().len() >= 10 {
        sents.push(text.trim().to_string());
    }
    sents
}

/// Check if sentence contains a state-change verb.
#[pyfunction]
fn has_change_verb(sentence: &str) -> bool {
    RE_CHANGE.is_match(sentence)
}

/// Extract words of 4+ characters (for entity overlap checks).
#[pyfunction]
fn tokenize_words(text: &str) -> Vec<String> {
    static RE_WORDS: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\w{4,}").unwrap());
    RE_WORDS
        .find_iter(text)
        .map(|m| m.as_str().to_string())
        .collect()
}

// ── FactIndex.query() kernel ──────────────────────────────────

static RE_WORD3: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\w{3,}").unwrap());

static RE_ENTITY: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b").unwrap());

static RE_QUOTED: LazyLock<Regex> = LazyLock::new(|| Regex::new(r#""([^"]{2,40})""#).unwrap());

fn tokenize_lower(text: &str) -> Vec<String> {
    let lower = text.to_lowercase();
    RE_WORD3
        .find_iter(&lower)
        .map(|m| m.as_str().to_string())
        .collect()
}

fn extract_entities(text: &str) -> Vec<String> {
    let mut entities = Vec::new();
    for m in RE_ENTITY.find_iter(text) {
        let pos = m.start();
        if pos == 0 {
            continue;
        }
        if pos >= 2 {
            let prev2 = &text[pos.saturating_sub(2)..pos];
            if prev2 == ". " || prev2 == "! " || prev2 == "? " {
                continue;
            }
        }
        if m.as_str().len() > 1 {
            entities.push(m.as_str().to_string());
        }
    }
    for cap in RE_QUOTED.captures_iter(text) {
        if let Some(q) = cap.get(1) {
            entities.push(q.as_str().to_string());
        }
    }
    entities
}

fn date_before(a: &str, b: &str) -> bool {
    a < b // ISO dates sort lexicographically
}

use std::collections::HashMap;

/// Persistent Rust-side fact index — avoids FFI conversion on every query.
#[pyclass]
struct RustFactIndex {
    keyword_to_facts: HashMap<String, Vec<usize>>,
    entity_to_facts: HashMap<String, Vec<usize>>,
    fact_valid_until: Vec<String>,
    fact_session_idx: Vec<f64>,
    fact_supersedes: Vec<bool>,
}

#[pymethods]
impl RustFactIndex {
    #[new]
    fn new(
        fact_texts: Vec<String>,
        fact_entities: Vec<Vec<String>>,
        fact_valid_until: Vec<String>,
        fact_session_idx: Vec<f64>,
        fact_supersedes: Vec<bool>,
    ) -> Self {
        let mut keyword_to_facts: HashMap<String, Vec<usize>> = HashMap::new();
        let mut entity_to_facts: HashMap<String, Vec<usize>> = HashMap::new();

        for (i, text) in fact_texts.iter().enumerate() {
            for token in tokenize_lower(text) {
                keyword_to_facts.entry(token).or_default().push(i);
            }
        }
        for (i, entities) in fact_entities.iter().enumerate() {
            for ent in entities {
                let key = ent.to_lowercase();
                entity_to_facts.entry(key).or_default().push(i);
            }
        }

        RustFactIndex {
            keyword_to_facts,
            entity_to_facts,
            fact_valid_until,
            fact_session_idx,
            fact_supersedes,
        }
    }

    fn query(
        &self,
        question: &str,
        reference_date: &str,
        filter_expired: bool,
        top_k: usize,
    ) -> Vec<(usize, f64)> {
        let q_tokens = tokenize_lower(question);
        let q_entities = extract_entities(question);

        let mut scores: HashMap<usize, f64> = HashMap::new();

        for token in &q_tokens {
            if let Some(indices) = self.keyword_to_facts.get(token) {
                for &idx in indices {
                    *scores.entry(idx).or_insert(0.0) += 1.0;
                }
            }
        }

        for ent in &q_entities {
            let key = ent.to_lowercase();
            if let Some(indices) = self.entity_to_facts.get(&key) {
                for &idx in indices {
                    *scores.entry(idx).or_insert(0.0) += 3.0;
                }
            }
        }

        if filter_expired && !reference_date.is_empty() {
            let expired: Vec<usize> = scores
                .keys()
                .filter(|&&idx| {
                    if idx < self.fact_valid_until.len() {
                        let vu = &self.fact_valid_until[idx];
                        !vu.is_empty() && date_before(vu, reference_date)
                    } else {
                        false
                    }
                })
                .copied()
                .collect();
            for idx in expired {
                scores.remove(&idx);
            }
        }

        let q_lower = question.to_lowercase();
        let is_update = [
            "current",
            "now",
            "latest",
            "most recent",
            "today",
            "right now",
        ]
        .iter()
        .any(|w| q_lower.contains(w));
        if is_update {
            let keys: Vec<usize> = scores.keys().copied().collect();
            for idx in keys {
                if idx < self.fact_session_idx.len() {
                    *scores.get_mut(&idx).unwrap() += self.fact_session_idx[idx] * 2.0;
                    if idx < self.fact_supersedes.len() && self.fact_supersedes[idx] {
                        *scores.get_mut(&idx).unwrap() += 5.0;
                    }
                }
            }
        }

        let mut ranked: Vec<(usize, f64)> = scores.into_iter().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked.truncate(top_k);
        ranked
    }
}

/// One-shot query (for backward compat). Prefer RustFactIndex for repeated queries.
#[pyfunction]
fn fact_index_query(
    keyword_to_facts: HashMap<String, Vec<usize>>,
    entity_to_facts: HashMap<String, Vec<usize>>,
    question: &str,
    fact_valid_until: Vec<String>,
    fact_session_idx: Vec<f64>,
    fact_supersedes: Vec<bool>,
    reference_date: &str,
    filter_expired: bool,
    top_k: usize,
) -> Vec<(usize, f64)> {
    let idx = RustFactIndex {
        keyword_to_facts,
        entity_to_facts,
        fact_valid_until,
        fact_session_idx,
        fact_supersedes,
    };
    idx.query(question, reference_date, filter_expired, top_k)
}

#[pymodule]
fn remanentia_fact_decomposer(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(classify_fact_type, m)?)?;
    m.add_function(wrap_pyfunction!(split_sentences, m)?)?;
    m.add_function(wrap_pyfunction!(has_change_verb, m)?)?;
    m.add_function(wrap_pyfunction!(tokenize_words, m)?)?;
    m.add_function(wrap_pyfunction!(fact_index_query, m)?)?;
    m.add_class::<RustFactIndex>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_fact_type_respects_priority_order() {
        assert_eq!(classify_fact_type("We decided to ship it"), "decision");
        assert_eq!(classify_fact_type("Actually that was wrong"), "correction");
        assert_eq!(classify_fact_type("always verify the input"), "principle");
        assert_eq!(classify_fact_type("I promise to finish it"), "commitment");
        assert_eq!(classify_fact_type("the command is make build"), "skill");
        assert_eq!(classify_fact_type("I plan to travel soon"), "plan");
        assert_eq!(classify_fact_type("I love pizza"), "preference");
        assert_eq!(classify_fact_type("She started a new job"), "state");
        assert_eq!(classify_fact_type("The sky is blue today"), "event");
        // Decision outranks a correction marker in the same sentence.
        assert_eq!(
            classify_fact_type("We decided, actually, to wait"),
            "decision"
        );
    }

    #[test]
    fn split_sentences_breaks_on_terminator_then_capital() {
        assert_eq!(
            split_sentences("First sentence here. Second sentence follows."),
            vec!["First sentence here.", "Second sentence follows."]
        );
        // No internal break: the whole (long enough) line is one sentence.
        assert_eq!(
            split_sentences("only one line here"),
            vec!["only one line here"]
        );
    }

    #[test]
    fn has_change_verb_detects_state_change() {
        assert!(has_change_verb("She joined the team"));
        assert!(!has_change_verb("a quiet morning"));
    }

    #[test]
    fn tokenize_words_keeps_four_plus_char_words() {
        assert_eq!(
            tokenize_words("The quick fox jumps"),
            vec!["quick", "jumps"]
        );
    }

    #[test]
    fn date_before_compares_iso_dates_lexicographically() {
        assert!(date_before("2023-01-01", "2023-06-01"));
        assert!(!date_before("2023-06-01", "2023-01-01"));
    }

    #[test]
    fn rust_fact_index_ranks_keyword_and_entity_hits_first() {
        let idx = RustFactIndex::new(
            vec![
                "the kubernetes cluster scaled".into(),
                "unrelated note about coffee".into(),
            ],
            vec![vec!["Kubernetes".into()], vec![]],
            vec![String::new(), String::new()],
            vec![0.0, 0.0],
            vec![false, false],
        );
        // "Kubernetes" (not at position 0) contributes the entity boost; the
        // shared keywords add to it, so fact 0 ranks ahead of the coffee note.
        let ranked = idx.query("the Kubernetes cluster status", "", false, 10);
        assert_eq!(ranked[0].0, 0);
    }
}
