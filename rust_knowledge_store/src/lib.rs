// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Remanentia — Rust knowledge store helpers (tokenize, extract, keywords)

use pyo3::prelude::*;
use regex::Regex;
use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;

// ── Tokenizer ──────────────────────────────────────────────────

static RE_TOKEN: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[a-z0-9][a-z0-9_]{2,}").unwrap());

/// Tokenize text into lowercase words (3+ chars).
#[pyfunction]
fn tokenize(text: &str) -> HashSet<String> {
    let lower = text.to_lowercase();
    RE_TOKEN
        .find_iter(&lower)
        .map(|m| m.as_str().to_string())
        .collect()
}

// ── Keyword extraction ─────────────────────────────────────────

static RE_KW_TOKEN: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[a-z0-9_]{4,}").unwrap());
static RE_CAPS: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[A-Z][a-z]+(?:[A-Z][a-z]+)*").unwrap());
static RE_VERSION: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"v\d+\.\d+(?:\.\d+)?").unwrap());

/// Extract keywords: frequency ≥2, capitalized terms, version numbers. Max 20.
#[pyfunction]
fn extract_keywords(text: &str) -> Vec<String> {
    let lower = text.to_lowercase();
    let mut freq: HashMap<String, usize> = HashMap::new();
    for m in RE_KW_TOKEN.find_iter(&lower) {
        *freq.entry(m.as_str().to_string()).or_insert(0) += 1;
    }
    let mut keywords: Vec<String> = freq
        .into_iter()
        .filter(|(_, c)| *c >= 2)
        .map(|(t, _)| t)
        .collect();
    for m in RE_CAPS.find_iter(text) {
        let s = m.as_str();
        if s.len() > 3 {
            keywords.push(s.to_lowercase());
        }
    }
    for m in RE_VERSION.find_iter(text) {
        keywords.push(m.as_str().to_string());
    }
    keywords.sort();
    keywords.dedup();
    keywords.truncate(20);
    keywords
}

// ── Entity extraction ──────────────────────────────────────────

static KNOWN_ENTITIES: &[&str] = &[
    "stdp",
    "bm25",
    "lif",
    "snn",
    "embedding",
    "pytorch",
    "cuda",
    "gpu",
    "locomo",
    "remanentia",
    "director-ai",
    "sc-neurocore",
    "scpn",
    "consolidation",
    "retrieval",
    "daemon",
    "mcp",
    "fastapi",
];
static RE_PCT: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\d+\.?\d*%").unwrap());
// Note: Rust regex does not support lookbehind, so we use a simpler pattern
// for names after sentence boundaries
static RE_NAME_AFTER_SENT: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[.!?\n] ([A-Z][a-z]{2,})").unwrap());

/// Extract entities: known concepts, version numbers, percentages, names.
#[pyfunction]
fn extract_entities(text: &str) -> HashSet<String> {
    let mut entities = HashSet::new();
    let lower = text.to_lowercase();

    for known in KNOWN_ENTITIES {
        if lower.contains(known) {
            entities.insert(known.to_string());
        }
    }
    for m in RE_VERSION.find_iter(text) {
        entities.insert(m.as_str().to_string());
    }
    for m in RE_PCT.find_iter(text) {
        entities.insert(m.as_str().to_string());
    }
    for m in RE_NAME_AFTER_SENT.captures_iter(text) {
        if let Some(name) = m.get(1) {
            entities.insert(name.as_str().to_lowercase());
        }
    }
    entities
}

// ── Person name extraction ─────────────────────────────────────

static RE_PERSON_COLON: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?m)^([A-Z][a-z]{2,}):").unwrap());
static RE_PERSON_SENT: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?:^|\.\s+|!\s+|\?\s+|\n\s*)([A-Z][a-z]{2,})\b").unwrap());
static STOP_WORDS: &[&str] = &[
    "the", "this", "that", "what", "when", "where", "who", "how", "why", "yes", "yeah", "wow",
    "hey", "thanks", "congrats", "glad", "great", "sure", "gonna",
];

/// Extract person names from conversational text.
#[pyfunction]
fn extract_person_names(text: &str) -> HashSet<String> {
    let mut names = HashSet::new();
    let stop: HashSet<&str> = STOP_WORDS.iter().copied().collect();

    for cap in RE_PERSON_COLON.captures_iter(text) {
        if let Some(m) = cap.get(1) {
            names.insert(m.as_str().to_lowercase());
        }
    }
    for cap in RE_PERSON_SENT.captures_iter(text) {
        if let Some(m) = cap.get(1) {
            let word = m.as_str().to_lowercase();
            if !stop.contains(word.as_str()) {
                names.insert(word);
            }
        }
    }
    names
}

// ── Stateless function versions (used when pyclass not constructed) ──

/// BM25-lite search over knowledge notes.
#[pyfunction]
fn knowledge_search(
    token_index: HashMap<String, HashSet<String>>,
    superseded_ids: HashSet<String>,
    query_tokens: HashSet<String>,
    top_k: usize,
    exclude_superseded: bool,
) -> Vec<(String, f64)> {
    knowledge_search_inner(
        &token_index,
        &superseded_ids,
        &query_tokens,
        top_k,
        exclude_superseded,
    )
}

fn knowledge_search_inner(
    token_index: &HashMap<String, HashSet<String>>,
    superseded_ids: &HashSet<String>,
    query_tokens: &HashSet<String>,
    top_k: usize,
    exclude_superseded: bool,
) -> Vec<(String, f64)> {
    let q_len = query_tokens.len().max(1) as f64;
    let mut scored: Vec<(String, f64)> = Vec::new();

    for (nid, tokens) in token_index {
        if exclude_superseded && superseded_ids.contains(nid) {
            continue;
        }
        let overlap = query_tokens.intersection(tokens).count();
        if overlap > 0 {
            scored.push((nid.clone(), overlap as f64 / q_len));
        }
    }

    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(top_k);
    scored
}

/// BFS graph traversal to find related note IDs.
#[pyfunction]
fn get_related_ids(
    note_links: HashMap<String, Vec<(String, String)>>,
    start_id: &str,
    depth: usize,
    edge_types: HashSet<String>,
    valid_ids: HashSet<String>,
) -> Vec<String> {
    get_related_ids_inner(&note_links, start_id, depth, &edge_types, &valid_ids)
}

fn get_related_ids_inner(
    note_links: &HashMap<String, Vec<(String, String)>>,
    start_id: &str,
    depth: usize,
    edge_types: &HashSet<String>,
    valid_ids: &HashSet<String>,
) -> Vec<String> {
    let filter_types = !edge_types.is_empty();
    let mut visited: HashSet<String> = HashSet::new();
    visited.insert(start_id.to_string());
    let mut frontier = vec![start_id.to_string()];
    let mut result = Vec::new();

    for _ in 0..depth {
        let mut next_frontier = Vec::new();
        for nid in &frontier {
            if let Some(links) = note_links.get(nid) {
                for (target, link_type) in links {
                    if filter_types && !edge_types.contains(link_type) {
                        continue;
                    }
                    if !visited.contains(target) && valid_ids.contains(target) {
                        visited.insert(target.clone());
                        next_frontier.push(target.clone());
                        result.push(target.clone());
                    }
                }
            }
        }
        frontier = next_frontier;
    }

    result
}

/// Combined seed search + graph traversal + re-ranking.
#[pyfunction]
fn graph_search(
    token_index: HashMap<String, HashSet<String>>,
    superseded_ids: HashSet<String>,
    note_links: HashMap<String, Vec<(String, String)>>,
    valid_ids: HashSet<String>,
    query_tokens: HashSet<String>,
    top_k: usize,
    hop_depth: usize,
) -> Vec<(String, f64)> {
    let seeds = knowledge_search_inner(&token_index, &superseded_ids, &query_tokens, 3, true);
    if seeds.is_empty() {
        return Vec::new();
    }

    let mut all_notes: HashMap<String, bool> = HashMap::new();
    for (id, _) in &seeds {
        all_notes.insert(id.clone(), true);
    }
    for (seed_id, _) in &seeds {
        let related =
            get_related_ids_inner(&note_links, seed_id, hop_depth, &HashSet::new(), &valid_ids);
        for rid in related {
            if !superseded_ids.contains(&rid) {
                all_notes.entry(rid).or_insert(false);
            }
        }
    }

    let q_len = query_tokens.len().max(1) as f64;
    let mut scored: Vec<(String, f64)> = Vec::new();
    for (nid, is_seed) in &all_notes {
        let tokens = match token_index.get(nid) {
            Some(t) => t,
            None => continue,
        };
        let overlap = query_tokens.intersection(tokens).count();
        let mut score = overlap as f64 / q_len;
        if *is_seed {
            score *= 1.5;
        }
        scored.push((nid.clone(), score));
    }

    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(top_k);
    scored
}

// ── Persistent index (pyclass) for repeated queries ──

/// Persistent Rust-side knowledge index — holds token_index + links natively.
#[pyclass]
struct RustKnowledgeIndex {
    token_index: HashMap<String, HashSet<String>>,
    superseded_ids: HashSet<String>,
    note_links: HashMap<String, Vec<(String, String)>>,
    valid_ids: HashSet<String>,
}

#[pymethods]
impl RustKnowledgeIndex {
    #[new]
    fn new(
        token_index: HashMap<String, HashSet<String>>,
        superseded_ids: HashSet<String>,
        note_links: HashMap<String, Vec<(String, String)>>,
        valid_ids: HashSet<String>,
    ) -> Self {
        RustKnowledgeIndex {
            token_index,
            superseded_ids,
            note_links,
            valid_ids,
        }
    }

    fn search(
        &self,
        query_tokens: HashSet<String>,
        top_k: usize,
        exclude_superseded: bool,
    ) -> Vec<(String, f64)> {
        knowledge_search_inner(
            &self.token_index,
            &self.superseded_ids,
            &query_tokens,
            top_k,
            exclude_superseded,
        )
    }

    fn get_related(
        &self,
        start_id: &str,
        depth: usize,
        edge_types: HashSet<String>,
    ) -> Vec<String> {
        get_related_ids_inner(
            &self.note_links,
            start_id,
            depth,
            &edge_types,
            &self.valid_ids,
        )
    }

    fn graph_search(
        &self,
        query_tokens: HashSet<String>,
        top_k: usize,
        hop_depth: usize,
    ) -> Vec<(String, f64)> {
        let seeds = knowledge_search_inner(
            &self.token_index,
            &self.superseded_ids,
            &query_tokens,
            3,
            true,
        );
        if seeds.is_empty() {
            return Vec::new();
        }

        let mut all_notes: HashMap<String, bool> = HashMap::new();
        for (id, _) in &seeds {
            all_notes.insert(id.clone(), true);
        }
        for (seed_id, _) in &seeds {
            let related = get_related_ids_inner(
                &self.note_links,
                seed_id,
                hop_depth,
                &HashSet::new(),
                &self.valid_ids,
            );
            for rid in related {
                if !self.superseded_ids.contains(&rid) {
                    all_notes.entry(rid).or_insert(false);
                }
            }
        }

        let q_len = query_tokens.len().max(1) as f64;
        let mut scored: Vec<(String, f64)> = Vec::new();
        for (nid, is_seed) in &all_notes {
            let tokens = match self.token_index.get(nid) {
                Some(t) => t,
                None => continue,
            };
            let overlap = query_tokens.intersection(tokens).count();
            let mut score = overlap as f64 / q_len;
            if *is_seed {
                score *= 1.5;
            }
            scored.push((nid.clone(), score));
        }

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);
        scored
    }
}

#[pymodule]
fn remanentia_knowledge_store(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(tokenize, m)?)?;
    m.add_function(wrap_pyfunction!(extract_keywords, m)?)?;
    m.add_function(wrap_pyfunction!(extract_entities, m)?)?;
    m.add_function(wrap_pyfunction!(extract_person_names, m)?)?;
    m.add_function(wrap_pyfunction!(knowledge_search, m)?)?;
    m.add_function(wrap_pyfunction!(get_related_ids, m)?)?;
    m.add_function(wrap_pyfunction!(graph_search, m)?)?;
    m.add_class::<RustKnowledgeIndex>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenize_yields_lowercase_three_plus_char_tokens() {
        let toks = tokenize("Hello World foo ab");
        assert!(toks.contains("hello"));
        assert!(toks.contains("world"));
        assert!(toks.contains("foo"));
        // "ab" is below the 3-char floor.
        assert!(!toks.contains("ab"));
    }

    #[test]
    fn extract_keywords_covers_frequency_caps_and_versions() {
        let kw = extract_keywords("error error Kubernetes v1.2.3");
        assert!(kw.contains(&"error".to_string())); // frequency >= 2
        assert!(kw.contains(&"kubernetes".to_string())); // capitalised term
        assert!(kw.contains(&"v1.2.3".to_string())); // version number
    }

    #[test]
    fn extract_entities_finds_known_terms_versions_and_percentages() {
        let ents = extract_entities("pytorch and bm25 at 90% on v2.0");
        assert!(ents.contains("pytorch"));
        assert!(ents.contains("bm25"));
        assert!(ents.contains("90%"));
        assert!(ents.contains("v2.0"));
    }

    #[test]
    fn extract_person_names_reads_speaker_and_sentence_starts() {
        let names = extract_person_names("Alice: hi there\nBob left early.");
        assert!(names.contains("alice"));
        assert!(names.contains("bob"));
    }
}
