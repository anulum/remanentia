// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Remanentia — Rust BM25 Search Engine

use pyo3::prelude::*;
use rayon::prelude::*;
use regex::Regex;
use rustc_hash::{FxHashMap, FxHashSet};
use std::sync::LazyLock;

// ── Index helper regexes ───────────────────────────────────────

static RE_TOKEN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"[a-z0-9][a-z0-9_]{2,}").unwrap()
});
static RE_CODE_START: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^\s*(def |fn |pub fn |class |impl )").unwrap()
});


/// BM25 index over paragraphs. Parallel scoring via rayon.
#[pyclass]
struct BM25Index {
    /// For each paragraph: list of token IDs
    paragraphs: Vec<Vec<u32>>,
    /// Token string -> ID mapping
    token_to_id: FxHashMap<String, u32>,
    /// IDF values per token ID
    idf: Vec<f32>,
    /// Paragraph lengths (in tokens)
    doc_lens: Vec<f32>,
    /// Average document length
    avg_dl: f32,
    /// Paragraph -> (doc_idx, para_idx) mapping
    para_map: Vec<(u32, u32)>,
}

#[pymethods]
impl BM25Index {
    #[new]
    fn new() -> Self {
        BM25Index {
            paragraphs: Vec::new(),
            token_to_id: FxHashMap::default(),
            idf: Vec::new(),
            doc_lens: Vec::new(),
            avg_dl: 1.0,
            para_map: Vec::new(),
        }
    }

    /// Build index from lists of token lists + paragraph mappings.
    /// paragraph_tokens: list of list of strings
    /// para_map: list of (doc_idx, para_idx) tuples
    fn build(
        &mut self,
        paragraph_tokens: Vec<Vec<String>>,
        para_map: Vec<(u32, u32)>,
    ) {
        let n = paragraph_tokens.len();
        self.para_map = para_map;

        // Build vocabulary
        let mut next_id: u32 = 0;
        let mut df: FxHashMap<u32, u32> = FxHashMap::default();

        self.paragraphs = paragraph_tokens
            .iter()
            .map(|tokens| {
                let mut seen: FxHashSet<u32> = FxHashSet::default();
                let ids: Vec<u32> = tokens
                    .iter()
                    .map(|t| {
                        let id = *self.token_to_id.entry(t.clone()).or_insert_with(|| {
                            let id = next_id;
                            next_id += 1;
                            id
                        });
                        if seen.insert(id) {
                            *df.entry(id).or_insert(0) += 1;
                        }
                        id
                    })
                    .collect();
                ids
            })
            .collect();

        // IDF: log(1 + N / (1 + df))
        let vocab_size = next_id as usize;
        self.idf = vec![0.0; vocab_size];
        for (&id, &count) in df.iter() {
            self.idf[id as usize] = (1.0 + n as f32 / (1.0 + count as f32)).ln();
        }

        // Document lengths
        self.doc_lens = self.paragraphs.iter().map(|p| p.len() as f32).collect();
        self.avg_dl = if n > 0 {
            self.doc_lens.iter().sum::<f32>() / n as f32
        } else {
            1.0
        };
    }

    /// Search: returns list of (paragraph_index, score) sorted descending.
    /// Uses rayon for parallel scoring.
    fn search(&self, query_tokens: Vec<String>, top_k: usize) -> Vec<(usize, f32)> {
        let k1: f32 = 1.5;
        let b: f32 = 0.75;

        // Convert query tokens to IDs
        let q_ids: Vec<u32> = query_tokens
            .iter()
            .filter_map(|t| self.token_to_id.get(t).copied())
            .collect();

        if q_ids.is_empty() {
            return Vec::new();
        }

        // Build query token set for fast lookup
        let q_set: FxHashSet<u32> = q_ids.iter().copied().collect();

        // Score all paragraphs in parallel (real term frequency)
        let mut scores: Vec<(usize, f32)> = self
            .paragraphs
            .par_iter()
            .enumerate()
            .filter_map(|(i, tokens)| {
                let dl = self.doc_lens[i];
                let mut score: f32 = 0.0;

                // Count term frequencies for query tokens in this paragraph
                let mut tf_map: FxHashMap<u32, u32> = FxHashMap::default();
                for &tid in tokens.iter() {
                    if q_set.contains(&tid) {
                        *tf_map.entry(tid).or_insert(0) += 1;
                    }
                }

                for (&tid, &count) in tf_map.iter() {
                    let tf = count as f32;
                    let idf = self.idf[tid as usize];
                    score += idf * (tf * (k1 + 1.0))
                        / (tf + k1 * (1.0 - b + b * dl / self.avg_dl));
                }

                if score > 0.0 {
                    Some((i, score))
                } else {
                    None
                }
            })
            .collect();

        // Sort descending by score
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(top_k);
        scores
    }

    /// Get paragraph mapping for a result index
    fn get_para_info(&self, para_idx: usize) -> (u32, u32) {
        if para_idx < self.para_map.len() {
            self.para_map[para_idx]
        } else {
            (0, 0)
        }
    }

    fn num_paragraphs(&self) -> usize {
        self.paragraphs.len()
    }

    fn vocab_size(&self) -> usize {
        self.token_to_id.len()
    }
}

/// Cosine similarity between two float vectors (for embedding rerank)
#[pyfunction]
fn cosine_batch(query: Vec<f32>, matrix: Vec<Vec<f32>>) -> Vec<f32> {
    let q_norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
    if q_norm < 1e-12 {
        return vec![0.0; matrix.len()];
    }

    matrix
        .par_iter()
        .map(|row| {
            let dot: f32 = query.iter().zip(row.iter()).map(|(a, b)| a * b).sum();
            let r_norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
            if r_norm < 1e-12 {
                0.0
            } else {
                dot / (q_norm * r_norm)
            }
        })
        .collect()
}

/// Tokenize text for BM25: lowercase, 3+ char alphanumeric words.
#[pyfunction]
fn tokenize(text: &str) -> Vec<String> {
    let lower = text.to_lowercase();
    RE_TOKEN.find_iter(&lower).map(|m| m.as_str().to_string()).collect()
}

/// Count occurrences of each token for real TF in BM25.
#[pyfunction]
fn token_counts(tokens: Vec<String>) -> FxHashMap<String, u32> {
    let mut counts = FxHashMap::default();
    for t in tokens {
        *counts.entry(t).or_insert(0) += 1;
    }
    counts
}

/// Tag paragraph with semantic type: function, code, decision, finding, metric, version, discussion.
#[pyfunction]
fn classify_paragraph(text: &str, is_code: bool) -> String {
    let lower = text.to_lowercase();
    if is_code {
        if RE_CODE_START.is_match(text) {
            return "function".into();
        }
        return "code".into();
    }
    let decision_kw = ["decided", "decision", "chose", "rejected", "we will", "the plan"];
    if decision_kw.iter().any(|w| lower.contains(w)) { return "decision".into(); }
    let finding_kw = ["found", "finding", "result", "measured", "shows that", "proved"];
    if finding_kw.iter().any(|w| lower.contains(w)) { return "finding".into(); }
    let metric_kw = ["p@1", "percent", "accuracy", "precision", "score", "benchmark"];
    if metric_kw.iter().any(|w| lower.contains(w)) { return "metric".into(); }
    let version_kw = ["version", "v0.", "v1.", "v2.", "v3.", "release", "shipped"];
    if version_kw.iter().any(|w| lower.contains(w)) { return "version".into(); }
    "discussion".into()
}

/// Split text into meaningful paragraphs for indexing.
#[pyfunction]
fn split_paragraphs(text: &str, is_code: bool) -> Vec<String> {
    if is_code {
        // Code: split on blank lines or function/class definitions
        let mut chunks = Vec::new();
        let mut current = String::new();
        for line in text.lines() {
            if line.trim().is_empty() && !current.trim().is_empty() {
                let trimmed = current.trim().to_string();
                if trimmed.len() >= 30 {
                    chunks.push(trimmed);
                }
                current.clear();
            } else {
                current.push_str(line);
                current.push('\n');
            }
            if chunks.len() >= 200 { break; }
        }
        let trimmed = current.trim().to_string();
        if trimmed.len() >= 30 && chunks.len() < 200 {
            chunks.push(trimmed);
        }
        return chunks;
    }

    // Text: split on double newlines
    let mut paragraphs = Vec::new();
    for block in text.split("\n\n") {
        let stripped = block.trim();
        if stripped.is_empty() { continue; }
        // Skip pure headers
        let lines: Vec<&str> = stripped.lines().collect();
        let has_content = lines.iter().any(|l| !l.trim_start().starts_with('#'));
        if !has_content && lines.len() == 1 { continue; }
        if stripped.len() >= 30 && stripped.len() <= 10_000 {
            paragraphs.push(stripped.to_string());
        }
    }
    paragraphs
}

#[pymodule]
fn remanentia_search(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<BM25Index>()?;
    m.add_function(wrap_pyfunction!(cosine_batch, m)?)?;
    m.add_function(wrap_pyfunction!(tokenize, m)?)?;
    m.add_function(wrap_pyfunction!(token_counts, m)?)?;
    m.add_function(wrap_pyfunction!(classify_paragraph, m)?)?;
    m.add_function(wrap_pyfunction!(split_paragraphs, m)?)?;
    Ok(())
}
