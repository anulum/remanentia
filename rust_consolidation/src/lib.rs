// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Remanentia — Rust consolidation engine helpers
// Imported crate; idiomatic clippy cleanup tracked in TODO.md (Rust WIP).
#![allow(
    clippy::needless_range_loop,
    clippy::type_complexity,
    clippy::useless_vec
)]

use pyo3::prelude::*;
use regex::Regex;
use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;

// ── Project patterns ───────────────────────────────────────────

static PROJECTS: &[(&str, &[&str])] = &[
    ("director-ai", &["director-ai", "director_ai"]),
    ("sc-neurocore", &["sc-neurocore", "neurocore"]),
    ("scpn-fusion-core", &["scpn-fusion", "fusion-core"]),
    (
        "scpn-phase-orchestrator",
        &["scpn-phase-orchestrator", "phase-orchestrator"],
    ),
    ("scpn-control", &["scpn-control"]),
    (
        "scpn-quantum-control",
        &["scpn-quantum-control", "quantum-control"],
    ),
    (
        "remanentia",
        &["remanentia", "arcane", "snn", "holographic"],
    ),
    ("revenue", &["revenue", "pricing", "commercial"]),
];

// ── Known concepts (60+) ───────────────────────────────────────

static CONCEPTS: &[&str] = &[
    "stdp",
    "lif",
    "kuramoto",
    "hopfield",
    "tf-idf",
    "bm25",
    "embedding",
    "pytorch",
    "cuda",
    "gpu",
    "cpu",
    "daemon",
    "holographic",
    "attractor",
    "inhibition",
    "spike",
    "neuron",
    "retrieval",
    "consolidation",
    "upde",
    "stuart-landau",
    "dimits",
    "gyrokinetic",
    "tokamak",
    "vqe",
    "heron",
    "bcpnn",
    "csdp",
    "hebbian",
    "perron-frobenius",
    "marchenko-pastur",
    "eigenvalue",
    "svd",
    "minilm",
    "sentence-transformer",
    "fastapi",
    "mcp",
    "docker",
    "prometheus",
    "grafana",
    "ci",
    "pytest",
    "rust",
    "pyo3",
    "maturin",
    "rayon",
    "arcaneneuron",
    "chirp",
    "chimera",
    "bifurcation",
    "entropy",
    "fisher",
    "lyapunov",
    "boltzmann",
    "hippocampus",
    "dentate gyrus",
    "pattern separation",
    "dale's law",
    "e/i balance",
    "mem0",
    "letta",
    "zep",
    "memos",
    "langmem",
    "joss",
    "neurips",
    "emnlp",
    "arxiv",
    "zenodo",
    "agpl",
    "pypi",
    "loihi",
];

// ── Regex patterns ─────────────────────────────────────────────

static RE_VERSION: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"v\d+\.\d+(?:\.\d+)?").unwrap());
static RE_PERCENT: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\d+\.?\d*%").unwrap());
static RE_FILEPATH: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[\w/\\]+\.(?:py|rs|md|json|yaml|toml)\b").unwrap());
static RE_FUNC_NAME: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b[a-z][a-z_]+(?:_[a-z]+){2,}\b").unwrap());
static RE_CAMEL_CASE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b").unwrap());

// Typed relation patterns
static RELATION_PATTERNS: LazyLock<Vec<(Regex, &'static str)>> = LazyLock::new(|| {
    vec![
        (
            Regex::new(r"(?i)\bbecause\b|\bcaused by\b|\bdue to\b|\broot cause\b").unwrap(),
            "caused_by",
        ),
        (
            Regex::new(r"(?i)\bfixed\b|\brepaired\b|\bcorrected\b|\bpatched\b").unwrap(),
            "fixed_by",
        ),
        (
            Regex::new(r"(?i)\breplaced\b|\bsuperseded\b|\binstead of\b").unwrap(),
            "replaced",
        ),
        (
            Regex::new(r"(?i)\bcontradicts?\b|\binconsistent with\b|\bconflicts? with\b").unwrap(),
            "contradicts",
        ),
        (
            Regex::new(r"(?i)\bv\d+\.\d+|\bversion\b").unwrap(),
            "version_of",
        ),
        (
            Regex::new(r"(?i)\bdepends on\b|\brequires\b|\bneeds\b").unwrap(),
            "depends_on",
        ),
        (
            Regex::new(r"(?i)\bimproved\b|\bfrom .+ to\b|\bincreased\b|\bdecreased\b").unwrap(),
            "improved",
        ),
        (
            Regex::new(r"(?i)\bproduced\b|\bcreated\b|\bgenerated\b|\bwrote\b").unwrap(),
            "produced",
        ),
        (
            Regex::new(r"(?i)\bused in\b|\bpart of\b|\bcomponent of\b").unwrap(),
            "used_in",
        ),
        (
            Regex::new(r"(?i)\btested\b|\bbenchmarked\b|\bevaluated\b|\bmeasured\b").unwrap(),
            "tested_with",
        ),
    ]
});

// Key line triggers
static TRIGGERS: &[&str] = &[
    "decided",
    "decision",
    "found",
    "finding",
    "result",
    "key insight",
    "conclusion",
    "fix",
    "resolved",
    "chose",
    "rejected",
    "confirmed",
    "measured",
    "p@1",
    "precision",
    "accuracy",
    "because",
    "therefore",
    "root cause",
    "the reason",
    "we proved",
    "this means",
    "critical",
    "important",
    "changed",
    "broke",
    "works",
    "doesn't work",
    "failed",
    "succeeded",
    "shipped",
    "version",
    "v0.",
    "v1.",
    "v2.",
    "v3.",
];

/// Extract entity names from trace text (3-layer extraction).
#[pyfunction]
fn extract_entities(text: &str) -> Vec<String> {
    let mut entities = HashSet::new();
    let lower = text.to_lowercase();

    // Layer 1: Projects
    for (proj, patterns) in PROJECTS {
        if patterns.iter().any(|p| lower.contains(p)) {
            entities.insert(proj.to_string());
        }
    }

    // Layer 2: Known concepts
    for concept in CONCEPTS {
        if lower.contains(concept) {
            entities.insert(concept.to_string());
        }
    }

    // Layer 3: Dynamic
    for m in RE_VERSION.find_iter(text) {
        entities.insert(m.as_str().to_string());
    }
    for m in RE_PERCENT.find_iter(text) {
        entities.insert(m.as_str().to_string());
    }
    for m in RE_FILEPATH.find_iter(text) {
        let s = m.as_str();
        let name = s.rsplit('/').next().unwrap_or(s);
        let name = name.rsplit('\\').next().unwrap_or(name);
        if name.len() > 3 {
            entities.insert(name.to_string());
        }
    }
    for m in RE_FUNC_NAME.find_iter(text) {
        if m.as_str().len() > 8 {
            entities.insert(m.as_str().to_string());
        }
    }
    for m in RE_CAMEL_CASE.find_iter(text) {
        entities.insert(m.as_str().to_string());
    }

    let mut sorted: Vec<String> = entities.into_iter().collect();
    sorted.sort();
    sorted
}

/// Extract decision/finding key lines from trace text with context capture.
#[pyfunction]
fn extract_key_lines(text: &str) -> Vec<String> {
    let lines: Vec<&str> = text.split('\n').collect();
    let mut key_lines = Vec::new();

    for (i, line) in lines.iter().enumerate() {
        let stripped = line.trim();
        if stripped.is_empty() || stripped.starts_with('#') {
            continue;
        }
        let lower = stripped.to_lowercase();
        if TRIGGERS.iter().any(|t| lower.contains(t)) {
            let clean = stripped
                .trim_start_matches(|c: char| "- *>".contains(c))
                .trim();
            if clean.len() > 20 {
                let mut context = vec![clean.to_string()];
                for j in (i + 1)..std::cmp::min(i + 3, lines.len()) {
                    let next = lines[j]
                        .trim()
                        .trim_start_matches(|c: char| "- *>".contains(c))
                        .trim();
                    if !next.is_empty() && !next.starts_with('#') && next.len() > 10 {
                        context.push(next.to_string());
                    }
                }
                key_lines.push(context.join(" "));
            }
        }
    }
    key_lines.truncate(30);
    key_lines
}

/// Extract typed relations between entity pairs from connecting text.
/// Returns Vec<(source, target, relation_type, evidence)>.
#[pyfunction]
fn extract_typed_relations(text: &str, entities: Vec<String>) -> Vec<(String, String, String)> {
    let lower = text.to_lowercase();
    let mut typed = Vec::new();

    for i in 0..entities.len() {
        for j in (i + 1)..entities.len() {
            let e1 = &entities[i];
            let e2 = &entities[j];
            let pos1 = lower.find(&e1.to_lowercase());
            let pos2 = lower.find(&e2.to_lowercase());
            if let (Some(p1), Some(p2)) = (pos1, pos2) {
                let start = std::cmp::min(p1, p2);
                let end = std::cmp::max(p1 + e1.len(), p2 + e2.len());
                if end <= text.len() {
                    let between = &text[start..end];
                    let mut found = false;
                    for (re, rel_type) in RELATION_PATTERNS.iter() {
                        if re.is_match(between) {
                            let mut pair = vec![e1.clone(), e2.clone()];
                            pair.sort();
                            typed.push((pair[0].clone(), pair[1].clone(), rel_type.to_string()));
                            found = true;
                            break;
                        }
                    }
                    if !found {
                        let mut pair = vec![e1.clone(), e2.clone()];
                        pair.sort();
                        typed.push((pair[0].clone(), pair[1].clone(), "co_occurs".to_string()));
                    }
                }
            }
        }
    }
    typed
}

/// Parse YAML-ish frontmatter from a semantic memory file.
/// Returns HashMap of key→value pairs, or empty if no frontmatter.
#[pyfunction]
fn parse_frontmatter(text: &str) -> HashMap<String, String> {
    let mut result = HashMap::new();
    if !text.starts_with("---") {
        return result;
    }
    if let Some(end) = text[3..].find("---") {
        let fm = text[3..3 + end].trim();
        for line in fm.lines() {
            let line = line.trim();
            if line.starts_with('-') || !line.contains(':') {
                continue;
            }
            if let Some((key, val)) = line.split_once(':') {
                result.insert(key.trim().to_string(), val.trim().to_string());
            }
        }
    }
    result
}

// ── Tier 3: cluster_traces ─────────────────────────────────────

/// Cluster traces by project + date proximity (2-day gap threshold).
///
/// Input: list of (trace_name, project, date_str) tuples.
/// Returns: list of clusters, each a list of trace names.
#[pyfunction]
fn cluster_traces(traces: Vec<(String, String, String)>) -> Vec<Vec<String>> {
    // Group by project
    let mut by_project: HashMap<String, Vec<(String, String)>> = HashMap::new();
    for (name, project, date) in traces {
        by_project.entry(project).or_default().push((name, date));
    }

    let mut clusters: Vec<Vec<String>> = Vec::new();

    for (_proj, mut entries) in by_project {
        entries.sort_by(|a, b| a.1.cmp(&b.1));
        if entries.is_empty() {
            continue;
        }

        let mut current_cluster = vec![entries[0].0.clone()];
        for i in 1..entries.len() {
            let prev_date = &entries[i - 1].1;
            let curr_date = &entries[i].1;
            let gap_days = date_gap_days(prev_date, curr_date);
            if gap_days > 2 {
                clusters.push(std::mem::take(&mut current_cluster));
            }
            current_cluster.push(entries[i].0.clone());
        }
        if !current_cluster.is_empty() {
            clusters.push(current_cluster);
        }
    }

    clusters
}

/// Parse YYYY-MM-DD and compute absolute day gap.
fn date_gap_days(a: &str, b: &str) -> i64 {
    let parse = |s: &str| -> Option<i64> {
        let s = if s.len() >= 10 { &s[..10] } else { s };
        let parts: Vec<&str> = s.split('-').collect();
        if parts.len() != 3 {
            return None;
        }
        let y: i64 = parts[0].parse().ok()?;
        let m: i64 = parts[1].parse().ok()?;
        let d: i64 = parts[2].parse().ok()?;
        // Simple days-from-epoch approximation (sufficient for gap calculation)
        Some(y * 365 + m * 30 + d)
    };
    match (parse(a), parse(b)) {
        (Some(da), Some(db)) => (db - da).abs(),
        _ => 0,
    }
}

// ── Tier 3: build_summary_dag ──────────────────────────────────

/// Build a hierarchical summary DAG from trace data.
///
/// Input: list of (name, date, project, entities, key_lines, text) tuples.
/// fanout: grouping factor per level (default 4).
/// Returns: list of dicts with node_id, level, summary, children, date_range, entities, project.
#[pyfunction]
fn build_summary_dag(
    trace_data: Vec<(String, String, String, Vec<String>, Vec<String>, String)>,
    fanout: usize,
) -> Vec<HashMap<String, PyObject>> {
    Python::with_gil(|py| {
        if trace_data.is_empty() {
            return Vec::new();
        }

        // Sort by date
        let mut sorted_data = trace_data;
        sorted_data.sort_by(|a, b| a.1.cmp(&b.1));

        // Level 0: leaf nodes
        struct Node {
            node_id: String,
            level: usize,
            summary: String,
            children: Vec<String>,
            date_range: (String, String),
            entities: Vec<String>,
            project: String,
        }

        let mut leaves: Vec<Node> = Vec::new();
        for (name, date, project, entities, key_lines, text) in &sorted_data {
            let summary = if !key_lines.is_empty() {
                key_lines
                    .iter()
                    .take(5)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(" ")
            } else {
                let end = std::cmp::min(200, text.len());
                text[..end].to_string()
            };
            let date_str = if date.len() >= 10 { &date[..10] } else { date };
            let ents: Vec<String> = entities.iter().take(20).cloned().collect();
            leaves.push(Node {
                node_id: format!("L0_{}", name),
                level: 0,
                summary,
                children: vec![name.clone()],
                date_range: (date_str.to_string(), date_str.to_string()),
                entities: ents,
                project: project.clone(),
            });
        }

        let mut all_nodes: Vec<Node> = Vec::new();
        // Move leaves into all_nodes, keep references for next level
        let mut current_level: Vec<usize> = Vec::new(); // indices into all_nodes
        for leaf in leaves {
            let idx = all_nodes.len();
            all_nodes.push(leaf);
            current_level.push(idx);
        }

        let mut level: usize = 1;

        while current_level.len() > 1 {
            let mut next_level: Vec<usize> = Vec::new();

            for chunk_start in (0..current_level.len()).step_by(fanout) {
                let chunk_end = std::cmp::min(chunk_start + fanout, current_level.len());
                let group = &current_level[chunk_start..chunk_end];

                let mut merged_parts: Vec<String> = Vec::new();
                let mut all_entities: HashSet<String> = HashSet::new();
                let mut earliest = "9999".to_string();
                let mut latest = "0000".to_string();
                let mut children_ids: Vec<String> = Vec::new();
                let mut projects: Vec<String> = Vec::new();

                for &idx in group {
                    let node = &all_nodes[idx];
                    let part: String = node.summary.chars().take(100).collect();
                    merged_parts.push(part);
                    for e in &node.entities {
                        all_entities.insert(e.clone());
                    }
                    if !node.date_range.0.is_empty() && node.date_range.0 < earliest {
                        earliest = node.date_range.0.clone();
                    }
                    if !node.date_range.1.is_empty() && node.date_range.1 > latest {
                        latest = node.date_range.1.clone();
                    }
                    children_ids.push(node.node_id.clone());
                    projects.push(node.project.clone());
                }

                // Most common project
                let project = most_common(&projects).unwrap_or_else(|| "general".to_string());
                let merged_summary = merged_parts.join(" | ");
                let node_id = format!("L{}_{}", level, chunk_start / fanout);
                if earliest == "9999" {
                    earliest = String::new();
                }
                if latest == "0000" {
                    latest = String::new();
                }
                if !earliest.is_empty() {
                    let node_id = format!("L{}_{}_{}", level, chunk_start / fanout, earliest);
                    let mut sorted_ents: Vec<String> = all_entities.into_iter().collect();
                    sorted_ents.sort();
                    sorted_ents.truncate(30);
                    let idx = all_nodes.len();
                    all_nodes.push(Node {
                        node_id,
                        level,
                        summary: merged_summary,
                        children: children_ids,
                        date_range: (earliest, latest),
                        entities: sorted_ents,
                        project,
                    });
                    next_level.push(idx);
                } else {
                    let mut sorted_ents: Vec<String> = all_entities.into_iter().collect();
                    sorted_ents.sort();
                    sorted_ents.truncate(30);
                    let idx = all_nodes.len();
                    all_nodes.push(Node {
                        node_id,
                        level,
                        summary: merged_summary,
                        children: children_ids,
                        date_range: (earliest, latest),
                        entities: sorted_ents,
                        project,
                    });
                    next_level.push(idx);
                }
            }

            current_level = next_level;
            level += 1;
        }

        // Convert to Python dicts
        all_nodes
            .into_iter()
            .map(|node| {
                let mut d: HashMap<String, PyObject> = HashMap::new();
                d.insert(
                    "node_id".to_string(),
                    node.node_id.into_pyobject(py).unwrap().into_any().unbind(),
                );
                d.insert(
                    "level".to_string(),
                    node.level.into_pyobject(py).unwrap().into_any().unbind(),
                );
                d.insert(
                    "summary".to_string(),
                    node.summary.into_pyobject(py).unwrap().into_any().unbind(),
                );
                d.insert(
                    "children".to_string(),
                    node.children.into_pyobject(py).unwrap().into_any().unbind(),
                );
                d.insert(
                    "date_range".to_string(),
                    vec![node.date_range.0, node.date_range.1]
                        .into_pyobject(py)
                        .unwrap()
                        .into_any()
                        .unbind(),
                );
                d.insert(
                    "entities".to_string(),
                    node.entities.into_pyobject(py).unwrap().into_any().unbind(),
                );
                d.insert(
                    "project".to_string(),
                    node.project.into_pyobject(py).unwrap().into_any().unbind(),
                );
                d
            })
            .collect()
    })
}

fn most_common(items: &[String]) -> Option<String> {
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for item in items {
        *counts.entry(item.as_str()).or_insert(0) += 1;
    }
    counts
        .into_iter()
        .max_by_key(|&(_, c)| c)
        .map(|(s, _)| s.to_string())
}

// ── Tier 3: cluster_notes ──────────────────────────────────────

/// Greedy overlap clustering of notes by shared keywords/entities.
///
/// Input: list of (keywords, entities) tuples for each note.
/// min_overlap: minimum shared terms to merge (default 2).
/// Returns: list of clusters (each a list of note indices), only clusters with 2+ notes.
#[pyfunction]
fn cluster_notes(notes: Vec<(Vec<String>, Vec<String>)>, min_overlap: usize) -> Vec<Vec<usize>> {
    let n = notes.len();
    if n == 0 {
        return Vec::new();
    }

    // Pre-compute keyword sets
    let sets: Vec<HashSet<&str>> = notes
        .iter()
        .map(|(kw, ent)| {
            let mut s: HashSet<&str> = HashSet::new();
            for k in kw {
                s.insert(k.as_str());
            }
            for e in ent {
                s.insert(e.as_str());
            }
            s
        })
        .collect();

    let mut clusters: Vec<Vec<usize>> = Vec::new();
    let mut assigned: HashSet<usize> = HashSet::new();

    for i in 0..n {
        if assigned.contains(&i) {
            continue;
        }
        let mut cluster = vec![i];
        assigned.insert(i);
        let mut i_keywords = sets[i].clone();

        for j in (i + 1)..n {
            if assigned.contains(&j) {
                continue;
            }
            let overlap = i_keywords.intersection(&sets[j]).count();
            if overlap >= min_overlap {
                cluster.push(j);
                assigned.insert(j);
                // Expand keywords (greedy)
                for k in &sets[j] {
                    i_keywords.insert(k);
                }
            }
        }

        if cluster.len() >= 2 {
            clusters.push(cluster);
        }
    }

    clusters
}

#[pymodule]
fn remanentia_consolidation(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(extract_entities, m)?)?;
    m.add_function(wrap_pyfunction!(extract_key_lines, m)?)?;
    m.add_function(wrap_pyfunction!(extract_typed_relations, m)?)?;
    m.add_function(wrap_pyfunction!(parse_frontmatter, m)?)?;
    m.add_function(wrap_pyfunction!(cluster_traces, m)?)?;
    m.add_function(wrap_pyfunction!(build_summary_dag, m)?)?;
    m.add_function(wrap_pyfunction!(cluster_notes, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| (*s).to_string()).collect()
    }

    #[test]
    fn parse_frontmatter_reads_key_values_and_skips_non_pairs() {
        let fm = parse_frontmatter("---\ntitle: Hello\ntags: a b\n- listitem\nnocolon\n---\nbody");
        assert_eq!(fm.get("title"), Some(&"Hello".to_string()));
        assert_eq!(fm.get("tags"), Some(&"a b".to_string()));
        assert_eq!(fm.get("nocolon"), None);
        // No leading fence -> empty map.
        assert!(parse_frontmatter("plain text").is_empty());
    }

    #[test]
    fn date_gap_days_is_absolute_and_zero_on_unparsable() {
        assert_eq!(date_gap_days("2023-06-01", "2023-06-03"), 2);
        assert_eq!(date_gap_days("2023-06-03", "2023-06-01"), 2);
        assert_eq!(date_gap_days("not-a-date", "2023-06-01"), 0);
    }

    #[test]
    fn most_common_returns_the_modal_item() {
        assert_eq!(most_common(&v(&["a", "b", "a"])), Some("a".to_string()));
        assert_eq!(most_common(&[]), None);
    }

    #[test]
    fn cluster_notes_merges_on_shared_terms() {
        let clusters = cluster_notes(
            vec![
                (v(&["x", "y"]), vec![]),
                (v(&["x", "y"]), vec![]),
                (v(&["z"]), vec![]),
            ],
            2,
        );
        // Notes 0 and 1 share 2 terms; note 2 stays alone (single-note cluster dropped).
        assert_eq!(clusters, vec![vec![0, 1]]);
    }

    #[test]
    fn cluster_traces_splits_on_a_gap_over_two_days() {
        let clusters = cluster_traces(vec![
            ("t1".into(), "p".into(), "2023-06-01".into()),
            ("t2".into(), "p".into(), "2023-06-02".into()),
            ("t3".into(), "p".into(), "2023-06-10".into()),
        ]);
        assert_eq!(clusters, vec![vec!["t1", "t2"], vec!["t3"]]);
    }
}
