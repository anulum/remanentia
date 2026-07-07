// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Remanentia — Rust entity extractor + relation detector (regex fallback)

use pyo3::prelude::*;
use regex::Regex;
use std::sync::LazyLock;

static RE_VER: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"v\d+\.\d+(?:\.\d+)?").unwrap());
static RE_FILE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[\w/\\]+\.(?:py|rs|md|json|yaml|toml)\b").unwrap());

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

/// Regex entity extraction. Returns list of (text, label, score).
#[pyfunction]
fn regex_entities(text: &str) -> Vec<(String, String, f64)> {
    let mut out = Vec::new();
    let lo = text.to_lowercase();
    let pats: &[(&str, &[&str])] = &[
        (
            "project",
            &[
                "director-ai",
                "sc-neurocore",
                "scpn-control",
                "scpn-fusion",
                "scpn-phase-orchestrator",
                "scpn-quantum-control",
                "remanentia",
            ],
        ),
        (
            "algorithm",
            &[
                "stdp",
                "lif",
                "kuramoto",
                "hopfield",
                "bcpnn",
                "csdp",
                "tf-idf",
                "bm25",
                "stuart-landau",
            ],
        ),
        ("hardware", &["gpu", "cuda", "loihi", "gtx 1060"]),
        (
            "software tool",
            &["pytorch", "numpy", "scipy", "fastapi", "docker"],
        ),
    ];
    for (label, terms) in pats {
        for t in *terms {
            if lo.contains(t) {
                out.push((t.to_string(), label.to_string(), 0.5));
            }
        }
    }
    for m in RE_VER.find_iter(text) {
        out.push((m.as_str().into(), "version number".into(), 0.6));
    }
    for m in RE_FILE.find_iter(text) {
        let n = m.as_str().rsplit('/').next().unwrap_or(m.as_str());
        let n = n.rsplit('\\').next().unwrap_or(n);
        if n.len() > 3 {
            out.push((n.into(), "file path".into(), 0.5));
        }
    }
    out
}

/// Extract typed relations. Returns (source, target, relation_type, evidence).
#[pyfunction]
fn extract_relations(
    text: &str,
    entity_texts: Vec<String>,
) -> Vec<(String, String, String, String)> {
    let lo = text.to_lowercase();
    let mut rels = Vec::new();
    for i in 0..entity_texts.len() {
        for j in (i + 1)..entity_texts.len() {
            let (t1, t2) = (&entity_texts[i], &entity_texts[j]);
            let (p1, p2) = match (lo.find(&t1.to_lowercase()), lo.find(&t2.to_lowercase())) {
                (Some(a), Some(b)) => (a, b),
                _ => continue,
            };
            let s = p1.min(p2);
            let e = (p1 + t1.len()).max(p2 + t2.len()).min(text.len());
            let between = &text[s..e];
            let mut found = false;
            for (pat, rt) in RELATION_PATTERNS.iter() {
                if pat.is_match(between) {
                    let ev = if between.len() > 200 {
                        &between[..200]
                    } else {
                        between
                    };
                    rels.push((t1.clone(), t2.clone(), rt.to_string(), ev.into()));
                    found = true;
                    break;
                }
            }
            if !found && (p1 as i64 - p2 as i64).unsigned_abs() < 500 {
                rels.push((t1.clone(), t2.clone(), "co_occurs".into(), String::new()));
            }
        }
    }
    rels
}

#[pymodule]
fn remanentia_entity_extractor(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(regex_entities, m)?)?;
    m.add_function(wrap_pyfunction!(extract_relations, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regex_entities_labels_known_terms_versions_and_files() {
        let ents = regex_entities("We use PyTorch and BM25 in Remanentia v1.2.3, see foo.py");
        assert!(ents.contains(&("pytorch".into(), "software tool".into(), 0.5)));
        assert!(ents.contains(&("bm25".into(), "algorithm".into(), 0.5)));
        assert!(ents.contains(&("remanentia".into(), "project".into(), 0.5)));
        assert!(ents.contains(&("v1.2.3".into(), "version number".into(), 0.6)));
        assert!(ents.contains(&("foo.py".into(), "file path".into(), 0.5)));
    }

    #[test]
    fn regex_entities_returns_empty_for_unremarkable_text() {
        assert!(regex_entities("nothing notable in this sentence").is_empty());
    }

    #[test]
    fn extract_relations_detects_typed_relation() {
        let rels = extract_relations(
            "the bug was fixed by the patch",
            vec!["bug".into(), "patch".into()],
        );
        assert!(rels
            .iter()
            .any(|r| r.0 == "bug" && r.1 == "patch" && r.2 == "fixed_by"));
    }

    #[test]
    fn extract_relations_falls_back_to_co_occurrence() {
        let rels = extract_relations(
            "alpha stands near beta",
            vec!["alpha".into(), "beta".into()],
        );
        assert_eq!(rels.len(), 1);
        assert_eq!(rels[0].2, "co_occurs");
    }

    #[test]
    fn extract_relations_skips_entities_absent_from_text() {
        assert!(extract_relations("unrelated prose", vec!["x".into(), "y".into()]).is_empty());
    }
}
