// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Remanentia — RRF fusion + entity graph scoring

use pyo3::prelude::*;
use rustc_hash::FxHashMap;

/// Reciprocal Rank Fusion across multiple ranked lists.
///
/// RRF score = sum(1 / (k + rank_i)) for each list where item appears.
/// k=60 is the standard constant from Cormack et al. (2009).
///
/// Returns: list of (para_idx, score) sorted by score descending.
#[pyfunction]
#[pyo3(signature = (ranked_lists, k=60))]
pub fn reciprocal_rank_fusion(
    ranked_lists: Vec<Vec<(usize, f64)>>,
    k: usize,
) -> Vec<(usize, f64)> {
    let mut rrf_scores: FxHashMap<usize, f64> = FxHashMap::default();

    for ranked in &ranked_lists {
        for (rank, &(para_idx, _score)) in ranked.iter().enumerate() {
            *rrf_scores.entry(para_idx).or_insert(0.0) += 1.0 / (k + rank + 1) as f64;
        }
    }

    let mut result: Vec<(usize, f64)> = rrf_scores.into_iter().collect();
    result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    result
}

/// Entity graph score between query entities and trace entities.
///
/// Computes weighted score based on entity connections in the relation graph.
/// Normalised to [0, 1].
///
/// Args:
///     q_entities: entity IDs found in the query
///     t_entities: entity IDs found in the trace filename
///     relations: list of (source, target, weight) tuples
#[pyfunction]
pub fn entity_graph_score(
    q_entities: Vec<String>,
    t_entities: Vec<String>,
    relations: Vec<(String, String, f64)>,
) -> f64 {
    if q_entities.is_empty() || t_entities.is_empty() || relations.is_empty() {
        return 0.0;
    }

    let q_set: rustc_hash::FxHashSet<&str> = q_entities.iter().map(|s| s.as_str()).collect();
    let t_set: rustc_hash::FxHashSet<&str> = t_entities.iter().map(|s| s.as_str()).collect();

    let mut score = 0.0;
    let mut max_w: f64 = 1.0;

    for (src, tgt, w) in &relations {
        let s = src.as_str();
        let t = tgt.as_str();
        if *w > max_w {
            max_w = *w;
        }

        if (q_set.contains(s) && t_set.contains(t)) || (q_set.contains(t) && t_set.contains(s)) {
            score += w;
        } else if q_set.contains(s) && t_set.contains(s) {
            score += w * 0.5;
        } else if q_set.contains(t) && t_set.contains(t) {
            score += w * 0.5;
        }
    }

    let denom = max_w * q_entities.len() as f64;
    if denom < 1e-12 {
        0.0
    } else {
        (score / denom).min(1.0)
    }
}
