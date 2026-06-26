// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Remanentia — Topology (Betti numbers via petgraph)

use petgraph::algo::connected_components;
use petgraph::graph::UnGraph;
use pyo3::prelude::*;

/// Compute Betti-1 (first Betti number) = E - V + C for an undirected graph.
fn betti_1(num_nodes: usize, edges: &[(usize, usize)]) -> usize {
    if num_nodes == 0 {
        return 0;
    }

    let mut graph = UnGraph::<(), ()>::new_undirected();
    let nodes: Vec<_> = (0..num_nodes).map(|_| graph.add_node(())).collect();

    for &(u, v) in edges {
        if u < num_nodes && v < num_nodes {
            graph.add_edge(nodes[u], nodes[v], ());
        }
    }

    let e = graph.edge_count();
    let v = graph.node_count();
    let c = connected_components(&graph);

    // Betti-1 (first Betti number) = E - V + C for undirected graphs
    (e + c).saturating_sub(v)
}

/// Normalised persistence metric: ln(1 + B₁) / ln(1 + V).
fn persistence(num_nodes: usize, edges: &[(usize, usize)]) -> f64 {
    let b1 = betti_1(num_nodes, edges) as f64;
    if num_nodes <= 1 {
        return 0.0;
    }
    (1.0 + b1).ln() / (1.0 + num_nodes as f64).ln()
}

#[pyfunction]
fn calculate_betti_1(num_nodes: usize, edges: Vec<(usize, usize)>) -> PyResult<usize> {
    Ok(betti_1(num_nodes, &edges))
}

#[pyfunction]
fn calculate_persistence(num_nodes: usize, edges: Vec<(usize, usize)>) -> PyResult<f64> {
    Ok(persistence(num_nodes, &edges))
}

#[pymodule]
fn remanentia_topology(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(calculate_betti_1, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_persistence, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Betti-1 tests ──────────────────────────────────────────────

    #[test]
    fn test_empty_graph() {
        // V=0, E=[] → B₁=0
        assert_eq!(betti_1(0, &[]), 0);
    }

    #[test]
    fn test_single_node() {
        // V=1, E=[] → B₁=0
        assert_eq!(betti_1(1, &[]), 0);
    }

    #[test]
    fn test_simple_tree() {
        // V=4, E=[(0,1),(1,2),(2,3)] → tree, B₁=0
        assert_eq!(betti_1(4, &[(0, 1), (1, 2), (2, 3)]), 0);
    }

    #[test]
    fn test_simple_cycle() {
        // V=3, E=[(0,1),(1,2),(2,0)] → triangle, B₁=1
        assert_eq!(betti_1(3, &[(0, 1), (1, 2), (2, 0)]), 1);
    }

    #[test]
    fn test_complete_k4() {
        // K4: V=4, E=6 → B₁ = 6 - 4 + 1 = 3
        let edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
        assert_eq!(betti_1(4, &edges), 3);
    }

    #[test]
    fn test_disconnected_components() {
        // V=4, E=[(0,1),(2,3)] → 2 components, B₁ = 2 - 4 + 2 = 0
        assert_eq!(betti_1(4, &[(0, 1), (2, 3)]), 0);
    }

    #[test]
    fn test_two_cycles() {
        // Two disjoint triangles: V=6, E=6, C=2 → B₁ = 6 - 6 + 2 = 2
        let edges = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)];
        assert_eq!(betti_1(6, &edges), 2);
    }

    #[test]
    fn test_out_of_bounds_edge_ignored() {
        // V=2, E=[(0,10)] → edge ignored (10 >= V), B₁=0
        assert_eq!(betti_1(2, &[(0, 10)]), 0);
    }

    // ── Persistence tests ──────────────────────────────────────────

    #[test]
    fn test_persistence_monotonicity() {
        // P(cycle) > P(tree) > P(empty)
        let p_empty = persistence(1, &[]);
        let p_tree = persistence(4, &[(0, 1), (1, 2), (2, 3)]);
        let p_cycle = persistence(3, &[(0, 1), (1, 2), (2, 0)]);
        assert!(
            p_cycle > p_tree,
            "cycle ({p_cycle}) should > tree ({p_tree})"
        );
        assert!(
            p_tree >= p_empty,
            "tree ({p_tree}) should >= empty ({p_empty})"
        );
    }

    #[test]
    fn test_persistence_bounds() {
        // 0 ≤ P < 1 for any reasonable graph
        let cases: Vec<(usize, Vec<(usize, usize)>)> = vec![
            (0, vec![]),
            (1, vec![]),
            (3, vec![(0, 1), (1, 2), (2, 0)]),
            (4, vec![(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]),
        ];
        for (v, edges) in cases {
            let p = persistence(v, &edges);
            assert!(p >= 0.0 && p < 1.0, "P={p} out of bounds for V={v}");
        }
    }
}
