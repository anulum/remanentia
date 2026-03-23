# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# Remanentia — Internal benchmark against actual GOTM corpus

"""50+ queries that test real Remanentia use cases.

Categories:
- decision: recall what was decided and why
- location: find where something is in the codebase
- metric: recall measured numbers
- temporal: recall when something happened
- debugging: recall what went wrong and how it was fixed
- cross_project: recall connections across projects
- factual: recall specific facts

Gold answers are substrings that MUST appear in the top-k results.
No filename leakage: queries use natural language, not filenames.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

QUERIES = [
    # ── Decisions ─────────────────────────────────────────────
    {"q": "why did we remove SNN from the retrieval scoring",
     "gold": ["zero", "signal", "0.00"], "cat": "decision"},
    {"q": "what did we decide about the STDP learning rule for retrieval",
     "gold": ["broken", "discriminat"], "cat": "decision"},
    {"q": "why was the daemon killed",
     "gold": ["contributes nothing", "GPU"], "cat": "decision"},
    {"q": "what was decided about the embedding weight",
     "gold": ["0.45"], "cat": "decision"},
    {"q": "why did we add Rule 11 to CLAUDE.md",
     "gold": ["honesty", "inflation"], "cat": "decision"},
    {"q": "what did the audit decide about hmac.new",
     "gold": ["false positive", "valid"], "cat": "decision"},
    {"q": "what was the decision about customer knowledge base",
     "gold": ["product", "grounded"], "cat": "decision"},

    # ── Metrics ───────────────────────────────────────────────
    {"q": "what is the LOCOMO benchmark accuracy",
     "gold": ["66.4", "50.0", "74.7", "81.2"], "cat": "metric"},
    {"q": "how many entities are in the graph",
     "gold": ["223"], "cat": "metric"},
    {"q": "how many relations are in the entity graph",
     "gold": ["6,434", "6434"], "cat": "metric"},
    {"q": "what was the open-domain BA for director-ai",
     "gold": ["75", "76"], "cat": "metric"},
    {"q": "how many documents are in the unified index",
     "gold": ["1,217", "1217"], "cat": "metric"},
    {"q": "how many paragraphs does the index have",
     "gold": ["15,938", "15938"], "cat": "metric"},
    {"q": "what is the Hindsight SOTA score",
     "gold": ["91.4"], "cat": "metric"},

    # ── Location ──────────────────────────────────────────────
    {"q": "where is the BM25 search implementation",
     "gold": ["memory_index"], "cat": "location"},
    {"q": "where is the consolidation pipeline",
     "gold": ["consolidation_engine"], "cat": "location"},
    {"q": "where are the entity extraction functions",
     "gold": ["entity_extractor"], "cat": "location"},
    {"q": "where is the MCP server code",
     "gold": ["mcp_server"], "cat": "location"},
    {"q": "where is the Kuramoto order parameter computed",
     "gold": ["compute_order_parameter", "kuramoto"], "cat": "location"},

    # ── Temporal ──────────────────────────────────────────────
    {"q": "when was the SNN retrieval failure discovered",
     "gold": ["2026-03-18", "March"], "cat": "temporal"},
    {"q": "when did the Dimits shift convergence happen",
     "gold": ["2026-03-17"], "cat": "temporal"},
    {"q": "when was the revenue strategy discussed",
     "gold": ["2026-03-17"], "cat": "temporal"},
    {"q": "when was the unified index built",
     "gold": ["2026-03-20", "2026-03-22"], "cat": "temporal"},

    # ── Debugging ─────────────────────────────────────────────
    {"q": "what was the root cause of SNN retrieval failure",
     "gold": ["cos", "projection", "dense", "positive", "uniform"], "cat": "debugging"},
    {"q": "why did the STDP LTD mask fail",
     "gold": ["mask", "sign", "inverted", "wrong", "corrected"], "cat": "debugging"},
    {"q": "what caused NaN in the gyrokinetic solver",
     "gold": ["CFL", "NaN"], "cat": "debugging"},
    {"q": "what went wrong with BCPNN retrieval",
     "gold": ["positive", "100%", "zero discrimination"], "cat": "debugging"},

    # ── Cross-project ─────────────────────────────────────────
    {"q": "what connects sc-neurocore and scpn-quantum-control",
     "gold": ["identity", "quantum", "classical"], "cat": "cross_project"},
    {"q": "how does the SCPN framework relate to remanentia",
     "gold": ["substrate", "mathematics", "binding"], "cat": "cross_project"},
    {"q": "what did the phase orchestrator audit find",
     "gold": ["domainpack", "Kuramoto", "Stuart-Landau"], "cat": "cross_project"},

    # ── Factual ───────────────────────────────────────────────
    {"q": "what encoding backends are available",
     "gold": ["hash", "LSH", "embedding", "sentence-transformer"], "cat": "factual"},
    {"q": "what competitors exist in the memory system space",
     "gold": ["Mem0", "Letta", "Zep"], "cat": "factual"},
    {"q": "what is the Kumiho technique",
     "gold": ["prospective", "query", "write time"], "cat": "factual"},
    {"q": "what learning rules were tested for SNN retrieval",
     "gold": ["STDP", "BCPNN", "Hebbian"], "cat": "factual"},
    {"q": "what is the Rust BM25 engine status",
     "gold": ["slower", "Python", "FFI", "overhead"], "cat": "factual"},
    {"q": "what GPU is available locally",
     "gold": ["GTX 1060", "6GB"], "cat": "factual"},
    {"q": "what is the false positive rate in automated audits",
     "gold": ["10%", "false positive"], "cat": "factual"},
    {"q": "what is the Dimits shift",
     "gold": ["zonal flow", "transport", "temperature gradient"], "cat": "factual"},
    {"q": "how does the consolidation pipeline cluster traces",
     "gold": ["project", "date", "2 day"], "cat": "factual"},
    {"q": "what are the 8 query intent types",
     "gold": ["location", "decision", "temporal", "metric"], "cat": "factual"},

    # ── More decisions (harder) ───────────────────────────────
    {"q": "why was TF-IDF replaced with BM25",
     "gold": ["BM25", "paragraph"], "cat": "decision"},
    {"q": "what approach was chosen for memory consolidation",
     "gold": ["heuristic", "LLM-free", "cluster"], "cat": "decision"},
    {"q": "why does the system use paragraph-level indexing instead of document-level",
     "gold": ["paragraph", "85", "50"], "cat": "decision"},

    # ── More metrics ──────────────────────────────────────────
    {"q": "what is the single-hop accuracy on LOCOMO",
     "gold": ["42", "55", "63", "70"], "cat": "metric"},
    {"q": "how long does the index take to build",
     "gold": ["30"], "cat": "metric"},
    {"q": "what is the query latency",
     "gold": ["47", "100", "ms"], "cat": "metric"},

    # ── More debugging ────────────────────────────────────────
    {"q": "what happened with the identity coherence R metric",
     "gold": ["never called", "garbage", "theatre"], "cat": "debugging"},
    {"q": "why does temporal search perform poorly",
     "gold": ["15.6", "42", "temporal", "date"], "cat": "debugging"},

    # ── Architecture ──────────────────────────────────────────
    {"q": "what is the three-stage search pipeline",
     "gold": ["BM25", "bi-encoder", "cross-encoder"], "cat": "factual"},
    {"q": "how does the entity graph work",
     "gold": ["co_occur", "evidence", "weight"], "cat": "factual"},
]


def run_benchmark(use_gpu: bool = False):
    from memory_index import MemoryIndex

    idx = MemoryIndex()
    if not idx.load():
        print("No pre-built index. Building (no GPU)...")
        idx.build(use_gpu_embeddings=False, use_gliner=False)
        idx.save(quantize=False)
    # Skip cross-encoder for benchmark speed (BM25-only)
    idx._cross_encoder = False  # sentinel: skip CE loading
    print(f"Index: {len(idx.documents)} docs, {len(idx.paragraph_index)} paragraphs\n")

    results_by_cat = {}
    total_correct = 0
    total_tested = 0
    failures = []

    for entry in QUERIES:
        q = entry["q"]
        gold = entry["gold"]
        cat = entry["cat"]

        t0 = time.monotonic()
        results = idx.search(q, top_k=5)
        elapsed_ms = (time.monotonic() - t0) * 1000

        # Check if any gold answer substring appears in top-5 results
        combined_text = " ".join(r.snippet + " " + r.name for r in results).lower()
        hit = any(g.lower() in combined_text for g in gold)

        if cat not in results_by_cat:
            results_by_cat[cat] = {"correct": 0, "total": 0}
        results_by_cat[cat]["total"] += 1
        results_by_cat[cat]["correct"] += int(hit)
        total_correct += int(hit)
        total_tested += 1

        if not hit:
            top_name = results[0].name if results else "NO RESULTS"
            failures.append(f"  MISS [{cat}] {q}\n    Expected: {gold}\n    Got: {top_name}")

    print(f"{'='*60}")
    print(f"INTERNAL BENCHMARK ({total_tested} queries)")
    print(f"{'='*60}")
    overall = total_correct / max(total_tested, 1) * 100
    print(f"\nOverall: {total_correct}/{total_tested} ({overall:.1f}%)")
    print(f"\nBy category:")
    for cat, stats in sorted(results_by_cat.items()):
        acc = stats["correct"] / max(stats["total"], 1) * 100
        print(f"  {cat:20s}: {stats['correct']:2d}/{stats['total']:2d} ({acc:.1f}%)")

    if failures:
        print(f"\nFailures ({len(failures)}):")
        for f in failures:
            print(f)

    out = {
        "total_correct": total_correct,
        "total_tested": total_tested,
        "accuracy": round(overall, 1),
        "by_category": results_by_cat,
    }
    out_path = Path(__file__).parent.parent / "paper" / "internal_benchmark.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    run_benchmark(use_gpu="--gpu" in sys.argv)
