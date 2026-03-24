# Remanentia 10x Plan — From 81.2% to 90%+ LOCOMO

**Date:** 2026-03-24
**Author:** Arcane Sapience
**Status:** EXECUTING

## Current Position

- LOCOMO: 81.2% (1612/1986) — #5 overall, #1 zero-LLM
- Weakest: temporal 47.9%, single-hop 70.9%
- Best: multi-hop 86.6%, adversarial 86.1%
- Gap to SOTA (Kumiho 93.3%): 12.1pp — entirely answer generation, not retrieval

## Three Interventions (ranked by measured impact)

### Phase 1: LLM-Powered Prospective Indexing
**Expected impact: +5-8pp LOCOMO**
**Evidence: Kumiho achieves 98.5% recall with this technique**

At build time, for each paragraph, call Claude Haiku to generate 3-5 hypothetical future queries.
We already have `_generate_prospective_queries()` with templates. Upgrade to LLM.

**Files:**
- `memory_index.py` — add `_generate_prospective_queries_llm()`, wire into `build()` when `use_llm_indexing=True`
- `pyproject.toml` — already has anthropic dep

**Cost:** ~$1-2 for 15K paragraphs (one-time at build)

**Verification:** Rebuild index with LLM queries, run LOCOMO, compare to 81.2% baseline.

### Phase 2: Multi-Paragraph Answer Synthesis
**Expected impact: +5-8pp LOCOMO**
**Evidence: The entire gap between retrieval systems (81%) and SOTA (93%) is answer generation**

Current `llm_extract_answer` sends one paragraph. Change to: send top-3 paragraphs + query to Haiku, get a synthesized answer with citations.

**Files:**
- `answer_extractor.py` — add `llm_synthesize_answer(query, paragraphs)` that takes multiple paragraphs
- `memory_index.py` — in search(), after collecting results, call synthesis on top-3 if `use_llm=True`

**Cost:** ~$0.003 per query

**Verification:** Run LOCOMO with synthesis enabled, measure per-category improvement.

### Phase 3: Temporal Knowledge Graph
**Expected impact: +15-25pp on temporal category (47.9% → 75%+)**
**Evidence: Mem0g scores 58.1% on temporal with graph. Zep uses bitemporal edges.**

At index time, parse all dates and events. Build temporal edges (before/after/same_day).
For temporal queries, augment BM25 results with graph traversal.

**Files:**
- `temporal_graph.py` — new module: parse dates, build edges, traverse
- `memory_index.py` — integrate temporal graph into search() for temporal intent queries
- `consolidation_engine.py` — emit temporal edges during consolidation

**Cost:** $0 (no LLM needed)

**Verification:** Run LOCOMO temporal category, measure improvement from 47.9%.

## Execution Order

1. Phase 1 (prospective indexing) — highest measured impact, fastest to implement
2. Phase 2 (answer synthesis) — closes the answer generation gap
3. Phase 3 (temporal graph) — fixes weakest category

## Success Criteria

- [ ] LOCOMO overall: 90%+ (from 81.2%)
- [ ] LOCOMO temporal: 75%+ (from 47.9%)
- [ ] All 344+ tests pass, 100% coverage maintained
- [ ] Zero-LLM mode still works (LLM is opt-in enhancement)
- [ ] Measured on actual LOCOMO dataset, not self-authored queries
