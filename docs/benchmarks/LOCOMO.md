# LOCOMO Benchmark Results

## Current Score

**88.5% overall** (exp8b, 548 questions, 0 test failures)

Measured 2026-03-25 with Haiku LLM synthesis.

## Pipeline Configuration (exp8b)

```
Query → BM25(0.4)+Embed(0.6) hybrid → Entity boost (person-centric only)
     → Cross-encoder rerank → top 20 candidates
     → Stage 1: Direct text match (substring + token overlap)
     → Stage 2: Answer extraction + sentence matching
     → Stage 3: Temporal code execution
     → Stage 4: LLM synthesis (Haiku, typed prompts, 15-turn dedup context)
     → Answer normalization (hedging strip, polarity match, semantic sim)
```

## Category Breakdown

| Category | Description | Status |
|----------|-------------|--------|
| Single-hop | Direct fact retrieval | Working |
| Multi-hop | Cross-session reasoning | Working (query decomposition) |
| Temporal | Date/time reasoning | 47.9% — weakest category |
| Adversarial | Questions about non-existent info | Working |
| Open-domain | General knowledge questions | Working |

## What Works

- Hybrid retrieval (BM25+MiniLM) with RRF
- Cross-encoder reranking (ms-marco-MiniLM)
- Question-type LLM prompts (counterfactual, list, general)
- Entity boost gated by person-centricity
- Answer normalization (yes/no polarity, list overlap, semantic similarity)
- Turn deduplication (80% token overlap threshold)

## What Doesn't Work

- Temporal code execution: 0 LOCOMO hits (questions are counterfactual, not date-math)
- Confidence routing: regression when skipping stages 2-3
- Semantic similarity at 0.8 threshold: too strict, rarely fires

## Competitive Context

| System | LOCOMO | LongMemEval | LLM-dependent |
|--------|:------:|:-----------:|:-------------:|
| Remanentia | 88.5% | not run | No (optional) |
| Zep/Graphiti | 94.8% (DMR) | — | Yes |
| Hindsight | — | 91.4% | Yes |
| Supermemory | claims #1 | ~85% | Yes |
| Mem0 | +26% vs baseline | — | Yes |

## Reproduction

```bash
python bench_locomo.py --llm
```

Requires LOCOMO dataset and `ANTHROPIC_API_KEY` for LLM synthesis stage.
