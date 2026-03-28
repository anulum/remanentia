# LOCOMO Benchmark Results

## Current Score (without LLM)

**74.7% overall** on 1,986 questions from the LOCOMO multi-session QA dataset.

Method: BM25 + token overlap + answer extraction. No embedding rerank, no LLM.

| Category | Accuracy |
|----------|----------|
| Multi-hop | 82.6% |
| Adversarial | 79.5% |
| Open-domain | 78.7% |
| Single-hop | 55.7% |
| Temporal | 42.7% |
| **Overall** | **74.7%** |

**Caveat:** These numbers are from experiment runs not committed to the repository. The LOCOMO dataset must be obtained separately. Results are not independently reproducible from committed code alone.

## With LLM Synthesis (exp8b, historical)

An experiment run with Haiku LLM synthesis reached 88.5% on 548 questions (a subset). This used:

- BM25(0.4)+Embed(0.6) hybrid retrieval
- Cross-encoder reranking
- 4-stage answer extraction with LLM fallback
- Answer normalisation (hedging strip, polarity match)

This result is not committed and not independently reproducible.

## What Works

- Hybrid retrieval (BM25+MiniLM) with RRF
- Cross-encoder reranking (ms-marco-MiniLM)
- Entity boost gated by person-centricity
- Answer normalisation (yes/no polarity, list overlap)

## What Doesn't Work

- Temporal code execution: 0 LOCOMO hits (questions are counterfactual, not date-math)
- Confidence routing: regression when skipping stages 2-3
- Semantic similarity at 0.8 threshold: too strict, rarely fires

## Competitive Context

Numbers below are from published papers or official documentation. "—" means not reported.

| System | LOCOMO | LongMemEval | LLM-dependent |
|--------|:------:|:-----------:|:-------------:|
| Remanentia | 74.7% (no LLM) | 69.0% (committed) | No (optional) |
| Zep/Graphiti | 94.8% (DMR, per their paper) | — | Yes |
| Hindsight | — | 91.4% (per their paper) | Yes |
| Mem0 | +26% vs baseline (per their docs) | — | Yes |

## Reproduction

```bash
# LOCOMO (requires dataset + optional API key)
python bench_locomo.py           # without LLM
python bench_locomo.py --llm     # with LLM synthesis (requires ANTHROPIC_API_KEY)

# LongMemEval (committed dataset, reproducible)
python bench_longmemeval.py --llm --evaluate
```
