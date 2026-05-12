# LOCOMO Benchmark Results

## Committed Result — BM25 + CE rerank + answer extraction + LLM synthesis

**83.1% overall** on 1,986 questions from the LOCOMO multi-session QA dataset.
Source: [`paper/locomo_results.json`](https://github.com/anulum/remanentia/blob/main/paper/locomo_results.json)
(committed 2026-04-17; 927 s wall-clock end-to-end).

| Category    | Correct / Total | Accuracy |
|-------------|----------------:|---------:|
| Multi-hop   | 285 / 321       | 88.8 %   |
| Temporal    | 60 / 96         | 62.5 %   |
| Single-hop  | 207 / 282       | 73.4 %   |
| Adversarial | 731 / 841       | 86.9 %   |
| Open-domain | 368 / 446       | 82.5 %   |
| **Overall** | **1651 / 1986** | **83.1 %** |

**Method:** BM25 + cross-encoder rerank + 4-stage answer extraction
+ LLM synthesis. 10 LOCOMO conversations, preprocessed into
question-answer pairs.

**Caveat:** The LOCOMO dataset is distributed separately and must be
obtained by the reproducer; we do not redistribute it. The
preprocessed form and exact question order used for this run are
checked into `bench_locomo.py`, so given the LOCOMO source the run
is reproducible up to the LLM-sampling envelope (`REMANENTIA_SEED`
pins every local RNG but does not constrain the hosted LLM).

## Historic Score (without LLM, pre-2026-04-17)

Before the current committed run, a no-LLM configuration reached
74.7 % on the same 1,986 questions. Kept here for reference;
superseded by the table above.

| Category    | Accuracy |
|-------------|---------:|
| Multi-hop   | 82.6 %   |
| Adversarial | 79.5 %   |
| Open-domain | 78.7 %   |
| Single-hop  | 55.7 %   |
| Temporal    | 42.7 %   |
| **Overall** | **74.7 %** |

Method: BM25 + token overlap + answer extraction. No embedding rerank, no LLM.

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
| Remanentia | 83.1% (committed, LLM synthesis); 74.7% historic no-LLM | 69.0% (committed) | Optional |
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
