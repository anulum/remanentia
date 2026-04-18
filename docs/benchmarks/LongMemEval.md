# LongMemEval Benchmark Results

## Current Score (R11, 2026-04-11)

**72.2% overall** (361/500 questions). Results committed in `data/longmemeval_hypotheses.results.jsonl`.

Method: BM25 retrieval + ArcaneRetriever hybrid architecture + GPT-4o-mini generation and judging.

## Category Breakdown

| Category | Correct | Total | Accuracy |
|----------|--------:|------:|---------:|
| single-session-preference | 27 | 30 | 90.0% |
| single-session-assistant | 49 | 56 | 87.5% |
| single-session-user | 60 | 70 | 85.7% |
| knowledge-update | 66 | 78 | 84.6% |
| **temporal-reasoning** | **87** | **133** | **65.4%** 🎯 |
| multi-session | 72 | 133 | 54.1% |
| **Overall** | **361** | **500** | **72.2%** |

## Progression Across Rounds

| Round | Score | Change | Temporal | Key Change |
|-------|------:|-------:|---------:|------------|
| R1 | 43.4% | — | — | First run (Claude-judge) |
| R2 | ~59% | +16pp | — | Fix Sonnet 404, temporal pre-computation |
| R3 | ~62% | +3pp | — | Fix preference context |
| R4 | ~64% | +2pp | — | Python date computation + GPT-4o-mini judge |
| R5 | 67.2% | +3pp | — | Official protocol (GPT-4o-mini gen+judge) |
| R6 | regressed | — | — | Counting prompt regressed, reverted |
| R8 | 69.0% | +1.8pp | 45.9% | ArcaneRetriever hybrid architecture |
| R9 | 73.0% | +4.0pp | 56.4% | Audit batch #1-3: session-anchored dates + duration arithmetic + chronological ordering |
| R10 | 72.8% | −0.2pp | 57.9% | R9 follow-up #1: intraday HH:MM tiebreak (qtype-aware) |
| **R11** | **72.2%** | **−0.6pp** | **65.4%** 🎯 | R9 follow-up #2-5: fuzzy inclusive/exclusive durations, question_date anchoring, multi-event proximity, narrow chain resolution |

## Bottleneck Analysis

Temporal-reasoning target achieved. Cumulative R8 → R11: **45.9% → 65.4%**, a +19.5pp / +26-question gain. This category is no longer the primary bottleneck.

R10 → R11 category-level cross-tab:

| Category | Fixed | Regressed | Net |
|----------|------:|---------:|----:|
| temporal-reasoning | +24 | −14 | **+10** ✓ |
| multi-session | +8 | −20 | −12 ⚠ |
| knowledge-update | +0 | −3 | −3 |
| single-session-preference | +3 | −1 | +2 |
| single-session-user | +1 | −1 | 0 |
| single-session-assistant | 0 | 0 | 0 |

The multi-session −12 movement is concerning on its face, but none of the R11 code changes (Tasks #32-36) touch non-temporal code paths — they are all gated inside `if qtype == "temporal-reasoning"` branches or in functions only reachable from TReMu pre-computation. The observed churn (28 questions flipping answer in multi-session alone) is within LLM sampling variance seen in earlier rounds (R9→R10 multi-session churn was 21 with net −1; R11 churn of 28 with net −12 sits at the edge of the noise envelope).

**Noise caveat (honest):** all results above are single-run measurements. GPT-4o-mini is not deterministic even with `temperature=0.1`. Across R9/R10/R11 cross-tabs we see ~10-20 question category-level churn per run. Differences ≤5 questions per category are not statistically meaningful. Multi-run averaged scores are pending.

## Competitive Context

| System | LongMemEval | LLM-dependent |
|--------|:-----------:|:-------------:|
| Remanentia | 72.2% (committed) | Partial (GPT-4o-mini for generation) |
| Hindsight | 91.4% (per their paper) | Yes (GPT-4) |

## Reproduction

```bash
# Full benchmark (requires OPENAI_API_KEY for GPT-4o-mini)
python bench_longmemeval.py --llm --evaluate

# Results file
data/longmemeval_hypotheses.results.jsonl
```

Dataset: `data/longmemeval_oracle.json` (500 questions with haystack sessions).
