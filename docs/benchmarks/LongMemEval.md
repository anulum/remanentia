# LongMemEval Benchmark Results

## Current Score (R8)

**69.0% overall** (345/500 questions). Results committed in `data/longmemeval_hypotheses.results.jsonl`.

Method: BM25 retrieval + ArcaneRetriever hybrid architecture + GPT-4o-mini generation and judging.

## Category Breakdown

| Category | Correct | Total | Accuracy |
|----------|--------:|------:|---------:|
| single-session-preference | 27 | 30 | 90.0% |
| single-session-assistant | 49 | 56 | 87.5% |
| knowledge-update | 68 | 78 | 87.2% |
| single-session-user | 58 | 70 | 82.9% |
| multi-session | 82 | 133 | 61.7% |
| temporal-reasoning | 61 | 133 | 45.9% |
| **Overall** | **345** | **500** | **69.0%** |

## Progression Across Rounds

| Round | Score | Change | Key Change |
|-------|------:|-------:|------------|
| R1 | 43.4% | — | First run (Claude-judge) |
| R2 | ~59% | +16pp | Fix Sonnet 404, temporal pre-computation |
| R3 | ~62% | +3pp | Fix preference context |
| R4 | ~64% | +2pp | Python date computation + GPT-4o-mini judge |
| R5 | 67.2% | +3pp | Official protocol (GPT-4o-mini gen+judge) |
| R6 | regressed | — | Counting prompt regressed, reverted |
| R8 | 69.0% | +1.8pp | ArcaneRetriever hybrid architecture |

## Bottleneck Analysis

Temporal-reasoning (45.9%) and multi-session (61.7%) together account for 266 of 500 questions — over half the dataset. These categories drag the overall score.

Getting temporal from 46% to 65% alone would push overall to ~75%.

## Competitive Context

| System | LongMemEval | LLM-dependent |
|--------|:-----------:|:-------------:|
| Remanentia | 69.0% (committed) | Partial (GPT-4o-mini for generation) |
| Hindsight | 91.4% (per their paper) | Yes (GPT-4) |

## Reproduction

```bash
# Full benchmark (requires OPENAI_API_KEY for GPT-4o-mini)
python bench_longmemeval.py --llm --evaluate

# Results file
data/longmemeval_hypotheses.results.jsonl
```

Dataset: `data/longmemeval_oracle.json` (500 questions with haystack sessions).
