# LongMemEval Benchmark Results

Remanentia reports LongMemEval in two distinct settings. The distinction is
load-bearing:

- **Full-S** is the realistic retrieval setting: each question has roughly 50
  sessions in the haystack and only about two are gold. This exercises retrieval
  and is the setting to compare against published leaderboards.
- **Oracle** gives the reader only the gold sessions. It measures synthesis over
  already-selected context, not retrieval, and is kept only for historical
  continuity.

Per-run history is recorded in `benchmarks/longmemeval_history.jsonl`. Quote
multi-run means, not single runs.

## Current Comparable Score: Full-S

Full-S uses the retrieved-context reader over the realistic large haystack:
`bench_longmemeval.py --full`, top-10 retrieved sessions, cross-encoder rerank on,
current 2026-06 `gpt-4o-mini`.

| Category | Full-S 3-run mean |
|----------|------------------:|
| knowledge-update | 79.5% |
| single-session-user | 73.8% |
| single-session-assistant | 71.4% |
| single-session-preference | 58.9% |
| multi-session | 42.1% |
| temporal-reasoning | 41.8% |
| **Overall** | **56.6%** |

Runs were 57.6%, 55.4%, and 56.8% overall, for a 2.2-point spread. Retrieval
recall@10 is high in the hardest categories (about 88% multi-session and 79%
temporal), so the remaining loss there is mostly synthesis over retrieved
context. Single-session losses are more retrieval-sensitive on the legacy BM25
path.

## Oracle History: R11

The R11 result below is the oracle setting: the haystack is exactly the gold
sessions, fed to the reader in full. It is not comparable to full-S leaderboards.

**Committed snapshot, April 2026** (`data/longmemeval_hypotheses.results.jsonl`):

| Category | Correct | Total | Accuracy |
|----------|--------:|------:|---------:|
| single-session-preference | 27 | 30 | 90.0% |
| single-session-assistant | 49 | 56 | 87.5% |
| single-session-user | 60 | 70 | 85.7% |
| knowledge-update | 66 | 78 | 84.6% |
| temporal-reasoning | 87 | 133 | 65.4% |
| multi-session | 72 | 133 | 54.1% |
| **Overall** | **361** | **500** | **72.2%** |

Current-model reruns of the same oracle path, with cross-encoder rerank restored,
average about 71.2% overall and 60.2% temporal-reasoning on the 2026-06
`gpt-4o-mini`. The roughly one-point overall and five-point temporal gap versus
the April snapshot is model drift, not a Remanentia pipeline regression.

## Progression Across Oracle Rounds

| Round | Score | Change | Temporal | Key change |
|-------|------:|-------:|---------:|------------|
| R1 | 43.4% | - | - | First run with an LLM judge |
| R2 | ~59% | +16pp | - | Fixed Sonnet 404, temporal pre-computation |
| R3 | ~62% | +3pp | - | Fixed preference context |
| R4 | ~64% | +2pp | - | Python date computation + GPT-4o-mini judge |
| R5 | 67.2% | +3pp | - | Official protocol |
| R6 | regressed | - | - | Counting prompt regressed, reverted |
| R8 | 69.0% | +1.8pp | 45.9% | ArcaneRetriever hybrid architecture |
| R9 | 73.0% | +4.0pp | 56.4% | Session-anchored dates + duration arithmetic |
| R10 | 72.8% | -0.2pp | 57.9% | Intraday HH:MM tiebreak |
| R11 | 72.2% | -0.6pp | 65.4% | Inclusive/exclusive durations, question-date anchoring, proximity, narrow chains |

The temporal-reasoning oracle target was achieved in April: 45.9% at R8 to 65.4%
at R11, a 19.5-point / 26-question gain. This does not mean the full-S temporal
problem is solved; full-S now shows the remaining work is synthesis over retrieved
large-haystack context.

## Competitive Context

| System | Setting | LongMemEval | Notes |
|--------|---------|------------:|-------|
| Remanentia | Full-S | 56.6% | 3-run mean, retrieved-context reader |
| Remanentia | Oracle | 72.2% | April R11 snapshot, retrieval bypassed |
| Hindsight | Full-S | 91.4% | Reported in paper; LLM-dependent |

Only the Full-S Remanentia row is comparable to published full-S leaderboard
numbers.

## Reproduction

```bash
# Realistic full-S benchmark; requires OPENAI_API_KEY for GPT-4o-mini.
python bench_longmemeval.py --full --llm --evaluate

# Historical oracle benchmark.
python bench_longmemeval.py --llm --evaluate
```

Datasets:

- `data/longmemeval_s.json` for full-S.
- `data/longmemeval_oracle.json` for oracle.
