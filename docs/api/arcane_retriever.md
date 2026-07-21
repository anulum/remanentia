# arcane_retriever

4-channel parallel retrieval with Reciprocal Rank Fusion (RRF) and
temporal recency decay.

## Purpose

The ArcaneRetriever is Remanentia's advanced retrieval engine for
structured fact-level search over decomposed conversation sessions. It
complements the main `memory_index.py` BM25 pipeline (which operates on
raw paragraphs) by searching over `AtomicFact` objects with typed
classification, temporal validity, and entity annotations.

The retriever is used by the LongMemEval benchmark pipeline and is the
primary retrieval path for temporal-reasoning and multi-session questions.

## Architecture

```
Query + qtype
      │
      ▼
    GATE (classify query → select channels + weights)
      │
      ▼
    ┌──────────────┬───────────────┬───────────────┬───────────────┐
    │ FAST (BM25)  │ WORKING       │ DEEP (graph)  │ TEMPORAL      │
    │ _ch_bm25()   │ _ch_entity()  │ _ch_session() │ _ch_temporal()│
    └──────┬───────┴───────┬───────┴───────┬───────┴───────┬───────┘
           │               │               │               │
           └───────────────┴───────────────┴───────────────┘
                                   │
                                   ▼
                         RRF FUSION × recency decay
                                   │
                                   ▼
                       SUFFICIENCY CHECK (heuristic)
                                   │
                             ┌─────┴─────┐
                        SUFFICIENT   INSUFFICIENT
                             │           │
                         results     rewrite query
                                     (max 3 iterations)
```

## Channels

### FAST (BM25)

Keyword/token overlap scoring via `FactIndex.query()`. No temporal
filtering — returns all matching facts regardless of validity window.
Activated for all query types.

### WORKING (entity)

Entity-boosted retrieval via `FactIndex.query()`. Entities in the query
receive a 3× score boost. Activated for all query types.

### DEEP (cross-session)

Session-diverse retrieval via `FactIndex.cross_session_query()`. Ensures
results span multiple sessions by applying a 3.0 diversity bonus to the
first fact from each session. Activated for `multi-session` queries.

### TEMPORAL

Date-aware retrieval via `FactIndex.temporal_query()`. Boosts facts with
date mentions (+5.0) and valid_from dates (+2.0). For ordering questions
(first, last, before, after), includes ALL dated facts even without
keyword match. Activated for `temporal-reasoning` and `knowledge-update`
queries.

When the C3 temporal relation classifier is available, results are
further re-scored: facts classified as temporally relevant (before,
after, same_day) receive a 1.3× boost.

## Temporal Recency Decay

New in v0.4. Inspired by Engram's temporal retrieval agent.

### Mechanism

Each fact's RRF score is multiplied by a recency weight:

```
weight = 2^(-age_days / half_life_days)
```

Where:
- `age_days` = reference_date - fact_date (in days)
- `half_life_days` = configurable (default: 30 days)

### Decay curve

| Age (days) | Weight (half_life=30) |
|------------|----------------------|
| 0 (today) | 1.000 |
| 15 | 0.707 |
| 30 | 0.500 |
| 60 | 0.250 |
| 90 | 0.125 |
| 180 | 0.016 |
| 365 | 0.000 (negligible) |

### Date resolution

The recency weight uses the fact's `valid_from` date. If that is empty,
it falls back to the session date (`session_dates[fact.session_idx]`). If
neither is available, the weight defaults to 1.0 (no penalty — unknown
dates are not penalised).

### Configuration

```python
ar = ArcaneRetriever(
    sessions,
    session_dates=["2024-01-01", "2024-06-01"],
    reference_date="2024-06-15",    # "today" for scoring
    recency_half_life_days=30,      # 30-day half-life (default)
)
```

Setting `recency_half_life_days=0` disables decay entirely.
Omitting `reference_date` also disables decay (all weights = 1.0).

### When to use

- **LongMemEval**: Set `reference_date` to the evaluation date. Facts from
  recent sessions score higher, which helps knowledge-update questions.
- **Live usage**: Set `reference_date` to today's date. Recent conversations
  naturally outrank older ones.
- **Historical analysis**: Set `recency_half_life_days=0` to disable decay
  and treat all facts equally.

## RRF Fusion

Reciprocal Rank Fusion combines results from all active channels:

```
score(fact) = recency_weight(fact) × Σ 1/(K + rank_i(fact))
```

Where K = 60 (standard RRF constant). Facts appearing in multiple channels
get higher scores (multi-channel agreement signal).

Deduplication: facts with identical text from different channels are merged
into a single `FusedResult` with combined RRF score and per-channel ranks.

## Sufficiency Loop

After fusion, a heuristic sufficiency check determines whether the
retrieval is good enough:

| Query type | Sufficiency criterion |
|------------|---------------------|
| temporal-reasoning | ≥2 dated facts in top 10 |
| multi-session | ≥2 sessions represented in top 10 |
| counting questions | ≥5 results |
| all types | ≥30% entity overlap with query |

If insufficient, the query is rewritten:

| Reason | Rewrite strategy |
|--------|-----------------|
| missing_dates | Append "date time when day month year" |
| single_session | Append entity names from results |
| insufficient_count | Prefix "all instances of" |
| low_entity_coverage | Simplify to top-5 keywords |

Maximum 3 iterations (original + 2 rewrites).

## Pipeline Integration

```
bench_longmemeval.py
  │
  ▼
ArcaneRetriever(sessions, session_dates)
  │
  ▼
retriever.retrieve(question, qtype, top_k=20)
  │                    ↑
  │    Uses: fact_decomposer.decompose_sessions()
  │          fact_decomposer.FactIndex
  │          temporal_relation.classify_relation (optional C3)
  │
  ▼
retriever.build_context(question, results)
  │
  ▼
LLM prompt → answer
```

The ArcaneRetriever is also available to `memory_recall.py` as an
alternative retrieval path when the `--arcane` flag is set.

## Performance

Measured with 20 sessions, 30-day half-life, `time.perf_counter()`:

| Operation | Time | Budget |
|-----------|------|--------|
| `retrieve()` single query | <50ms | 50ms |
| `_recency_weight()` per fact | <0.001ms | negligible |
| RRF fusion (4 channels) | <1ms | 5ms |
| Full pipeline (decompose + retrieve) | <100ms | 100ms |

The recency decay adds negligible overhead — it is a single `pow()` call
per fact in the fusion step.

## Data Structures

### RetrievalResult

```python
@dataclass
class RetrievalResult:
    fact: AtomicFact   # The retrieved fact
    score: float       # Channel-specific score
    channel: str       # "bm25", "entity", "temporal", "session"
    rank: int          # Rank within the channel (0-based)
```

### FusedResult

```python
@dataclass
class FusedResult:
    fact: AtomicFact          # The retrieved fact
    rrf_score: float          # Fused score (× recency weight)
    channels: list[str]       # Which channels contributed
    per_channel_ranks: dict[str, int]  # Channel → rank
```

## Usage

```python
from arcane_retriever import ArcaneRetriever

sessions = [
    [{"role": "user", "content": "I started at Google in January."}],
    [{"role": "user", "content": "I moved to Microsoft in June."}],
]
dates = ["2024-01-15", "2024-06-01"]

ar = ArcaneRetriever(
    sessions,
    session_dates=dates,
    reference_date="2024-07-01",
    recency_half_life_days=30,
)

# Retrieve with recency decay
results = ar.retrieve("where does the user work now", "knowledge-update")
for r in results[:3]:
    print(f"[{r.fact.fact_type}] {r.fact.text} (rrf={r.rrf_score:.4f})")

# Build context for LLM
context = ar.build_context("current employer", results)
```

## Test Coverage

46 tests in `tests/test_arcane_retriever.py`:

- **Gate**: channel selection for all query types
- **Channels**: BM25, entity, temporal (with C3 classifier), session
- **Parallel retrieve**: multi-channel execution
- **RRF fusion**: score computation, deduplication, ranking
- **Sufficiency**: temporal, multi-session, counting, entity overlap
- **Query rewrite**: all 4 rewrite strategies
- **Roundtrip**: sessions → retrieve → build_context
- **Recency decay**: boost recent, half-life math, no-decay modes,
  invalid dates, empty sessions, same-day, very old
- **Performance**: retrieve < 50ms per query

Test dimensions: empty, error, negative, pipeline, roundtrip, and performance.

## API Reference

::: arcane_retriever.ArcaneRetriever
    options:
      show_source: true
      members_order: source

::: arcane_retriever.RetrievalResult

::: arcane_retriever.FusedResult
