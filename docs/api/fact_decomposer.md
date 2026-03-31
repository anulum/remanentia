# fact_decomposer

Decomposes conversation turns into atomic facts with temporal validity windows,
typed classification, and supersession tracking.

## Purpose

The fact decomposer is the entry point for all structured knowledge extraction
in Remanentia. Raw conversation turns (user/assistant message pairs) are split
into atomic sentences, each classified into one of 9 fact types, annotated with
dates and entities, and tracked for supersession (newer facts invalidating older
ones).

This module feeds directly into:

- **FactIndex** — keyword + entity + temporal search over atomic facts
- **ArcaneRetriever** — 4-channel parallel retrieval with RRF fusion
- **consolidation_engine** — episodic-to-semantic memory compression

Without the fact decomposer, Remanentia cannot answer temporal, multi-session,
or knowledge-update questions. It is the first stage of the retrieval pipeline.

## Fact Type Taxonomy (9 types)

The classifier uses a priority-ordered regex cascade. Higher-priority types are
checked first, so a sentence matching both "decision" and "state" patterns is
classified as "decision" (the more specific type).

| Priority | Type | Description | Example |
|----------|------|-------------|---------|
| 1 | **decision** | Explicit group/individual decisions | "We decided to use BM25 for retrieval." |
| 2 | **correction** | Retractions, fixes to prior beliefs | "Actually, the STDP approach was wrong." |
| 3 | **principle** | Rules, guidelines, invariants | "Always verify coverage before pushing." |
| 4 | **commitment** | Promises, deadlines, obligations | "The deadline is April 15, I committed." |
| 5 | **skill** | Procedures, how-to, workflows | "To fix this, run pytest with -x flag." |
| 6 | **plan** | Future intentions | "I plan to add temporal reasoning." |
| 7 | **preference** | Likes, dislikes, favourites | "I prefer dark mode interfaces." |
| 8 | **state** | State changes (supersedes prior facts) | "She started a new job at Google." |
| 9 | **event** | Default — everything else | "The meeting lasted two hours." |

### Why 9 types instead of 4?

The original taxonomy (state, event, preference, plan) could not distinguish
between:

- A **decision** and a **state change** — "We decided to switch to BM25" was
  classified as "state" because "decided" matched the change-verb pattern. But
  decisions carry different retrieval semantics: they answer "what did we decide"
  queries, not "what changed".

- A **correction** and an **event** — "Actually, the previous result was wrong"
  was classified as "event". But corrections invalidate prior facts and should
  boost in knowledge-update queries.

- A **principle** and an **event** — "Always verify coverage before pushing" is
  a durable rule, not a one-time event. It should surface for "what are our
  guidelines" queries and never expire.

The extended taxonomy was inspired by Engram's 10-category system (fact,
decision, preference, correction, relationship, principle, commitment, moment,
skill, rule). Remanentia adapts 6 of these and merges "fact" into "event" and
"moment" into "event" (no distinct retrieval semantics).

## Pattern Details

Each type is detected by a compiled regex pattern. Patterns are evaluated in
priority order — the first match wins. All patterns are case-insensitive.

### decision

```python
r"\b(we (?:decided|chose|picked|selected|agreed|concluded|resolved)|"
r"decision was|the verdict|final (?:choice|answer|call)|consensus)\b"
```

Triggers on explicit group decision language. "We decided", "consensus",
"the verdict". Does NOT trigger on individual "I decided" — that maps to
"state" via the change-verb pattern.

### correction

```python
r"\b(actually|correction|was wrong|mistake|misunderstood|"
r"should have been|turned out|in fact|not true|incorrect|"
r"clarification|I was wrong|we were wrong)\b"
```

Triggers on retraction/fix language. Critical for knowledge-update queries
where the system must surface the correction, not the original wrong fact.

### principle

```python
r"\b(always (?:do|use|check|ensure|verify)|never (?:do|use|skip|delete)|"
r"rule(?:s)? (?:is|are|of)|principle|best practice|guideline|"
r"must always|must never|invariant|axiom|law of)\b"
```

Triggers on durable guideline language. Principles should never be aged out
(they remain valid indefinitely unless explicitly contradicted).

### commitment

```python
r"\b(I (?:promise|commit|guarantee|owe|agreed to)|"
r"deadline|due (?:by|on|date)|deliverable|"
r"committed to|obligation|must (?:deliver|finish|complete) by)\b"
```

Triggers on obligation/deadline language. Commitments have natural expiry
(the deadline passes), making them good candidates for temporal filtering.

### skill

```python
r"\b(to (?:do this|fix this|run this|build this|deploy|install|configure|set up)|"
r"step (?:1|2|3|one|two)|the (?:command|procedure|workflow|recipe|process) is|"
r"how to|run the following|execute|you need to)\b"
```

Triggers on procedural/how-to language. Skills are the most durable fact
type after principles — a deployment procedure remains valid until the
infrastructure changes.

## Pipeline Integration

The fact decomposer is wired into the Remanentia pipeline at multiple points:

```
Conversation sessions
        │
        ▼
  decompose_sessions()          ← fact_decomposer.py
        │
        ▼
  FactIndex (keyword + entity + temporal search)
        │
        ▼
  ArcaneRetriever._ch_bm25()   ← arcane_retriever.py
  ArcaneRetriever._ch_entity()
  ArcaneRetriever._ch_temporal()
  ArcaneRetriever._ch_session()
        │
        ▼
  _rrf_fusion() with recency decay
        │
        ▼
  build_context() → LLM prompt
```

### Verification

The pipeline wiring is verified by `test_new_types_flow_through_pipeline` which:

1. Creates sessions with sentences matching all 9 fact types
2. Decomposes them via `decompose_sessions()`
3. Builds a `FactIndex`
4. Queries for a decision-type fact
5. Asserts the top result has `fact_type == "decision"`

This is an end-to-end test — if any link in the chain breaks, it fails.

## Performance

Classification of a single sentence completes in under **0.005ms** (5
microseconds). The 9-pattern regex cascade adds negligible overhead compared to
the original 4-pattern version.

Measured performance (1,000 iterations × 9 sentences, `time.perf_counter()`):

| Operation | Time |
|-----------|------|
| `_classify_fact()` per call | <0.005ms |
| `decompose_sessions()` for 8 turns | 0.60ms |
| `FactIndex` construction (8 facts) | 0.09ms |
| `FactIndex.query()` | 0.02ms |
| **Full pipeline** (decompose + index + query) | **0.71ms** |

Budget assertion: `per_call_ms < 0.1` in `test_classify_fact_performance`.

### Rust path

The Rust crate `remanentia_fact_decomposer` provides `classify_fact_type()` and
`split_sentences()` as Python-callable functions via PyO3. The Rust
implementation uses `LazyLock<Regex>` for compiled patterns and achieves ~7×
speedup on large workloads.

Python fallback is preserved — if the Rust crate is not installed, the pure
Python implementation is used with identical semantics.

## Supersession Tracking

When a fact with `fact_type == "state"` contains a change verb (started,
switched, moved, etc.), it is marked `supersedes = True`. The decomposer then
searches for prior state facts about the same entities and sets their
`valid_until` field, creating a temporal validity window.

Example:

```
Session 1: "I work at Microsoft."     → valid_from="", valid_until=""
Session 5: "I started at Google."     → valid_from=session_5_date, supersedes=True
           ↳ Previous fact updated:   → valid_until=session_5_date
```

This enables temporal filtering: a query for "current job" with
`reference_date="session 7"` will return only the Google fact.

## Data Structures

### AtomicFact

```python
@dataclass
class AtomicFact:
    text: str           # The atomic sentence
    session_idx: int    # Which session (0-based)
    turn_idx: int       # Which turn within the session
    role: str           # "user" or "assistant"
    fact_type: str      # One of 9 types (see taxonomy above)
    valid_from: str     # ISO date or "" (still valid from origin)
    valid_until: str    # ISO date or "" (still valid)
    entities: list[str] # Extracted entity names
    date_mentions: list[str]  # ISO dates found in text
    supersedes: bool    # True if this updates a prior fact
```

### FactIndex

In-memory index with two lookup structures:

- `_entity_to_facts`: entity name → list of fact indices (3× boost)
- `_keyword_to_facts`: token → list of fact indices (1× score)

Three specialised query methods:

- `query()` — general retrieval with optional temporal filtering
- `temporal_query()` — boosts facts with date mentions, includes all dated facts for ordering questions
- `cross_session_query()` — ensures session diversity in results

## Usage

```python
from fact_decomposer import decompose_sessions, FactIndex

sessions = [
    [
        {"role": "user", "content": "We decided to use BM25."},
        {"role": "assistant", "content": "I plan to add reranking."},
    ]
]

facts = decompose_sessions(sessions)
idx = FactIndex(facts)

# General query
results = idx.query("what did we decide about retrieval")
for fact, score in results[:3]:
    print(f"[{fact.fact_type}] {fact.text} (score={score})")

# Temporal query with validity filtering
results = idx.query("current job", reference_date="2024-06-15")
```

## Test Coverage

91 tests in `tests/test_fact_decomposer.py`:

- **Tokeniser**: basic, short filter, empty
- **Sentence splitting**: multiple, short filter, single long
- **Date extraction**: ISO, written, MDY, normaliser
- **Entity extraction**: capitalised names, quoted strings
- **Classification**: all 9 types (3+ tests each), priority/edge cases,
  reachability (all types reachable), empty/gibberish
- **Pipeline integration**: types survive decompose → FactIndex → query
- **Performance**: classify < 0.1ms per call
- **Supersession**: state facts update prior facts' valid_until
- **FactIndex**: query, temporal_query, cross_session_query
- **Build fact**: ML classifier fallback, date normaliser

All 6 STRONG dimensions verified: empty, error, negative, pipeline, roundtrip, performance.

## API Reference

::: fact_decomposer.FactIndex
    options:
      show_source: true
      members_order: source

::: fact_decomposer.AtomicFact

::: fact_decomposer.decompose_sessions
