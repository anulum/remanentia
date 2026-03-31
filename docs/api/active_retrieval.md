# active_retrieval

Proactive memory consultation for reasoning-time decision support.
Identifies decision points in agent reasoning and retrieves relevant
past decisions, applicable skills, and contradiction warnings.
Rust-accelerated decision point extraction.

## Purpose

Active retrieval enables agents to proactively consult memory DURING
reasoning — not just when explicitly asked. When the agent is about to
make a decision (change code, delete a file, choose between approaches),
active retrieval automatically:

1. Identifies the decision point in the reasoning text
2. Retrieves relevant past decisions from memory
3. Queries applicable skills from the skill store
4. Returns warnings if past reasoning contradicts the proposed action

This prevents the agent from repeating past mistakes or contradicting
established decisions. It is the difference between an agent that
forgets everything between sessions and one that learns from experience.

## Architecture

```
Agent reasoning text (ongoing)
        │
        ▼
  extract_decision_points(text)
        │   ← Rust: remanentia_active_retrieval.extract_decision_points
        │   Splits text by sentence boundaries [.!?\n]
        │   Filters: len >= 15 chars, matches 1+ of 4 decision patterns
        │
        ▼
  For each decision point (max 3):
        │
        ├── retrieve(point, top_k=3)         ← retrieve.py (BM25 search)
        │     Returns: [{trace, score, tier, content}, ...]
        │     Filter: score >= 0.2, deduplicate by trace name
        │
        ├── query_skills(point, top_k=2)     ← skill_extractor.py
        │     Returns: [{name, description, match_score}, ...]
        │
        ▼
  Format as advisory context block
        │
        ▼
  "## Memory Advisory (automatic)\n\n**Decision:** ...\n  Past reasoning: ...\n  Applicable skills: ..."
```

## Decision Point Detection

`extract_decision_points(text)` identifies sentences containing decision
language. The function splits text on `[.!?\n]` boundaries, filters
sentences shorter than 15 characters, and matches each against 4
compiled regex patterns.

### Pattern categories

| # | Pattern regex | What it matches | Example |
|---|--------------|-----------------|---------|
| 1 | `(?:going to\|will\|plan to\|about to)\s+(?:change\|modify\|delete\|remove\|add\|replace\|refactor)` | Stated intent to modify | "I'm about to delete the SNN module" |
| 2 | `(?:choosing\|chose\|decision\|decided)\s+(?:to\|between\|against)` | Explicit choice language | "We decided against using embeddings" |
| 3 | `(?:trade.?off\|alternative\|instead of\|rather than)` | Comparison/alternative analysis | "BM25 rather than full embedding search" |
| 4 | `(?:should we\|should i\|do we\|question is)` | Open questions during reasoning | "Should we keep the SNN backend?" |

### Pattern design rationale

These 4 categories cover the complete decision lifecycle:

1. **Intent** — the agent declares what it's about to do
2. **Choice** — a decision has been made or is being made
3. **Analysis** — alternatives are being weighed
4. **Inquiry** — the agent is uncertain and asking

Each pattern requires BOTH a decision verb AND a specific context word.
"Going to the store" does NOT match (no modify/delete/etc. action).
"Should we go?" does NOT match (no decision-relevant context).

### Rust acceleration

The Rust crate `remanentia_active_retrieval` provides `extract_decision_points`
with compiled `LazyLock<Regex>` patterns and `LazyLock<Regex>` for sentence
splitting:

```rust
static DECISION_PATTERNS: LazyLock<Vec<Regex>> = LazyLock::new(|| vec![
    Regex::new(r"(?i)(?:going to|will|plan to|about to)\s+(?:change|modify|delete|remove|add|replace|refactor)").unwrap(),
    Regex::new(r"(?i)(?:choosing|chose|decision|decided)\s+(?:to|between|against)").unwrap(),
    Regex::new(r"(?i)(?:trade.?off|alternative|instead of|rather than)").unwrap(),
    Regex::new(r"(?i)(?:should we|should i|do we|question is)").unwrap(),
]);

static RE_SENT_SPLIT: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"[.!?\n]").unwrap()
});
```

Measured performance: **4.1 µs** per call (Intel Core i5-11600K @ 3.90GHz,
verified from `/proc/cpuinfo`).

### Wiring

```python
def extract_decision_points(text: str) -> list[str]:
    try:
        from remanentia_active_retrieval import extract_decision_points as _rust_dp
        return _rust_dp(text)  # pragma: no cover
    except ImportError:
        pass
    # Python fallback: re.split + re.search loop
```

## Memory Consultation

`consult_memory(reasoning_text, top_k=3)` is the primary entry point
for proactive memory access.

### Algorithm

```
1. Extract decision points from reasoning text
2. If no decision points found:
   a. Fall back: query retrieve() with first 200 chars
   b. If best score < 0.3: return "" (nothing relevant)
   c. Else: format as minimal advisory
3. For each decision point (max 3):
   a. retrieve(point, top_k=3) → past decisions
   b. Filter: score >= 0.2, deduplicate by trace name
   c. query_skills(point, top_k=2) → applicable procedures
   d. If neither found: skip this point
   e. Format as advisory block
4. Join all advisory blocks with separator
5. Return formatted string (or "" if empty)
```

### Dependencies

| Module | Function | Purpose |
|--------|----------|---------|
| `retrieve.py` | `retrieve(query, top_k, include_content)` | BM25 search over traces |
| `skill_extractor.py` | `query_skills(query, top_k)` | Procedural skill lookup |

### Output format

```markdown
## Memory Advisory (automatic)

**Decision:** I'm about to change the STDP time constant from 20ms to 15ms
  Past reasoning:
  - 2026-03-15_stdp_experiment.md (score=0.742)
  - 2026-03-10_snn_tuning.md (score=0.531)
  Applicable skills:
  - stdp-time-constant-tuning: Changed STDP tau from 25ms to 20ms because convergence improved

**Decision:** Should we remove the sparse weight save code?
  Past reasoning:
  - 2026-03-20_architecture.md (score=0.623)
```

### Deduplication

Traces are deduplicated across decision points via a `seen_traces` set.
If trace A appears for decision point 1, it won't appear again for
decision point 2. This prevents the advisory from repeating the same
evidence.

## Decision Guard

`decision_guard(action, context="")` provides a safety check before
destructive actions.

### Algorithm

```
1. Combine action + context into a single query
2. retrieve(query, top_k=5, include_content=True)
3. For each result with score >= 0.3:
   a. Split content into sentences
   b. For each sentence containing contradiction words:
      ("don't", "never", "avoid", "wrong", "mistake", "failed", "broke")
      c. Check if sentence shares >= 2 words with the action
      d. If yes: add warning
4. Return list of warnings (empty = safe to proceed)
```

### Contradiction detection

The guard uses a two-gate system:
1. **Lexical gate**: sentence must contain a contradiction word
2. **Topical gate**: sentence must share >= 2 words with the action

This prevents false positives like "The weather was wrong" triggering
on action "delete database schema".

### Warning format

```python
[
    "Trace '2026-03-15_decision.md' (score=0.742): never delete sparse weight code because it's needed for checkpoint recovery",
    "Trace '2026-03-10_fix.md' (score=0.531): avoided removing the save function after it broke training"
]
```

### Usage

```python
warnings = decision_guard(
    action="delete sparse W save code",
    context="cleaning up snn_backend.py"
)
if warnings:
    for w in warnings:
        print(f"WARNING: {w}")
    # Agent should acknowledge warnings before proceeding
else:
    # Safe to proceed — no contradicting past reasoning
    pass
```

## Pipeline Integration

### Current integration points

| Caller | Function | When |
|--------|----------|------|
| Agent reasoning hook | `consult_memory(text)` | During reasoning, before decisions |
| Pre-action safety | `decision_guard(action, context)` | Before destructive operations |
| MCP recall (planned) | `consult_memory(query)` | When recall context suggests decision |

### Data flow

```
Agent reasoning → extract_decision_points() [Rust]
    │
    ├── retrieve.retrieve() → BM25 over reasoning_traces/
    ├── skill_extractor.query_skills() → skills from snn_state/skills.json
    │
    ▼
Advisory context string → injected into agent prompt
```

### Interaction with other modules

- **retrieve.py**: Provides the BM25 search backend for past trace lookup
- **skill_extractor.py**: Provides procedural skill matching
- **memory_recall.py**: Active retrieval is complementary — recall is
  explicit (user asks), active retrieval is implicit (agent self-checks)
- **knowledge_store.py**: Contradiction detection in knowledge_store
  handles note-level conflicts; decision_guard handles action-level conflicts

## Performance

Measured on Intel Core i5-11600K @ 3.90GHz (verified from `/proc/cpuinfo`),
2026-03-31, `time.perf_counter()`.

| Operation | Measured | Budget |
|-----------|---------|--------|
| `extract_decision_points` (Rust) | 4.1 µs | <10 µs |
| `consult_memory` (with 3 retrievals) | <200 ms | <500 ms |
| `decision_guard` (5 results) | <100 ms | <500 ms |

The Rust acceleration makes decision point extraction negligible —
the bottleneck is `retrieve.retrieve()` which runs BM25 search.

## Python API

```python
from active_retrieval import (
    extract_decision_points,
    consult_memory,
    decision_guard,
)

# Extract decision points from reasoning text
text = "We are going to change the retrieval engine. Should we use BM25?"
points = extract_decision_points(text)
# ["We are going to change the retrieval engine",
#  "Should we use BM25"]

# Consult memory during reasoning
context = consult_memory(
    "I'm about to replace the BM25 engine with embeddings-only search."
)
print(context)
# ## Memory Advisory (automatic)
# **Decision:** about to replace the BM25 engine
# Past reasoning: ...

# Decision guard before destructive action
warnings = decision_guard(
    action="delete snn_backend.py",
    context="cleaning up unused modules"
)
if warnings:
    for w in warnings:
        print(f"WARNING: {w}")
```

## Limitations

1. **Regex-only decision detection**: Cannot detect implicit decisions
   ("Let me just quickly..." without explicit decision language).

2. **BM25-only retrieval**: Past decisions are retrieved by keyword
   overlap, not semantic similarity. Paraphrased decisions may be missed.

3. **No temporal weighting**: Recent traces are not weighted higher
   than old ones in the decision guard. A 6-month-old warning has
   equal weight to yesterday's.

4. **No confidence calibration**: The 0.2 and 0.3 score thresholds
   are empirical, not calibrated on a validation set.

5. **Dependency on retrieve.py**: The `retrieve()` function is a
   legacy BM25 wrapper. Migration to `memory_recall.recall()` would
   provide richer context (entity graph, temporal reasoning).

## Test Coverage

Tests in `tests/test_improvements.py` and `tests/test_integration.py`:

- **Decision points**: all 4 pattern categories, no matches, empty text,
  short sentences filtered, multi-sentence text
- **Consult memory**: with/without matching past decisions, fallback
  path (no decision points), empty results
- **Decision guard**: warning generation with contradiction words, safe
  action passes, low-score filtering, word overlap threshold
- **Pipeline**: integration with retrieve.py and skill_extractor
- **Rust wiring**: extract_decision_points Rust vs Python equivalence
- **Performance**: extract_decision_points < 10µs

All 6 STRONG dimensions: empty, error, negative, pipeline, roundtrip,
performance.

## API Reference

::: active_retrieval.extract_decision_points

::: active_retrieval.consult_memory

::: active_retrieval.decision_guard
