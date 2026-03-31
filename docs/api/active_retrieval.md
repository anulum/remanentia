# active_retrieval

Proactive memory consultation for reasoning-time decision support.
Identifies decision points in agent reasoning and retrieves relevant
past decisions and skills. Rust-accelerated decision point extraction.

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
established decisions.

## Architecture

```
Agent reasoning text
        │
        ▼
  extract_decision_points(text)     ← Rust: remanentia_active_retrieval
        │
        ▼
  For each decision point:
        │
        ├── memory_recall.recall(point)  → past decisions
        ├── skill_extractor.query_skills(point) → applicable procedures
        │
        ▼
  consult_memory(reasoning_text)
        │
        ▼
  Formatted context string injected into agent prompt
```

## Decision Point Detection

`extract_decision_points(text)` identifies sentences containing decision
language. 4 pattern categories:

| Pattern | Matches |
|---------|---------|
| Action intent | "going to change", "will modify", "about to delete" |
| Choice language | "choosing between", "decided to", "chose against" |
| Trade-off analysis | "trade-off", "alternative", "instead of" |
| Questions | "should we", "should I", "do we", "question is" |

### Rust acceleration

The Rust crate `remanentia_active_retrieval` provides `extract_decision_points`
with compiled `LazyLock<Regex>` patterns:

| Function | Measured | Wired into |
|----------|---------|------------|
| `extract_decision_points` | 1.0 µs | `active_retrieval.extract_decision_points` |

```rust
static DECISION_PATTERNS: LazyLock<Vec<Regex>> = LazyLock::new(|| vec![
    Regex::new(r"(?i)(?:going to|will|plan to|about to)\s+(?:change|modify|delete|remove|add|replace|refactor)").unwrap(),
    Regex::new(r"(?i)(?:choosing|chose|decision|decided)\s+(?:to|between|against)").unwrap(),
    Regex::new(r"(?i)(?:trade.?off|alternative|instead of|rather than)").unwrap(),
    Regex::new(r"(?i)(?:should we|should i|do we|question is)").unwrap(),
]);
```

## Memory Consultation

`consult_memory(reasoning_text, top_k=3)` is the main entry point:

1. Extract decision points from the reasoning text
2. For each point, query `memory_recall.recall()` for past decisions
3. Query `skill_extractor.query_skills()` for applicable procedures
4. Format results as a context string

Returns a formatted string suitable for injection into an agent's
system prompt or reasoning context.

### Format

```
MEMORY CONSULTATION:

Decision point: "going to change the retrieval engine"
Past context:
  - [2026-03-15] We decided to use BM25 because embeddings were too slow.
  - [2026-03-20] BM25 accuracy measured at 81.2% on LOCOMO.

Applicable skills:
  - [3×] fixed-bm25-scoring: Fixed the BM25 scoring bug by updating IDF.

---
```

## Decision Guard

`decision_guard(action, context="")` provides a safety check before
destructive actions:

```python
warnings = decision_guard(
    action="delete sparse W save code",
    context="cleaning up snn_backend.py"
)
# Returns ["WARNING: Past reasoning recommends keeping sparse W..."]
```

Returns an empty list if no past reasoning contradicts the action.
Returns warning strings if contradictions are found.

## Pipeline Integration

Active retrieval is wired into:

```
MCP server → on recall with reasoning context → consult_memory()
CLI → remanentia guard "action" → decision_guard()
Agent reasoning loop → extract_decision_points() → inject context
```

The module depends on:
- `memory_recall.recall()` — for past decision retrieval
- `skill_extractor.query_skills()` — for procedural knowledge
- `remanentia_active_retrieval` (Rust) — for decision point extraction

## Performance

| Operation | Measured | Budget |
|-----------|---------|--------|
| `extract_decision_points` (Rust) | 1.0 µs | <10 µs |
| `consult_memory` (with 3 recalls) | <200 ms | <500 ms |
| `decision_guard` (single action) | <100 ms | <500 ms |

The Rust acceleration makes decision point extraction negligible —
the bottleneck is `memory_recall.recall()` which runs the full BM25
search pipeline.

## Usage

```python
from active_retrieval import (
    extract_decision_points,
    consult_memory,
    decision_guard,
)

# Extract decision points
text = "We are going to change the retrieval engine. Should we use BM25?"
points = extract_decision_points(text)
# ["We are going to change the retrieval engine",
#  "Should we use BM25"]

# Consult memory during reasoning
context = consult_memory(
    "I'm about to replace the BM25 engine with embeddings-only search."
)
print(context)
# MEMORY CONSULTATION:
# Decision point: "about to replace the BM25 engine"
# Past context: ...

# Decision guard before destructive action
warnings = decision_guard(
    action="delete snn_backend.py",
    context="cleaning up unused modules"
)
if warnings:
    for w in warnings:
        print(f"WARNING: {w}")
```

## Test Coverage

Tests in `tests/test_improvements.py` and `tests/test_integration.py`:

- **Decision points**: all 4 pattern categories, no matches, empty text
- **Consult memory**: with/without matching past decisions
- **Decision guard**: warning generation, safe action passes
- **Pipeline**: integration with memory_recall and skill_extractor
- **Performance**: extract_decision_points < 10µs

All 6 STRONG dimensions: empty, error, negative, pipeline, roundtrip, performance.

## API Reference

::: active_retrieval.extract_decision_points

::: active_retrieval.consult_memory

::: active_retrieval.decision_guard
