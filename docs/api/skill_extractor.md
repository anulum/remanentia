# skill_extractor

Extracts recurring procedural skills from reasoning traces via
trigger-action pattern matching, token-overlap clustering, and
Rust-accelerated text processing.

## Purpose

The skill extractor scans reasoning traces for recurring patterns —
decisions, bug fixes, refactoring strategies, configuration choices —
and groups them into named skills. Skills represent procedural memory:
"when situation X occurs, do Y."

This is Remanentia's procedural memory layer, complementing:
- **Semantic memory** (consolidation_engine) — facts and findings
- **Episodic memory** (reasoning_traces) — raw session logs
- **Graph memory** (knowledge_store) — entity relationships

## Architecture

```
reasoning_traces/*.md
        │
        ▼
  extract_skills(traces_dir)
        │
        ├── For each trace line:
        │     ├── _tokenize_lower(line)      ← Rust: remanentia_skill_extractor.tokenize_lower
        │     └── matches_skill_marker(line)  ← Rust: remanentia_skill_extractor.matches_skill_marker
        │
        ├── Cluster entries by token overlap (>30% Jaccard)
        │
        └── Generate skill descriptors:
              name, description, evidence, frequency, key_terms, last_seen
        │
        ▼
  snn_state/skills.json (persisted)
```

## Skill Markers

5 trigger-action pairs detect skill-like patterns in trace text:

| Trigger | Action | Example |
|---------|--------|---------|
| `fix(ed\|ing)?` | `update\|change\|modify\|add\|remove\|replace` | "Fixed the scoring bug by updating IDF" |
| `bug\|error\|fail(ure\|ed)?\|broke` | `fix\|resolve\|patch\|workaround` | "Error resolved with patch to BM25" |
| `chose\|decision\|trade-off` | `because\|reason\|rationale` | "Chose AGPL because of openness" |
| `pattern\|approach\|strategy` | `works?\|better\|cleaner\|faster` | "This approach works better for recall" |
| `refactor` | `extract\|split\|merge\|rename\|move` | "Refactored to extract the tokeniser" |

A line must match BOTH a trigger AND an action pattern to be considered a skill.

## Rust Acceleration (remanentia_skill_extractor)

3 functions are Rust-accelerated:

| Function | Python module | Rust function | Measured |
|----------|-------------|---------------|---------|
| `_tokenize_lower` | `skill_extractor._tokenize_lower` | `tokenize_lower` | 1.9 µs |
| skill marker check | `skill_extractor.extract_skills` | `matches_skill_marker` | 2.9 µs |
| skill ranking | (available) | `rank_skills_by_overlap` | — |

### Rust implementation

```rust
static SKILL_MARKERS: LazyLock<Vec<(Regex, Regex)>> = LazyLock::new(|| vec![
    (
        Regex::new(r"(?i)fix(?:ed|ing)?").unwrap(),
        Regex::new(r"(?i)(?:update|change|modify|add|remove|replace)").unwrap(),
    ),
    // ... 4 more pairs
]);

fn matches_skill_marker(text: &str) -> bool {
    let lower = text.to_lowercase();
    SKILL_MARKERS.iter().any(|(trigger, action)| {
        trigger.is_match(&lower) && action.is_match(&lower)
    })
}
```

## Clustering

Extracted entries are clustered using greedy single-linkage with
Jaccard token overlap threshold of 30%:

1. For each unassigned entry, find all entries with >30% token overlap
2. Group them into a cluster
3. Name the cluster using the 4 most common non-stop tokens
4. Use the longest entry as the representative description

## Skill Descriptor

```python
{
    "name": "fixed-bm25-scoring-bug",
    "description": "Fixed the BM25 scoring bug by updating the IDF calculation.",
    "evidence": ["2026-03-15_decision.md", "2026-03-20_fix.md"],
    "frequency": 3,
    "key_terms": ["fixed", "bm25", "scoring", "bug"],
    "last_seen": "2026-03-20_fix.md"
}
```

## Query Skills

`query_skills(query, top_k=5)` finds skills relevant to a query by
token overlap between query words and skill key_terms. Returns skills
sorted by overlap score (descending).

The Rust function `rank_skills_by_overlap` provides accelerated ranking
when the crate is installed.

## Pipeline Integration

```
CLI: remanentia skills --list       → load_skills() → print
CLI: remanentia skills --query "X"  → query_skills("X") → ranked results
Observer heartbeat                  → extract_skills() → save_skills()
Active retrieval                    → query_skills() → inject into context
```

The skill extractor is called by:
- **`cli.py`** — `skills` subcommand for listing and querying
- **`active_retrieval.consult_memory()`** — queries skills for decision context
- **Manual**: `python skill_extractor.py --list` / `--query "CI failure"`

## Performance

| Operation | Measured | Budget |
|-----------|---------|--------|
| `_tokenize_lower` (Rust) | 1.9 µs | <10 µs |
| `matches_skill_marker` (Rust) | 2.9 µs | <10 µs |
| `extract_skills` (10 traces) | <100 ms | <500 ms |
| `query_skills` (20 skills) | <5 ms | <20 ms |

## Persistence

Skills are stored in `snn_state/skills.json` as a JSON array.
`save_skills()` writes, `load_skills()` reads. File is re-extracted
on every extraction run (not incremental — traces are small).

## Usage

```python
from skill_extractor import extract_skills, save_skills, query_skills

# Extract skills from traces
skills = extract_skills()
print(f"Found {len(skills)} skills")
for s in skills[:5]:
    print(f"  [{s['frequency']}×] {s['name']}: {s['description'][:60]}")

# Persist
save_skills(skills)

# Query
results = query_skills("CI failure fix", top_k=3)
for s in results:
    print(f"  {s['name']}: {s['description'][:60]}")
```

## Test Coverage

Tests in `tests/test_improvements.py` (shared test file):

- **Extract skills**: with traces, empty dir, no matches
- **Query skills**: token overlap ranking, empty query
- **Save/load**: roundtrip persistence
- **Pipeline**: skills used in active_retrieval.consult_memory()

All 6 STRONG dimensions: empty, error, negative, pipeline, roundtrip, performance.

## API Reference

::: skill_extractor.extract_skills

::: skill_extractor.query_skills

::: skill_extractor.save_skills

::: skill_extractor.load_skills
