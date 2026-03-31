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

Procedural memory is the most durable form: a debugging procedure
learned from 3 incidents becomes permanent knowledge that prevents
the 4th. The skill extractor automates this — no manual curation
needed.

## Architecture

```
reasoning_traces/*.md
        │
        ▼
  extract_skills(traces_dir)
        │
        ├── For each trace file:
        │     ├── Read text, split into lines
        │     ├── Skip headers (#) and empty lines
        │     ├── Strip markdown bullets (- *)
        │     │
        │     └── For each line:
        │           ├── _tokenize_lower(line)
        │           │     ← Rust: remanentia_skill_extractor.tokenize_lower
        │           │
        │           └── matches_skill_marker(line)
        │                 ← Rust: remanentia_skill_extractor.matches_skill_marker
        │                 Must match BOTH trigger AND action pattern
        │
        ├── Cluster entries by Jaccard token overlap (>30%)
        │     ├── Greedy single-linkage clustering
        │     ├── O(N²) pairwise comparison
        │     └── Stop words filtered: the, and, was, for, this, that, with, from, not, but
        │
        └── Generate skill descriptors:
              ├── name: top 4 non-stop tokens joined by "-"
              ├── description: longest entry text (representative)
              ├── evidence: list of source trace filenames
              ├── frequency: cluster size (higher = more reliable)
              ├── key_terms: top 6 non-stop tokens by frequency
              └── last_seen: most recent trace filename
        │
        ▼
  snn_state/skills.json (persisted)
```

## Skill Markers

5 trigger-action pairs detect skill-like patterns in trace text.
A line must match BOTH a trigger AND an action regex to be considered
a skill-like pattern. This two-gate requirement prevents false
positives: "fixed a typo" matches "fix" trigger but not any action
pattern, so it is correctly excluded.

| # | Trigger regex | Action regex | What it catches |
|---|--------------|-------------|----------------|
| 1 | `fix(ed\|ing)?` | `update\|change\|modify\|add\|remove\|replace` | Bug fixes with code changes |
| 2 | `bug\|error\|fail(ure\|ed)?\|broke` | `fix\|resolve\|patch\|workaround` | Error resolution procedures |
| 3 | `chose\|decision\|trade-off` | `because\|reason\|rationale` | Decisions with rationale |
| 4 | `pattern\|approach\|strategy` | `works?\|better\|cleaner\|faster` | Validated approaches |
| 5 | `refactor` | `extract\|split\|merge\|rename\|move` | Refactoring recipes |

### Pattern design rationale

The triggers identify SITUATIONS (something went wrong, a choice was
made, a pattern was tried). The actions identify RESPONSES (what was
done about it). This trigger→action pairing maps directly to the
if-then structure of procedural memory.

Patterns are case-insensitive. Each pair is compiled as a separate
regex in both Python (`re.search`) and Rust (`LazyLock<Regex>`).

### False positive mitigation

Lines are pre-filtered:
- Empty lines skipped
- Lines starting with `#` skipped (markdown headers)
- Leading `- ` and `* ` stripped (markdown bullets)
- Text capped at 200 chars per entry

## Rust Acceleration (remanentia_skill_extractor)

3 functions are Rust-accelerated via PyO3:

| Function | Python fallback | Rust function | Measured (µs) |
|----------|----------------|---------------|--------------|
| `_tokenize_lower` | `re.findall(r"[a-z0-9_]+", text.lower())` | `tokenize_lower` | 1.3 |
| skill marker check | `any(re.search(t,l) and re.search(a,l) for t,a in markers)` | `matches_skill_marker` | 1.6 |
| skill ranking | Python loop with set intersection | `rank_skills_by_overlap` | — |

### Rust implementation details

```rust
static RE_TOKEN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"[a-z0-9_]+").unwrap()
});

static SKILL_MARKERS: LazyLock<Vec<(Regex, Regex)>> = LazyLock::new(|| vec![
    (
        Regex::new(r"(?i)fix(?:ed|ing)?").unwrap(),
        Regex::new(r"(?i)(?:update|change|modify|add|remove|replace)").unwrap(),
    ),
    (
        Regex::new(r"(?i)bug|error|fail(?:ure|ed)?|broke").unwrap(),
        Regex::new(r"(?i)(?:fix|resolve|patch|workaround)").unwrap(),
    ),
    (
        Regex::new(r"(?i)chose|decision|trade.?off").unwrap(),
        Regex::new(r"(?i)(?:because|reason|rationale)").unwrap(),
    ),
    (
        Regex::new(r"(?i)pattern|approach|strategy").unwrap(),
        Regex::new(r"(?i)(?:works?|better|cleaner|faster)").unwrap(),
    ),
    (
        Regex::new(r"(?i)refactor").unwrap(),
        Regex::new(r"(?i)(?:extract|split|merge|rename|move)").unwrap(),
    ),
]);
```

All regexes use `LazyLock` — compiled once on first call, zero
allocation on subsequent calls. The `matches_skill_marker` function
short-circuits on first matching pair.

### Wiring pattern

```python
def _tokenize_lower(text: str) -> list[str]:
    try:
        from remanentia_skill_extractor import tokenize_lower as _rust_tok
        return _rust_tok(text)  # pragma: no cover
    except ImportError:
        pass
    return re.findall(r"[a-z0-9_]+", text.lower())
```

The `matches_skill_marker` is wired inline within `extract_skills()`:

```python
try:
    from remanentia_skill_extractor import matches_skill_marker as _rust_match
    is_skill = _rust_match(stripped)  # pragma: no cover
except ImportError:
    lower = stripped.lower()
    is_skill = any(
        re.search(trigger_re, lower) and re.search(action_re, lower)
        for trigger_re, action_re in _SKILL_MARKERS
    )
```

## Clustering Algorithm

Extracted entries are clustered using greedy single-linkage with
Jaccard token overlap:

### Algorithm

```
Input: entries = [{text, source, tokens}, ...]
Output: clusters = [[entry, ...], ...]

used = ∅
for i = 0 to len(entries):
    if i ∈ used: continue
    cluster = [entries[i]]
    used ← used ∪ {i}
    for j = i+1 to len(entries):
        if j ∈ used: continue
        overlap = entries[i].tokens ∩ entries[j].tokens
        union   = entries[i].tokens ∪ entries[j].tokens
        if |overlap| / max(|union|, 1) > 0.3:
            cluster.append(entries[j])
            used ← used ∪ {j}
    clusters.append(cluster)
```

### Threshold choice: 30%

The 30% Jaccard threshold was chosen empirically:
- <20%: Too many false clusters (unrelated entries merged)
- 30%: Balances recall (finds related patterns) and precision
  (avoids mixing unrelated patterns)
- >50%: Too strict — near-duplicates only, misses paraphrased patterns

### Naming

The cluster name is constructed from the top 4 most frequent
non-stop tokens, joined by hyphens:

```python
stop = {"the", "and", "was", "for", "this", "that", "with", "from", "not", "but"}
key_tokens = [t for t, c in all_tokens.most_common(6) if t not in stop and len(t) > 2]
name = "-".join(key_tokens[:4])
```

Example: tokens `{fixed, bm25, scoring, bug, the, by}` →
name `"fixed-bm25-scoring-bug"`.

### Representative description

The longest entry in the cluster is chosen as the representative
description. This heuristic assumes longer entries contain more
context about the procedure.

## Skill Descriptor

```python
{
    "name": "fixed-bm25-scoring-bug",
    "description": "Fixed the BM25 scoring bug by updating the IDF calculation to use log(1+N/(1+df)).",
    "evidence": ["2026-03-15_decision.md", "2026-03-20_fix.md"],
    "frequency": 3,
    "key_terms": ["fixed", "bm25", "scoring", "bug"],
    "last_seen": "2026-03-20_fix.md"
}
```

| Field | Type | Semantics |
|-------|------|-----------|
| name | str | Hyphenated top tokens (human-readable identifier) |
| description | str | Longest entry text (representative procedure) |
| evidence | list[str] | Source trace filenames (provenance) |
| frequency | int | Cluster size (higher = more validated) |
| key_terms | list[str] | Top 6 non-stop tokens (for query matching) |
| last_seen | str | Most recent trace filename (recency signal) |

## Query Skills

`query_skills(query, top_k=5)` finds skills relevant to a query:

1. Load skills from `snn_state/skills.json`
2. Tokenize query via `_tokenize_lower(query)` (Rust-accelerated)
3. For each skill: compute `|q_tokens ∩ key_terms| / |q_tokens|`
4. Filter: only skills with overlap > 0
5. Sort by match_score descending
6. Return top_k with added `match_score` field

The Rust function `rank_skills_by_overlap` provides the same
computation with HashSet intersection for larger skill sets.

## Pipeline Integration

```
CLI: remanentia skills --list       → load_skills() → print
CLI: remanentia skills --query "X"  → query_skills("X") → ranked results
CLI: remanentia skills --extract    → extract_skills() → save_skills()
Observer heartbeat                  → (planned: periodic re-extraction)
Active retrieval                    → query_skills() → inject into context
```

### Call sites

| Caller | Function | Purpose |
|--------|----------|---------|
| `cli.py` (skills subcommand) | `extract_skills`, `load_skills`, `query_skills` | User-facing skill management |
| `active_retrieval.consult_memory()` | `query_skills(point, top_k=2)` | Decision-time skill lookup |
| `skill_extractor.main()` | `extract_skills`, `save_skills` | Standalone CLI execution |

### Data flow through pipeline

```
reasoning_traces/*.md
    │
    ▼ (extract_skills)
snn_state/skills.json
    │
    ▼ (query_skills)
active_retrieval.consult_memory()
    │
    ▼
Agent reasoning context
    │
    ▼
Better decisions (fewer repeated mistakes)
```

## Performance

Measured on Intel Core i5-11600K @ 3.90GHz (verified from `/proc/cpuinfo`),
2026-03-31, `time.perf_counter()`.

| Operation | Measured | Budget |
|-----------|---------|--------|
| `_tokenize_lower` (Rust) | 1.3 µs | <10 µs |
| `matches_skill_marker` (Rust) | 1.6 µs | <10 µs |
| `extract_skills` (10 traces, ~50 lines each) | <100 ms | <500 ms |
| `query_skills` (20 skills) | <5 ms | <20 ms |
| `save_skills` / `load_skills` (JSON) | <1 ms | <10 ms |

Bottleneck: file I/O reading traces, not regex matching.

## Persistence

Skills are stored in `snn_state/skills.json` as a JSON array.

- `save_skills(skills)` — writes JSON with `indent=2`
- `load_skills()` — reads JSON, returns `[]` on missing/corrupt file
- File is regenerated on every `extract_skills()` call (not incremental)
- Traces are typically small (<100 files) so full re-extraction is fast

## CLI Usage

```bash
# Extract skills from reasoning traces and save
python skill_extractor.py --extract

# List all extracted skills
python skill_extractor.py --list

# Query skills matching a pattern
python skill_extractor.py --query "CI failure"

# Via remanentia CLI
remanentia skills --list
remanentia skills --query "BM25 fix"
```

## Python API

```python
from skill_extractor import extract_skills, save_skills, load_skills, query_skills

# Extract from traces
skills = extract_skills()
print(f"Found {len(skills)} skills")
for s in skills[:5]:
    print(f"  [{s['frequency']}×] {s['name']}: {s['description'][:60]}")

# Persist
save_skills(skills)

# Load (from saved)
skills = load_skills()

# Query
results = query_skills("CI failure fix", top_k=3)
for s in results:
    print(f"  [{s['match_score']:.2f}] {s['name']}: {s['description'][:60]}")

# Custom traces directory
from pathlib import Path
skills = extract_skills(traces_dir=Path("/custom/traces"))
```

## Limitations

1. **Regex-only detection**: Cannot detect paraphrased skills that don't
   match the 5 trigger-action patterns (e.g., "solved by upgrading"
   doesn't match any trigger).

2. **O(N²) clustering**: For large trace sets (>1,000 entries), the
   pairwise comparison becomes slow. Mitigation: capped at first pass,
   re-extraction is infrequent.

3. **No semantic similarity**: Clustering uses token overlap (Jaccard),
   not embeddings. Two semantically similar but lexically different
   procedures won't cluster.

4. **Single-linkage leakage**: Greedy single-linkage can chain unrelated
   entries if intermediate entries bridge two unrelated clusters.

5. **No skill refinement**: Skills are extracted but never refined or
   merged over time. A skill extracted from session 1 is identical to
   the same skill extracted from session 10.

## Test Coverage

Tests in `tests/test_improvements.py` (shared test file):

- **Extract skills**: with traces, empty dir, no matching patterns
- **Query skills**: token overlap ranking, empty query, no skills
- **Save/load**: roundtrip persistence, corrupt file handling
- **Pipeline**: skills used in `active_retrieval.consult_memory()`
- **Tokenize**: Rust vs Python equivalence
- **Marker matching**: all 5 trigger-action pairs, false negatives

All 6 STRONG dimensions: empty, error, negative, pipeline, roundtrip,
performance.

## API Reference

::: skill_extractor.extract_skills

::: skill_extractor.query_skills

::: skill_extractor.save_skills

::: skill_extractor.load_skills
