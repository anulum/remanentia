# knowledge_store

Zettelkasten atomic notes with typed edges, multi-hop graph search,
prospective queries, contradiction detection, and Rust-accelerated
text processing.

## Purpose

The KnowledgeStore is Remanentia's write-path memory — every observation,
decision, finding, and preference extracted from conversations and files
is stored as a KnowledgeNote with typed links to related notes. It
provides a structured overlay on top of the raw BM25 index in
`memory_index.py`, enabling:

- **Multi-hop graph search** — follow typed edges across notes (2-hop default)
- **Contradiction detection** — flag when new facts conflict with existing ones
- **Prospective triggers** — conditional actions that fire when matching queries arrive
- **Kumiho prospective queries** — hypothetical questions generated at write time
  to bridge the cue-trigger semantic gap

## Architecture

```
Observer / MCP server
        │
        ▼
  KnowledgeStore.add_note(content, source)
        │
        ├── _tokenize(content)              ← Rust: remanentia_knowledge_store.tokenize
        ├── _extract_keywords(content)      ← Rust: remanentia_knowledge_store.extract_keywords
        ├── _extract_entities(content)      ← Rust: remanentia_knowledge_store.extract_entities
        ├── extract_person_names(content)   ← Rust: remanentia_knowledge_store.extract_person_names
        ├── _generate_prospective_queries() ← prospective query generation (Kumiho)
        ├── _detect_contradiction()         ← token overlap + entity intersection
        │
        ▼
  KnowledgeNote (dataclass)
        │
        ├── Stored in notes.jsonl
        ├── Typed links (edge_type: supersedes, relates, contradicts, ...)
        └── Prospective queries appended to note for retrieval at search time
```

## Rust Acceleration (remanentia_knowledge_store)

4 functions are Rust-accelerated via PyO3. Python fallback is preserved
for environments without the Rust crate installed.

| Function | Python module | Rust function | Measured |
|----------|-------------|---------------|---------|
| `_tokenize` | `knowledge_store._tokenize` | `tokenize` | 1.5 µs |
| `_extract_keywords` | `knowledge_store._extract_keywords` | `extract_keywords` | 13.2 µs |
| `_extract_entities` | `knowledge_store._extract_entities` | `extract_entities` | 1.7 µs |
| `extract_person_names` | `knowledge_store.extract_person_names` | `extract_person_names` | 7.1 µs |

### Wiring pattern

Every function follows the same try/except pattern:

```python
def _tokenize(text: str) -> set[str]:
    try:
        from remanentia_knowledge_store import tokenize as _rust_tok
        return _rust_tok(text)  # pragma: no cover
    except ImportError:
        pass
    return set(re.findall(r"[a-z0-9][a-z0-9_]{2,}", text.lower()))
```

### Rust implementation

The Rust crate uses `LazyLock<Regex>` for compiled patterns (initialised once,
zero allocation on subsequent calls). The `extract_keywords` function uses a
`HashMap` for frequency counting, which is faster than Python's `defaultdict`
for the word-counting hot loop.

```rust
static RE_TOKEN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"[a-z0-9][a-z0-9_]{2,}").unwrap()
});
```

## Text Processing Pipeline

### Tokenisation

`_tokenize(text)` extracts lowercase alphanumeric tokens of 3+ characters.
Used for:
- Token overlap scoring in `search()`
- Contradiction detection in `_detect_contradiction()`

### Keyword extraction

`_extract_keywords(text)` identifies important terms via:
1. **Frequency**: tokens appearing 2+ times in the text
2. **Capitalisation**: CamelCase terms (likely names/concepts)
3. **Version numbers**: `v0.1.0`, `v3.14.0`, etc.

Returns at most 20 keywords, sorted alphabetically.

### Entity extraction

`_extract_entities(text)` identifies domain entities via:
1. **Known concepts**: 18 hardcoded domain terms (stdp, bm25, lif, snn, ...)
2. **Version numbers**: `v\d+\.\d+`
3. **Percentages**: `\d+\.?\d*%`
4. **Person names**: capitalised words after sentence boundaries

### Person name extraction

`extract_person_names(text)` identifies people in conversational text:
1. **"Person:" pattern**: `^Alice:` at line starts (chat transcript format)
2. **Post-sentence names**: capitalised words after `.`, `!`, `?`, `\n`
3. **Stop word filtering**: excludes "The", "What", "Thanks", etc.

## Prospective Queries (Kumiho technique)

At write time, `_generate_prospective_queries()` generates hypothetical
future queries that might retrieve this note. This bridges the semantic
gap between how information is stored and how it is later queried.

Example: A note containing "Caroline mentioned she likes pottery" generates:
- "what about caroline"
- "what happened to caroline"
- "what does caroline like"
- "hobbies pottery"
- "interests pottery"
- "pottery"

12 pattern categories:
1. Entity-based questions ("what about X", "what happened to X")
2. Keyword expansion
3. Title as question
4. Activity/preference detection
5. Occupation/role patterns
6. Allergy/dietary patterns
7. Location patterns
8. Temporal context
9. Relationship patterns
10. Skill/expertise mentions
11. Opinion/preference expansion
12. Numeric score context

## Contradiction Detection

When adding a note, `_detect_contradiction()` scans existing notes for:
- High token overlap (>40% Jaccard similarity) AND
- At least one shared entity

If both conditions are met, the new note is flagged as potentially
contradicting the existing one, and a "contradicts" typed link is added.

## Data Structures

### KnowledgeNote

```python
@dataclass
class KnowledgeNote:
    id: str              # SHA-256 digest prefix of content[:200] + source
    content: str         # The atomic fact text
    source: str          # Origin file name
    keywords: list[str]  # Extracted keywords (max 20)
    entities: list[str]  # Extracted entities
    prospective_queries: list[str]  # Kumiho queries
    timestamp: float     # Creation time
    title: str           # Auto-generated title
    typed_links: dict[str, list[str]]  # edge_type → list of note IDs
    superseded_by: str   # ID of note that supersedes this one (or "")
```

### Trigger

```python
@dataclass
class Trigger:
    condition: str   # Query pattern to match
    action: str      # What to do when triggered
    created: float   # Creation timestamp
```

## Pipeline Integration

The KnowledgeStore is called from:
- **`observer.observe_once()`** — adds notes for new/changed files
- **`mcp_server.handle_remember()`** — adds notes via MCP write tool
- **`memory_recall.recall()`** — searches notes via `store.search()`
- **`cli.cmd_notes()`** — lists/searches notes via CLI

```
Filesystem change → observer → KnowledgeStore.add_note()
MCP remember     → mcp_server → KnowledgeStore.add_note()
MCP recall       → memory_recall → KnowledgeStore.search()
CLI notes        → cli → KnowledgeStore.search()
```

## Performance

| Operation | Measured | Budget |
|-----------|---------|--------|
| `add_note` | 0.023 ms | <50 ms |
| `search` (20 notes) | <20 ms | <20 ms |
| `_tokenize` (Rust) | 1.5 µs | <10 µs |
| `_extract_keywords` (Rust) | 13.2 µs | <50 µs |
| `_extract_entities` (Rust) | 1.7 µs | <10 µs |
| `extract_person_names` (Rust) | 7.1 µs | <20 µs |

## Usage

```python
from knowledge_store import KnowledgeStore

store = KnowledgeStore()
store.load()

# Add a note
note = store.add_note(
    "BM25 accuracy measured at 81.2% on LOCOMO benchmark.",
    source="experiment_log.md",
)

# Search
results = store.search("LOCOMO accuracy", top_k=5)
for note in results:
    print(f"[{note.id}] {note.title}")

# Graph search (multi-hop)
related = store.graph_search("BM25 retrieval", top_k=5, hop_depth=2)

# Check triggers
triggers = store.check_triggers("BM25 accuracy dropped")
for t in triggers:
    print(f"TRIGGER: {t.action}")

# Contradiction detection (automatic on add_note)
note2 = store.add_note("BM25 accuracy measured at 74.7%.", source="new_log.md")
# → auto-detects contradiction with first note, adds typed link
```

## Test Coverage

74 tests in `tests/test_knowledge_store.py`:

- **Add note**: basic, with source, duplicate detection
- **Search**: keyword, entity boost, empty query
- **Graph search**: multi-hop, typed edges, cycle detection
- **Contradiction detection**: token overlap, entity intersection
- **Triggers**: creation, matching, non-matching
- **Prospective queries**: all 12 pattern categories
- **Person names**: chat transcript, sentence boundary, stop words
- **Persistence**: save/load roundtrip, corrupt file handling
- **Performance**: add_note <50ms, search <20ms

All 6 STRONG dimensions: empty, error, negative, pipeline, roundtrip, performance.

## API Reference

::: knowledge_store.KnowledgeStore
    options:
      show_source: true
      members_order: source

::: knowledge_store.KnowledgeNote
