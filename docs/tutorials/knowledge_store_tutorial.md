# Tutorial: Knowledge Store — Atomic Notes and Graph Search

The Knowledge Store is a Zettelkasten-style atomic note system. Each note is auto-linked to related notes via token similarity and shared entities. Notes can supersede each other when contradictions are detected.

## Creating Notes

```python
from knowledge_store import KnowledgeStore

ks = KnowledgeStore()

# add_note() auto-extracts entities, keywords, and prospective queries
note = ks.add_note(
    "BM25 retrieval accuracy improved from 71% to 85% after switching to real TF.",
    source="session_log.md",
)

print(f"ID: {note.id}")
print(f"Title: {note.title}")
print(f"Entities: {note.entities}")
print(f"Keywords: {note.keywords}")
print(f"Prospective queries: {note.prospective_queries[:3]}")
```

### What add_note() Does Internally

1. **Entity extraction** — finds known terms (bm25, stdp, pytorch, etc.) and capitalised names
2. **Keyword extraction** — frequency-based, keeps terms appearing 2+ times or capitalised
3. **Prospective query generation** — the "Kumiho technique": generates hypothetical questions at write time to bridge the cue-trigger semantic gap. A note about "Caroline likes pottery" generates "what hobbies does Caroline have" — matching future queries that use different words
4. **Duplicate detection** — if token similarity > 0.9 with an existing note, merges instead of creating a new one
5. **Contradiction detection** — if the same entity appears with opposite action verbs, marks the new note as superseding the old one
6. **Similarity linking** — links to the 3 most similar existing notes (Jaccard on tokens)
7. **Entity linking** — if 2+ entities overlap with another note, creates a typed link

## Searching Notes

```python
# Keyword search (uses token overlap + prospective queries)
results = ks.search("what retrieval method do we use", top_k=5)
for note, score in results:
    print(f"  {note.id} (score={score:.1f}): {note.content[:80]}")
```

The search matches against both the note content AND the prospective queries generated at write time. This means a search for "hobbies" can find a note that says "likes pottery" — because "what hobbies" was generated as a prospective query.

## Graph Search (Multi-Hop)

```python
# Follow links up to 2 hops deep
related = ks.graph_search("BM25 scoring", top_k=5, hop_depth=2)
for note in related:
    print(f"  {note.id}: {note.content[:80]}")
    for link in note.links:
        print(f"    -> {link['target']} ({link['type']})")
```

`graph_search()` starts from the best keyword matches, then traverses `related`, `supersedes`, and `superseded_by` links to find connected notes. `hop_depth=2` means "follow links from the initial matches, then follow links from those results."

## Prospective Triggers

Triggers fire automatically when a future query matches a condition:

```python
# Set a trigger
trigger = ks.add_trigger(
    condition="authentication",
    action="We switched from JWT to session tokens on 2026-03-15",
)

# Later, when someone asks about authentication:
matched = ks.check_triggers("what authentication method do we use")
for t in matched:
    print(f"  [TRIGGER] {t.action}")
```

Triggers are checked by the MCP server on every `remanentia_recall` call. They surface proactively — the user doesn't need to know the trigger exists.

## Contradiction Detection

When a new note contradicts an existing one, both are preserved with a supersession chain:

```python
ks.add_note("BM25 accuracy is 85% on LOCOMO.", source="v1.md")
n2 = ks.add_note("BM25 accuracy dropped to 65% after the index rebuild.", source="v2.md")

if n2.supersedes:
    print(f"Note {n2.id} supersedes {n2.supersedes}")
    # The old note gets a superseded_by link
    # The new note gets a supersedes link
    # Both remain searchable — no silent overwriting
```

## Persistence

```python
# Save to disk
ks.save()  # uses default paths: knowledge_notes.jsonl + knowledge_triggers.jsonl

# Load from disk
ks2 = KnowledgeStore()
ks2.load()
```

## KnowledgeNote Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | str | MD5 hash of content + source |
| `title` | str | Auto-extracted from first line or keywords |
| `content` | str | Full note text |
| `keywords` | list[str] | Extracted keywords |
| `entities` | list[str] | Extracted entity names |
| `source` | str | Origin file/session |
| `created` | str | ISO timestamp |
| `updated` | str | ISO timestamp |
| `links` | list[dict] | Typed links to other notes |
| `supersedes` | str | ID of the note this one replaces (if any) |
| `superseded_by` | str | ID of the note that replaced this one |
| `prospective_queries` | list[str] | Auto-generated hypothetical queries |
| `searchable_text` | str | Content + prospective queries (used for search) |

## Next Steps

- [Temporal Queries Tutorial](temporal_tutorial.md) — date reasoning
- [API Reference: knowledge_store](../api/knowledge_store.md) — full API docs
