# User Manual

## Memory Types

Remanentia maintains five types of memory:

### Episodic Memory

Raw session decisions stored as markdown files in `reasoning_traces/`.
Each trace captures what happened in a work session — decisions made,
problems encountered, solutions found.

```markdown
# Decision: Switch from binary TF to real term frequency
Date: 2026-03-26
Project: remanentia

Binary TF treated all term occurrences equally. Paragraphs mentioning
"STDP" five times scored the same as those mentioning it once.
Switched to real TF: IDF * tf * (k1+1) / (tf + k1*(1-b+b*dl/avg_dl)).
```

### Semantic Memory

Consolidated facts extracted from episodic traces. YAML frontmatter tags
each fact with project, type, date, and source traces.

### Graph Memory

Entity-entity relations stored as JSONL. 11 typed relation types:
caused_by, fixed_by, replaced, contradicts, version_of, depends_on,
improved, produced, used_in, tested_with, co_occurs.

### Knowledge Memory

Zettelkasten atomic notes with bi-directional typed edges (related,
supersedes, superseded_by, depends_on, derived_from, contains).
Multi-hop graph search during recall.

### Procedural Memory

Extracted skills and workflows stored as JSON. Captures repeated
patterns of work for future reference.

## Search

### Basic Search

```bash
remanentia search "STDP learning rule"
```

### Filtered Search

```python
from memory_index import MemoryIndex

idx = MemoryIndex()
idx.load()  # or idx.build() on first use

results = idx.search(
    "authentication decision",
    top_k=10,
    project="director-ai",      # filter by project
    after="2026-03-01",          # only recent
    doc_type="code",             # code files only
)
```

### Query Intelligence

The search engine classifies queries automatically:

| Type | Example | Behaviour |
|------|---------|-----------|
| Location | "where is MemoryIndex defined" | 2x code boost, lookup term matching |
| Temporal | "latest LOCOMO score" | Recency boost, date expression detection |
| Decision | "why did we switch to RRF" | Decision paragraph boost |
| General | "STDP learning rule" | Entity graph boost |

### Multi-Hop Queries

Complex queries are automatically decomposed:

```
"What hobbies does the person who works at Google have?"
→ sub-queries: ["who works at Google?", "what hobbies?"]
→ results combined and re-ranked
```

## Consolidation

### Automatic

Every `remanentia_remember` call triggers debounced consolidation (10s).
New traces are clustered by project and date, structured facts extracted,
entity graph updated, conflicts detected.

### Manual

```bash
remanentia consolidate --force
```

### Conflict Detection

When new facts contradict existing semantic memories, both versions are
preserved with evidence trails in `consolidation/conflicts.json`.
No silent overwriting.

## Entity Graph

### Extraction

GLiNER2 (205M params, zero-shot NER) extracts 13 entity types:
person, project, software tool, hardware, algorithm, file path,
metric value, version number, neural network model, benchmark,
mathematical concept, programming language, scientific concept.

Falls back to regex patterns when GLiNER is unavailable.

### Relations

11 typed relation patterns extracted from connecting text between
entities. Typed relations get 1.5x boost weight in retrieval vs
generic co-occurrence edges.

### Graph Query

```bash
remanentia graph "STDP" --hops 2
```

## SNN Daemon

```bash
python snn_daemon.py --neurons 1000 --interval 60
```

The daemon maintains a persistent LIF spiking network. New reasoning
traces are encoded as current injection patterns. STDP modifies
synaptic weights based on temporal correlations in the input.

The SNN's role is consolidation orchestration and novelty detection —
it detects when information is surprising relative to stored patterns
and triggers consolidation cycles.
