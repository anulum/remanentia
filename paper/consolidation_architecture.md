# Arcane Sapience — Memory Consolidation Architecture

**Date:** 2026-03-19
**Status:** Design → Implementation

## The Insight

The SNN failed as a retriever because retrieval is a MATCHING problem
(query→trace) and embeddings already solve matching. But the SNN runs
24/7 and processes everything — that makes it a natural CONSOLIDATOR.

Biological parallel: the hippocampus doesn't store long-term memories.
It manages the PROCESS of consolidation — replaying experiences during
sleep, detecting novelty, compressing episodes into semantic knowledge,
building associative links. The SNN daemon already runs continuously,
ingests all traces, and replays them. It just needs a consolidation
pipeline instead of a retrieval pipeline.

## Memory Type Model (adapted from CoALA)

| Type | What | Storage | Remanentia Location |
|------|------|---------|-------------------|
| **Semantic** | Extracted facts, decisions, knowledge | Structured markdown | `memory/semantic/` |
| **Episodic** | Raw experiences, session history | Traces, logs | `reasoning_traces/`, `session_states/` |
| **Procedural** | Skills, workflows, patterns | Extracted rules | `skills/`, `memory/procedural/` |
| **Graph** | Entity relationships, concept links | JSON adjacency | `memory/graph/` |
| **Working** | Current session context | In-context | Claude's session window |

## Six Memory Operations

| Operation | What Happens | Who Does It |
|-----------|-------------|-------------|
| **Generate** | New trace/stimulus arrives | Agent sessions, git hooks |
| **Store** | Write to episodic (traces) or semantic (facts) | Consolidation engine |
| **Retrieve** | Query → ranked results | TF-IDF + best-paragraph (92.9%) |
| **Consolidate** | Episodic → semantic compression | SNN daemon + LLM extraction |
| **Update** | Modify existing facts when new info arrives | Conflict resolver |
| **Forget** | Remove obsolete/contradicted memories | Decay + explicit deletion |

## Directory Structure

```
04_ARCANE_SAPIENCE/
├── memory/
│   ├── semantic/                  # Consolidated knowledge
│   │   ├── projects/              # Per-project summaries
│   │   │   ├── director-ai.md
│   │   │   ├── sc-neurocore.md
│   │   │   └── scpn-quantum-control.md
│   │   ├── decisions/             # Key decisions with rationale
│   │   │   ├── 2026-03-17_scpn-fusion-port.md
│   │   │   └── 2026-03-18_dynamic-retrieval-approach.md
│   │   ├── concepts/              # Domain knowledge
│   │   │   ├── stdp-learning.md
│   │   │   ├── kuramoto-synchronization.md
│   │   │   └── holographic-memory.md (negative result)
│   │   └── findings/              # Experimental results
│   │       ├── snn-retrieval-negative.md
│   │       └── best-paragraph-matching.md
│   ├── episodic/                  # Raw experiences (symlinks)
│   │   ├── traces/ → ../reasoning_traces/
│   │   └── sessions/ → ../session_states/
│   ├── procedural/                # How-to knowledge
│   │   ├── skills/ → ../skills/
│   │   └── workflows/
│   │       ├── ci-push-preflight.md
│   │       └── gpu-daemon-restart.md
│   └── graph/                     # Relationship structure
│       ├── entities.jsonl         # Entity registry
│       ├── relations.jsonl        # Entity-entity links
│       └── trace_clusters.json    # Episode groupings
├── consolidation/                 # Pipeline state
│   ├── pending.json               # Traces awaiting consolidation
│   ├── last_consolidation.json    # Timestamp + stats
│   └── conflicts.json             # Detected contradictions
├── consolidation_engine.py        # The pipeline
├── gpu_daemon.py                  # Orchestrates cycles
├── retrieve.py                    # Retrieval (unchanged, 92.9%)
└── monitor.py                     # Dashboard (adds consolidation panel)
```

## Consolidation Pipeline

### Phase 1: Detect New Material
- Daemon watches `reasoning_traces/` and `snn_stimuli/`
- New files → add to `consolidation/pending.json`
- Trigger: every 10 daemon cycles OR when pending count > 5

### Phase 2: Cluster Episodes
- Load all pending traces + recent consolidated traces
- Compute pairwise similarity via best-paragraph embedding
- Group into episodes (traces about the same work effort)
- Write clusters to `memory/graph/trace_clusters.json`

### Phase 3: Extract Structured Facts
- For each episode cluster, prompt an LLM (or use heuristic extraction):
  - Key decisions made
  - Technical findings
  - Unresolved questions
  - Entities mentioned (projects, concepts, people)
- Write to `memory/semantic/{type}/{date}_{topic}.md`
- Format: YAML frontmatter + markdown body

### Phase 4: Build/Update Entity Graph
- Extract entities from each fact (project names, concept names, trace refs)
- Add to `memory/graph/entities.jsonl`
- Create relations between co-occurring entities
- Relations weighted by co-occurrence frequency

### Phase 5: Conflict Detection
- Compare new facts against existing semantic memories
- Flag contradictions (e.g., "repo X is at v3.9" vs "repo X is at v4.0")
- Write to `consolidation/conflicts.json` for human review
- Auto-resolve if newer fact has clear evidence

### Phase 6: Update Index
- Add new semantic memories to retrieval corpus
- Update retrieval caches
- Log consolidation stats

## SNN Daemon's Role

The daemon doesn't do retrieval. It does three things:

### 1. Novelty Detection
When a new trace is injected, compare its spike pattern against the
running average. Traces that produce UNUSUAL spike patterns (high
deviation from mean) contain novel information → priority consolidation.

Metric: `novelty = 1 - cosine(spike_pattern, running_mean_pattern)`

This uses the spike features for what they CAN do — detect statistical
outliers — rather than what they can't do — discriminate between memories.

### 2. Consolidation Scheduling
- Track pending traces count and novelty scores
- High novelty → immediate consolidation
- Low novelty → batch consolidation at next cycle
- The neuron firing rate is a proxy for "information load"

### 3. Association Discovery
- During replay cycles, inject pairs of traces from same project/timeframe
- Track which neuron groups co-activate across trace pairs
- Use co-activation to suggest entity relationships for the graph
- This is what STDP actually encodes: temporal co-occurrence patterns

## Semantic Memory Format

```markdown
---
type: decision
date: 2026-03-18
project: remanentia
source_traces:
  - 2026-03-18_dynamic_retrieval_breakthrough.md
  - 2026-03-18_retrieval_failure_analysis.md
entities: [STDP, spike-matching, retrieval, TF-IDF]
confidence: 0.9
last_validated: 2026-03-19
supersedes: null
---

# Dynamic Spike Matching for Retrieval

Static weight projection (cos(Wx,Wy)) fails at 0% P@1 because dense
positive W makes all projections converge. Dynamic spike pattern matching
achieves 79% P@1 by injecting stimuli and comparing spike count vectors.

However, the shuffled-W control experiment (2026-03-19) showed that
STDP-learned W performs identically to random W. The discrimination
comes from the encoding, not the learned connectivity.

Production retrieval uses TF-IDF + best-paragraph embedding at 92.9% P@1.
```

## Entity Graph Format

```jsonl
{"id": "stdp", "type": "concept", "label": "Spike-Timing Dependent Plasticity", "first_seen": "2026-03-17", "trace_count": 8}
{"id": "director-ai", "type": "project", "label": "Director-AI", "version": "v3.9.0", "first_seen": "2026-03-15"}
{"id": "remanentia", "type": "product", "label": "Remanentia", "first_seen": "2026-03-18"}
```

```jsonl
{"source": "stdp", "target": "remanentia", "type": "used_in", "weight": 5, "evidence": ["2026-03-18_dynamic_retrieval_breakthrough.md"]}
{"source": "director-ai", "target": "remanentia", "type": "funds", "weight": 1, "evidence": ["2026-03-17T1500_revenue-strategy-discussion.md"]}
```

## Retrieval Enhancement (via consolidation)

The 92.9% retrieval stays as-is (TF-IDF + best-paragraph). But now it
also searches semantic memories, not just raw traces:

1. Query → TF-IDF + embedding over (traces + semantic memories)
2. If match is a semantic memory → it has source_traces → can expand
3. Entity graph → suggest related concepts and traces
4. Result: structured answer with provenance, not just raw text

## What Makes This Different

Every other memory system (Mem0, LangMem, Letta) runs consolidation
as a post-hoc LLM pipeline. Remanentia runs consolidation through a
persistent spiking neural network that:

- Detects NOVELTY in new information (spike deviation)
- Discovers ASSOCIATIONS through temporal co-activation (STDP)
- Runs CONTINUOUSLY, not just after sessions
- Is BIOLOGICALLY inspired (hippocampal replay model)

The SNN doesn't store memories. It manages the PROCESS of memory formation.
That's the paper's contribution — not "SNN retrieval" but "SNN-orchestrated
memory consolidation with novelty detection and temporal association."

## Implementation Priority

1. Create directory structure
2. Implement consolidation_engine.py (clustering + extraction)
3. Wire into gpu_daemon.py (novelty detection + scheduling)
4. Add semantic memory to retrieve.py search corpus
5. Build entity graph from existing traces
6. Dashboard panel for consolidation status
