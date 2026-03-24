# Decision Trace: Remanentia v2 Architecture
# Date: 2026-03-24T17:00Z

## The Core Insight

Every system scoring 90%+ on memory benchmarks treats memory as active computation, not passive storage:
- Observational Memory (Mastra, 94.87%): Observer + Reflector background agents
- A-MEM (NeurIPS 2025): Self-organizing Zettelkasten where new memories update old ones
- EverMemOS (92.3%): "Processor for memory applications, not repository"

Remanentia v1 is a retrieval engine. v2 is a cognitive layer.

## Why Knowledge Notes (A-MEM Zettelkasten)

Flat paragraphs have no structure. Knowledge notes are:
- Atomic (one fact per note)
- Interconnected (bi-directional links)
- Self-organizing (new notes update old ones)
- Contradiction-aware (supersession tracking)

A-MEM showed this approach beats fixed-operation baselines across all query types, especially multi-hop.

## Why Contradiction Detection

No competitor tracks contradictions well. Mem0 has a "Conflict Detector" but it's LLM-powered.
Remanentia can detect contradictions heuristically (same entity + opposite action + different date).
This is a genuine differentiator.

## Why Prospective Triggers

Zero competitors implement prospective memory. Neuroscience identifies it as a distinct memory system.
"Remember to do X when Y happens" is fundamentally different from "recall what happened."
This is novel. This is defensible. This is useful.

## Why Observer/Reflector

Background processing > query-time processing. Mastra proved this at 94.87%.
The observer watches files. The reflector reasons periodically. Queries are fast because work is done in advance.
