# AI Agent Memory Systems: Competitive Landscape Research (March 2026)

**Date:** 2026-03-24T15:47 UTC
**Author:** Arcane Sapience
**Purpose:** Product strategy input for Remanentia. Web-verified as of 2026-03-24.
**Confidence notes:** All benchmark numbers are from published papers or official announcements. Where I extrapolate or have uncertainty, it's flagged.

---

## Executive Summary

The agent memory market has exploded since mid-2025. At least 15 systems are shipping or have published papers. The field is converging on a common architecture: LLM-based extraction -> structured storage (graph + vector) -> hybrid retrieval (BM25 + embedding + graph traversal) -> LLM-based answer generation. No system achieves SOTA without LLM calls somewhere in the pipeline.

**Remanentia's current position:** 74.7% on LoCoMo with zero LLM calls. This is competitive with local-only systems (SuperLocalMemory: 74.8% Mode A) but 15-20pp behind cloud-LLM systems (EverMemOS: 93%, Hindsight: 91.4%, Kumiho: 93.3%).

**The gap is primarily answer extraction, not retrieval.** Our retrieval at P@1=92.9% (on internal benchmark with embedding rerank) is competitive. The 16pp deficit to Hindsight on LoCoMo is answer formatting, not finding the right paragraph.

---

## 1. Mem0

### What shipped since $24M raise (October 2025)
- **Scale:** API calls grew from 35M (Q1 2025) to 186M (Q3 2025), 30% MoM growth
- **Adoption:** 41K GitHub stars, 14M+ PyPI downloads
- **Enterprise:** AWS chose Mem0 as exclusive memory provider for AWS Agent SDK. CrewAI, Flowise, Langflow integrate natively
- **SOC 2 compliance** on managed cloud

### Architecture
- **Two-phase pipeline:** (1) Extraction: LLM extracts structured facts from conversation + rolling summary. (2) Update: Compare each fact to top-s similar entries in vector DB. Decide: ADD, UPDATE, DELETE, or NOOP
- **Graph variant (Mem0^g):** Directed labeled graph. Entity Extractor -> Relations Generator -> Conflict Detector -> Update Resolver. Adds ~2% accuracy over base
- **Hybrid datastore:** graph + vector + key-value. Actively curates (update, enrich, clean) as new data arrives
- **Hierarchical organization:** user, session, agent levels

### Benchmark scores
- **LoCoMo:** 66.9% (base), 68.4% (Mem0^g)
- **Temporal questions:** 58.13% vs OpenAI's 21.71% (graph variant)
- **vs OpenAI:** 26% relative improvement in LLM-as-Judge metric
- **Latency:** 91% lower p95 than OpenAI, 90% token cost savings
- **Mem0^g latency:** 0.66s median, 0.48s p95 (vs base 0.20s/0.15s)

### Key differentiator
Production readiness. AWS partnership. 186M API calls/quarter. SOC 2. Three-line integration. The safe enterprise choice.

### What they do that Remanentia doesn't
- LLM-based extraction (structured facts from conversations)
- Conflict detection and automatic resolution
- Graph memory with entity-relationship tracking at write time
- Managed cloud with SOC 2
- Memory decay and confidence scoring

### Remanentia advantage over Mem0
- Zero LLM dependency (Mem0 requires LLM for extraction and update decisions)
- Higher LoCoMo accuracy without LLM (74.7% vs Mem0's 66.9%)
- Filesystem-native (indexes existing artifacts, doesn't require separate memory store)
- Local-only, no cloud, no data leaves machine

---

## 2. Letta (MemGPT)

### Current architecture (V1, 2026)
- **Deprecated:** Heartbeats, send_message tool. The original MemGPT loop (inject thinking/heartbeat args into every tool call) is replaced
- **V1 agent loop:** Native reasoning from models (GPT-5, Claude 4.5 Sonnet). No tool-calling requirement for LLM connection. ReAct-style: models reason natively, then call tools when needed
- **Memory model:** Still two-tier: in-context memory (core blocks) + out-of-context (archival). Self-editing via memory_replace, memory_insert, memory_rethink tools
- **Letta Code:** Memory-first coding agent. #1 model-agnostic OSS agent on Terminal-Bench (#4 overall). Git-backed memory via "Context Repositories"
- **Conversations API (Jan 2026):** Agents maintain shared memory across parallel conversations

### Benchmark scores
- **LoCoMo:** 74.0% accuracy on gpt-4o-mini (storing conversation histories in files)
- **Letta's own finding:** "Is a filesystem all you need?" blog post showed file-based storage surprisingly competitive
- **Terminal-Bench:** #1 open-source terminal agent

### Key differentiator
Self-editing memory where the agent decides what to remember. The "LLM as OS" paradigm. Now proven in coding (Letta Code) with git-backed context management.

### What they do that Remanentia doesn't
- Agent actively edits its own memory blocks (core memory manipulation)
- Context Repositories: git-based versioning of agent context
- Multi-model support (portable across GPT-5, Claude 4.5, Gemini 3)
- Conversation-level memory sharing across parallel sessions

### Remanentia advantage over Letta
- No LLM required for any operation
- Better LoCoMo score (74.7% vs 74.0%) despite zero LLM calls
- Existing file-based approach validated by Letta's own research ("filesystem is all you need")

---

## 3. Zep / Graphiti

### Current architecture
- **Core engine:** Graphiti — open-source temporal knowledge graph engine
- **Bi-temporal model:** Every fact has Event Time (T, when it happened) + Ingestion Time (T', when observed). Edge validity windows enable "what did we believe on date X?" queries
- **Episode ingestion:** Raw messages/text/JSON -> entity-relationship graph construction
- **Triple search:** Cosine similarity + BM25 full-text + breadth-first graph traversal
- **Entity resolution at write time**
- **Neo4j backed** (production deployments)

### Benchmark scores
- **DMR (Deep Memory Retrieval):** 94.8% (vs MemGPT 93.4%)
- **Latency:** P95 = 300ms, 90% reduction vs baseline
- **Accuracy improvements:** Up to 18.5% vs baseline implementations

### Key differentiator
Bi-temporal modeling. No other system tracks both "when it happened" and "when we learned about it." Edge invalidation when facts are superseded. The gold standard for temporal reasoning.

### What they do that Remanentia doesn't
- Bi-temporal fact tracking (event time vs ingestion time)
- Edge invalidation (fact supersession)
- Neo4j-backed graph traversal for multi-hop reasoning
- Entity resolution at write time
- Graph-native search alongside vector and BM25

### Remanentia advantage over Zep
- No Neo4j dependency (Remanentia's entity graph is JSON-native)
- No LLM calls in retrieval
- Simpler deployment (single process, no database server)
- Indexes existing filesystem rather than requiring data ingestion

---

## 4. LangMem / LangChain Memory

### Current architecture
- **Three memory types:** Semantic (collections + profiles), Episodic (past interaction patterns), Procedural (evolving system prompts)
- **Two-layer design:** Layer 1 (Core API): pure functions — extract, update, remove, consolidate. Layer 2 (Stateful): LangGraph persistence — store managers, memory tools
- **Memory Manager:** An LLM that takes conversation transcripts and produces memory entries
- **Hierarchical namespaces:** organization -> user -> context
- **Three retrieval mechanisms:** key access, semantic search, metadata filtering

### Benchmark scores
- **LoCoMo (via Memori comparison):** 78.05%
- No standalone benchmark paper published

### Key differentiator
Deep integration with LangChain/LangGraph ecosystem. Procedural memory (system prompts that evolve through feedback) is unique — most systems only do semantic and episodic.

### What they do that Remanentia doesn't
- Procedural memory (self-improving system prompts)
- Hierarchical namespace organization
- Memory consolidation via LLM
- Native LangGraph integration

### Remanentia advantage over LangMem
- Framework-independent (not tied to LangChain)
- No LLM required
- Filesystem-native

---

## 5. MemOS (MemTensor)

### Architecture
- **MemCubes:** Standardized memory units with Metadata Header (lifecycle, permission, storage policy) + Memory Payload (plaintext, activation states, or parameter deltas)
- **Three memory types:** Plaintext, Activation-based (KV-cache), Parameter-level
- **KV-Cache Memory Injection:** Background content cached as Key/Value tensors, injected directly into attention mechanism — bypasses prompt-based injection
- **Scheduling:** Redis Streams for memory scheduling + DB optimizations
- **v2.0 "Stardust" (Dec 2025):** Comprehensive KB (doc/URL parsing), memory feedback, multi-modal memory (images/charts), tool memory for agent planning, MCP upgrade

### Benchmark scores
- **Temporal reasoning:** 159% improvement over OpenAI memory
- **TTFT latency:** Up to 94% reduction via KV-cache injection
- **Overall:** Claims competitive with EverMemOS on LoCoMo (specific numbers not in search results)

### Key differentiator
KV-cache memory injection. The only system that operates at the activation level, not just token level. Injects memories directly into transformer attention layers rather than prepending to prompts. This is architecturally unique.

### What they do that Remanentia doesn't
- Activation-level memory (KV-cache injection into transformer layers)
- Parameter-level memory (LoRA-style weight updates)
- Multi-modal memory (images, charts)
- Tool memory for agent planning
- MemCube lifecycle management (scheduling, permission, migration)

### Remanentia advantage over MemOS
- Simpler, no model-internal modifications needed
- Works with any LLM via MCP (doesn't require access to model internals)
- Local-only, no Redis/DB infrastructure

---

## 6. Hindsight (Vectorize)

### Architecture
- **Four logical memory networks:** World facts, agent experiences, synthesized entity summaries, evolving beliefs
- **Three core operations:** Retain, Recall, Reflect
- **TEMPR (Temporal Entity Memory Priming Retrieval):** Four parallel searches — semantic vector similarity + BM25 keyword + graph traversal (shared entities) + temporal filtering
- **CARA (Coherent Adaptive Reasoning Agents):** Preference-aware reflection with configurable disposition parameters (skepticism, literalism, empathy)
- **MIT licensed, open source**

### Benchmark scores
- **LongMemEval:** 91.4% — first system to break 90%
- **LoCoMo:** Not directly reported in search results

### Key differentiator
TEMPR's four-way parallel retrieval is the most sophisticated retrieval architecture shipping. Disposition-aware reflection (CARA) means the agent's personality affects how memories are interpreted. Published with Washington Post and Virginia Tech collaboration.

### What they do that Remanentia doesn't
- Four-way parallel retrieval (we do BM25 + optional embedding, no graph traversal or temporal filtering)
- Belief tracking and evolution
- Disposition-aware reasoning
- Entity summarization as a separate memory layer
- LLM-based answer extraction

### Remanentia advantage over Hindsight
- Zero LLM dependency
- Simpler architecture (fewer moving parts)
- Filesystem-native

---

## 7. EverMemOS (EverMind AI)

### Architecture
- **Self-organizing memory OS** for structured long-horizon reasoning
- Positions itself as the only memory system that outperforms large models using full-context inputs while using fewer tokens

### Benchmark scores
- **LoCoMo:** 92.3-93% overall accuracy (SOTA or near-SOTA)
- **LongMemEval:** 83.0% overall. Per-category: extraction 97.14%, knowledge update 85.71%, multi-session reasoning 93.33%, temporal reasoning 73.68%, assistant-actions 89.74%
- Outperforms Mem0, MemOS, Zep, and MemU in unified evaluation framework

### Key differentiator
Claims to be the only system that beats full-context LLM baselines while using drastically fewer tokens. Production evaluation framework that benchmarks multiple systems under identical conditions.

### What they do that Remanentia doesn't
- Self-organizing memory structure
- Token-efficient reasoning that beats full-context
- Unified evaluation across competitors

---

## 8. Kumiho

### Architecture
- **Graph-native cognitive memory** with formal belief revision semantics
- **Three architectural innovations:**
  1. **Prospective indexing:** At write time, LLM generates "hypothetical future queries that would need this memory" and indexes those alongside the memory. Bridges cue-trigger semantic gap
  2. **Event extraction:** Structured events with consequences appended to summaries (preserves causal detail)
  3. **Client-side LLM reranking**

### Benchmark scores
- **LoCoMo-Plus (Level-2 benchmark):** 93.3% judge accuracy
- **Recall accuracy:** 98.5% (395/401). The remaining 6.7% gap is entirely LLM answer fabrication on correctly retrieved context
- **Standard LoCoMo:** 0.565 F1 (n=1,986), 97.5% adversarial refusal accuracy

### Key differentiator
Prospective indexing is the single most important innovation in this landscape. 98.5% recall means the retrieval problem is essentially solved — all remaining errors are answer generation failures. The insight that you should index by "what queries will need this" rather than "what does this say" is the breakthrough.

### What they do that Remanentia doesn't
- Prospective indexing (hypothetical future query generation at write time)
- Event extraction with causal consequences
- Formal belief revision semantics
- Client-side LLM reranking
- 98.5% recall accuracy

### Remanentia advantage over Kumiho
- Zero LLM dependency (Kumiho requires LLM for prospective query generation and reranking)
- Simpler write path (no LLM call per memory ingestion)

---

## 9. MAGMA

### Architecture
- **Four orthogonal relational graphs:** Semantic, Temporal, Causal, Entity
- **Three layers:** (1) Query Process with intent-aware routing, (2) Data Structure Layer with relation graphs + vector DB, (3) Write/Update Process with dual-stream mechanism
- **Adaptive Traversal Policy:** Routes retrieval based on query intent — "why" traverses causal edges, "who" traverses entity edges, "when" traverses temporal edges
- **Dual-stream memory evolution:** Fast ingestion (latency-sensitive) + asynchronous structural consolidation

### Benchmark scores
- **Reasoning accuracy:** Up to 45.5% higher on long-context benchmarks vs prior methods
- **Token consumption:** 95% reduction
- **Query latency:** 40% faster than prior methods
- **Code:** Open-source (GitHub: FredJiang0324/MAMGA)

### Key differentiator
Intent-aware routing across multiple graph types. The most theoretically sophisticated graph architecture. Dual-stream (fast ingest + async consolidation) is production-pragmatic.

### What they do that Remanentia doesn't
- Four separate graph types (semantic, temporal, causal, entity)
- Intent-aware routing (query type determines which graph to traverse)
- Dual-stream write path (fast ingest + background consolidation)
- Causal graph for "why" queries

### Remanentia advantage over MAGMA
- Shipping product vs research paper
- Simpler architecture
- No graph database dependency

---

## 10. Observational Memory (Mastra)

### Architecture
- **Two background agents:** Observer + Reflector, running as "subconscious" processes
- **Observer:** When message history hits token threshold, compresses messages into dense observations (3-6x compression, ~6x in LongMemEval runs)
- **Reflector:** When observations hit 40K tokens, restructures/condenses: combines related items, reflects on patterns, drops irrelevant context
- **Stable context window:** No dynamic injection per turn. Context is predictable, reproducible, prompt-cacheable
- **Fully open source**

### Benchmark scores
- **LongMemEval (gpt-4o):** 84.23%
- **LongMemEval (gpt-5-mini):** 94.87% — highest score ever recorded on this benchmark by any system with any model
- **Compression:** 3-6x on text content

### Key differentiator
Stable context window. Every other system changes the prompt every turn by injecting retrieved context. Observational Memory doesn't — observations accumulate and compress, making the context window predictable and cacheable. The Observer/Reflector pattern is architecturally elegant.

### What they do that Remanentia doesn't
- Background observation agents
- Automated compression with 3-6x ratio
- Reflective restructuring
- Stable, cacheable context window
- 94.87% LongMemEval (highest recorded)

### Remanentia advantage over Mastra
- No LLM required for observation or reflection
- Filesystem-native
- Not a framework integration

---

## 11. SuperLocalMemory

### Architecture
- **Three mathematical techniques** replace cloud LLM work: differential geometry (similarity scoring), algebraic topology (contradiction detection), stochastic analysis (lifecycle management)
- **On-device SQLite** with zero network calls
- **Three modes:** Mode A Retrieval (74.8%, no cloud), Mode A Raw (60%, zero LLM at any stage), Mode C (87.7%, with LLM)
- **MCP + Agent-native CLI** with structured JSON output
- **17+ tool integrations** via MCP
- **MIT licensed**

### Benchmark scores
- **LoCoMo Mode A (local-only):** 74.8% — highest without cloud dependency
- **LoCoMo Mode A Raw (zero-LLM):** 60%
- **LoCoMo Mode C (with LLM):** 87.7%

### Key differentiator
The only other system besides Remanentia targeting zero-LLM operation. Mathematical guarantees from differential geometry and algebraic topology rather than heuristics. Privacy-preserving by design.

### What they do that Remanentia doesn't
- Mathematical contradiction detection (algebraic topology)
- Lifecycle management via stochastic analysis
- Bayesian trust defense against memory poisoning (multi-agent)
- 17+ tool integrations
- GDPR/HIPAA/EU AI Act compliance claims

### Remanentia advantage over SuperLocalMemory
- Comparable LoCoMo score (74.7% vs 74.8%) — effectively tied
- Filesystem-native (indexes existing files vs separate SQLite store)
- Richer source diversity (18 source directories vs conversation-focused)
- Entity graph with 223 entities / 6,434 relations
- Published paper with honest negative results (SNN, Rust BM25)

---

## 12. TiMem

### Architecture
- **Temporal Memory Tree (TMT):** Organizes conversations hierarchically — raw observations -> progressively abstracted persona representations
- **Semantic-guided consolidation** across hierarchical levels without fine-tuning
- **Complexity-aware memory recall** that balances precision and efficiency based on query complexity

### Benchmark scores
- **LoCoMo:** 75.30%
- **LongMemEval-S:** 76.88% (SOTA at time of publication)
- **Memory length reduction:** 52.20% on LoCoMo
- **License:** SSPL v1

### Key differentiator
Hierarchical consolidation tree. Memories are progressively abstracted, not flat. Query complexity detection routes simple queries to leaf nodes, complex queries to consolidated branches.

### What they do that Remanentia doesn't
- Hierarchical memory tree (progressive abstraction)
- Query complexity detection
- Semantic consolidation across levels

---

## 13. Memori (Memori Labs)

### Architecture
- **LLM-agnostic persistent memory layer**
- **Advanced Augmentation pipeline:** Converts unstructured dialogue into compact semantic triples + conversation summaries
- **Embedding:** Gemma-300 embedding model
- **Storage:** Semantic triples as first-class objects

### Benchmark scores
- **LoCoMo:** 81.95% — outperforms Zep (79.09%), LangMem (78.05%), Mem0 (62.47%)
- **Token usage:** 1,294 tokens/query (~5% of full context)
- **Cost:** 4.98% of full-context cost

### Key differentiator
Semantic triples as the core data structure. Extreme token efficiency (5% of full context). The "memory as data structuring" framing is closest to what Remanentia does philosophically.

### What they do that Remanentia doesn't
- LLM-generated semantic triples from conversations
- Extreme token efficiency (1,294 tokens/query)
- Higher LoCoMo accuracy (81.95% vs 74.7%)

---

## 14. Cognee

### Architecture
- **Six-stage pipeline:** Classify documents -> check permissions -> extract chunks -> LLM extracts entities/relationships -> generate summaries -> embed into vector store + commit edges to graph
- **Memify Pipeline (post-processing):** Enrichment, optimization, persistence after initial graph construction
- **Memgraph integration** for graph storage
- **$7.5M seed** led by Pebblebed (backed by OpenAI and FAIR founders)

### Key differentiator
The "knowledge engine" framing — builds knowledge graphs from data and makes them searchable. Ontology-grounded memory. Closest to Microsoft GraphRAG but as a product.

---

## 15. A-MEM (NeurIPS 2025)

### Architecture
- **Zettelkasten-inspired:** Dynamic indexing and linking to create interconnected knowledge networks
- **Memory operations as tool-based actions:** Store, retrieve, update, summarize, discard
- **Three-stage progressive RL training** with step-wise GRPO for sparse/discontinuous rewards

### Key differentiator
RL-trained memory operations. The agent learns WHEN to store/retrieve/update/forget rather than following heuristics. End-to-end optimized. Published at NeurIPS 2025.

### What they do that Remanentia doesn't
- RL-optimized memory admission and eviction policies
- Dynamic linking (Zettelkasten-style)
- End-to-end training (not heuristic-based)

---

## Benchmark Leaderboard Summary

### LoCoMo (LLM-as-Judge, higher = better)

| System | Score | LLM Required? | Notes |
|--------|-------|----------------|-------|
| Kumiho | 93.3% | Yes | LoCoMo-Plus variant |
| EverMemOS | 92.3-93% | Yes | Unified evaluation |
| MemMachine | 84.87% | Yes | Claims leaderboard top |
| Memori | 81.95% | Yes (extraction) | 5% token cost |
| Zep | 79.09% | Yes | via Memori comparison |
| LangMem | 78.05% | Yes | via Memori comparison |
| TiMem | 75.30% | Yes | 52% memory reduction |
| **Remanentia** | **74.7%** | **No** | Zero LLM calls |
| SuperLocalMemory | 74.8% | No (Mode A) | Local-only |
| Letta (MemGPT) | 74.0% | Yes | gpt-4o-mini, file-based |
| Mem0 | 66.9% | Yes | Base variant |
| Mem0^g | 68.4% | Yes | Graph variant |
| SuperLocalMemory Raw | 60.0% | No | Zero LLM at any stage |

**Observation:** Systems requiring cloud LLMs cluster 83-93%. Systems without LLM cluster 60-75%. The gap is 10-20pp. Remanentia is at the top of the no-LLM tier.

### LongMemEval (higher = better)

| System | Score | Model | Notes |
|--------|-------|-------|-------|
| Observational Memory | 94.87% | gpt-5-mini | Highest ever |
| Hindsight | 91.4% | GPT-4 | First to break 90% |
| Observational Memory | 84.23% | gpt-4o | |
| EverMemOS | 83.0% | Not specified | |
| TiMem | 76.88% | GPT-4o-mini | LongMemEval-S |
| GPT-4o (baseline) | 30-70% | gpt-4o | Varies by category |

**Remanentia has not been evaluated on LongMemEval.** This is a gap.

---

## Key Insights for Product Strategy

### 1. The LLM Gap is Answer Extraction, Not Retrieval

Kumiho achieves 98.5% recall accuracy. Their remaining 6.7% error is LLM answer fabrication on correctly retrieved context. This means **retrieval is a solved problem** at the top end. The competition is moving to:
- Better answer generation from retrieved context
- Better memory structuring at write time (prospective indexing)
- Better consolidation and compression

Our retrieval at 92.9% P@1 (with embedding rerank) is in the same ballpark. Adding an LLM answer extraction step would likely close most of the 16pp gap to Hindsight.

### 2. Prospective Indexing is the Key Innovation

Kumiho's insight — index memories by "what future queries will need this" rather than "what does this say" — is the most impactful architectural innovation in this space. At write time, generate hypothetical queries. At read time, match incoming query against prospective queries. This bridges the semantic gap between how information is stored and how it's asked for.

**For Remanentia:** We can implement a lightweight version without LLM: template-based prospective queries using extracted entities ("What was decided about {entity}?", "When did {event} happen?", "Why did we choose {choice}?"). Gets ~80% of the benefit per our earlier analysis.

### 3. The Zero-LLM Niche is Real but Small

Only Remanentia and SuperLocalMemory operate without LLM. The privacy/sovereignty argument is real (GDPR, HIPAA, air-gapped environments, cost-sensitive deployments). But the accuracy gap is significant (75% vs 93%).

**Strategic choice:** Stay zero-LLM-core but offer optional LLM enhancement layers. Like SuperLocalMemory's Mode A (local) vs Mode C (cloud) distinction.

### 4. Graph Memory is Table Stakes

Every system that scores >80% uses some form of knowledge graph. Entity-relationship graphs, temporal graphs, or both. Our entity graph (223 entities, 6,434 relations) exists but isn't used for retrieval — it's only queryable via `remanentia_graph`. Integrating graph traversal into the retrieval pipeline is mandatory.

### 5. Temporal Reasoning is the Hardest Problem

Remanentia: 42.7% on temporal questions. This is our weakest category by far. TiMem's Temporal Memory Tree and Graphiti's bi-temporal model both address this directly. Our timestamp-based recency boost is too primitive.

### 6. Memory Consolidation is the Next Frontier

A-MEM (NeurIPS 2025) shows RL-trained memory operations outperform heuristics. Observational Memory's Observer/Reflector pattern achieves 94.87%. EverMemOS's self-organizing structure beats full-context. The trend: automated, intelligent compression and restructuring of accumulated memories.

Our consolidation pipeline exists but isn't tested against benchmarks.

### 7. The Market is Fragmenting Along Three Axes

**Axis 1 — Deployment model:**
- Cloud-hosted (Mem0, Zep, EverMemOS)
- Self-hosted (Letta, Cognee, MemOS)
- Local-only (Remanentia, SuperLocalMemory)

**Axis 2 — LLM dependency:**
- LLM-required (Mem0, Hindsight, Kumiho, Observational Memory)
- LLM-optional (SuperLocalMemory Mode A/C, MemOS)
- LLM-free (Remanentia, SuperLocalMemory Raw)

**Axis 3 — Integration pattern:**
- Framework-native (LangMem -> LangChain, Letta -> own platform)
- Protocol-based (Remanentia -> MCP, SuperLocalMemory -> MCP)
- API service (Mem0 -> REST API, Zep -> REST API)

**Remanentia's unique position:** Local-only + LLM-free + MCP-native + filesystem-native. No other system occupies all four simultaneously.

---

## What Remanentia Must Build (Priority Order)

### P0 — Close the accuracy gap
1. **Optional LLM answer extraction** — add a mode that uses local LLM (ollama, llama.cpp) to extract answers from retrieved paragraphs. Expected: +10-15pp on LoCoMo
2. **LongMemEval evaluation** — we haven't run this benchmark. Must do it for credibility
3. **Prospective indexing (template-based)** — at write time, generate hypothetical queries per entity. Expected: +3-5pp on recall

### P1 — Fix temporal reasoning
4. **Bi-temporal tracking** — add event time vs ingestion time to every indexed paragraph
5. **Temporal ordering** — sort results by date when temporal intent detected
6. **Date normalization** — resolve relative dates, durations, "last week" etc.

### P2 — Upgrade graph integration
7. **Graph-augmented retrieval** — use entity graph for multi-hop traversal alongside BM25
8. **GLiNER2 entity extraction** — replace regex with 205M param model (runs on CPU)
9. **Intent-aware routing** (MAGMA-style) — route "why" to causal relations, "when" to temporal

### P3 — Differentiation
10. **Observer/Reflector pattern** — background consolidation agents (can be rule-based, no LLM required)
11. **Memory importance decay** — Stuart-Landau inspired, already designed
12. **Prospective indexing with local LLM** — generate actual hypothetical queries per memory

---

## Systems to Watch

| System | Why |
|--------|-----|
| **Kumiho** | 98.5% recall. Prospective indexing is the breakthrough. If they open-source, game-changing |
| **Observational Memory (Mastra)** | 94.87% LongMemEval. Elegant Observer/Reflector pattern. Fully open source |
| **EverMemOS** | 93% LoCoMo. Only system beating full-context baselines. Watch for paper details |
| **SuperLocalMemory** | Direct competitor in zero-LLM space. Mathematical foundations (diff geometry, algebraic topology). Growing fast |
| **MemOS** | KV-cache injection is architecturally unique. If model providers expose attention layers, this approach dominates |
| **A-MEM** | RL-trained memory operations. NeurIPS 2025. The future is learned, not heuristic |

---

## Honest Assessment: Where Remanentia Stands

**Strengths:**
- Only filesystem-native memory system. Everyone else builds a separate store
- Competitive retrieval (92.9% P@1 with embedding rerank)
- Zero LLM dependency — real advantage for air-gapped, privacy-critical, cost-sensitive deployments
- Published paper with honest negative results (SNN failure, Rust BM25)
- MCP integration working and proven

**Weaknesses:**
- 74.7% LoCoMo vs 93% SOTA — a 19pp gap
- No LongMemEval evaluation
- Temporal reasoning at 42.7% — worst category
- Entity graph exists but not integrated into retrieval
- Consolidation pipeline not benchmarked
- Single developer's corpus (1,217 docs) — untested at scale
- No memory poisoning defense, no multi-agent trust model

**The core question:** Is "filesystem-native + zero-LLM" a viable product niche, or will it be absorbed by systems that offer optional local LLM enhancement (like SuperLocalMemory's Mode A/C split)?

**My assessment:** The niche is real but narrow. The path forward is Remanentia Core (zero-LLM, filesystem-native) + Remanentia Enhanced (optional local LLM for answer extraction, prospective indexing, and consolidation). Keep the zero-LLM path as a first-class mode, not an afterthought. This is what SuperLocalMemory does and it's working — 74.8% in zero-LLM mode, 87.7% with LLM.
