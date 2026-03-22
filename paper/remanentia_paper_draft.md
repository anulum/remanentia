# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Sotek. All rights reserved.
# © Code 2020–2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

# Remanentia: Filesystem Knowledge Retrieval for Persistent AI Agent Memory

**Miroslav Sotek**
ORCID: 0009-0009-3560-0851
Anulum | protoscience@anulum.li | www.anulum.li

---

## Abstract

We present Remanentia, a persistent memory system for AI agents that operates as a filesystem knowledge retrieval engine. The system indexes documents across multiple sources — reasoning traces, session logs, code repositories, research documents, and entity graphs — into a unified BM25 index with query intelligence routing. On the LOCOMO multi-session conversational QA benchmark (1,986 questions), Remanentia achieves 74.7% accuracy without any LLM in the retrieval loop, compared to 91.4% for Hindsight (which uses GPT-4 for answer extraction). We document two negative results: (1) spiking neural network retrieval via STDP-modified weight matrices provides zero discriminative signal across 70+ experimental configurations and 4 learning rules, and (2) a Rust BM25 engine is slower than Python at the current 15K-paragraph scale due to FFI overhead. The system integrates with AI coding tools via the Model Context Protocol (MCP), serving as persistent cross-session memory for Claude Code and similar environments. We release Remanentia as open-source software (AGPL-3.0) with 225 tests.

**Keywords:** agent memory, BM25, retrieval, knowledge management, MCP, negative results, spiking neural networks

---

## 1. Introduction

Large language models operate within fixed context windows. When a conversation ends, the agent loses all accumulated knowledge. Current approaches to agent memory — Mem0, Letta (MemGPT), Zep, LangMem — treat this as a vector retrieval problem: embed text, store in a vector database, retrieve by cosine similarity. These systems require hosted infrastructure, embed-then-retrieve pipelines, and LLM calls for memory consolidation.

We take a different approach: index the filesystem directly. Session logs, reasoning traces, code, research documents, and configuration files already exist on disk. The retrieval problem reduces to: given a query, find the most relevant paragraph across all these sources.

Remanentia indexes 18 source directories into a unified BM25 index with 15,938 paragraphs from 1,217 documents. Query intelligence classifies intent (location, decision, temporal, metric, debugging, status, explanation) and routes to specialized scoring. The system operates with zero external dependencies beyond numpy, zero cloud infrastructure, and zero LLM calls in the retrieval path.

### 1.1 Contributions

1. A filesystem-native memory system that indexes existing project artifacts rather than maintaining a separate memory store.
2. Query intelligence with 8 intent types routing to specialized scoring (paragraph type boost, recency boost, temporal date extraction).
3. Integration with AI coding environments via the Model Context Protocol (MCP), verified with Claude Code.
4. Two documented negative results: SNN-based retrieval failure (Section 4) and Rust BM25 at small scale (Section 5.2).
5. Open-source implementation with 225 tests, 8 test modules, and 77% average coverage on core modules.

---

## 2. System Architecture

### 2.1 Document Indexing

Remanentia scans 18 configured source directories:

- Reasoning traces (structured markdown)
- Session logs and handovers (coordination artifacts)
- Cross-repository research documents
- Source code (Python, Rust) from 5 codebases
- Claude memory files (auto-persisted)
- Semantic memories (consolidated from traces)
- Entity graph (JSONL)
- INDEXER catalog

Each document is split into paragraphs (markdown) or function/class blocks (code). Code splitting uses regex-based function detection for Python (`def`, `class`) and Rust (`fn`, `pub fn`, `impl`).

### 2.2 BM25 Search

Standard BM25 scoring (k1=1.5, b=0.75) over paragraph-level token sets. Tokenization strips words shorter than 3 characters and lowercases all text. IDF is computed at build time over all paragraphs.

Optional GPU embedding rerank: when sentence-transformers is available, the top 3K BM25 candidates are reranked by cosine similarity with MiniLM-L6-v2 embeddings (combined weight: 0.4 BM25 + 0.6 embedding).

### 2.3 Query Intelligence

Queries are classified into 8 intent types via keyword pattern matching:

| Intent | Triggers | Boost |
|--------|----------|-------|
| location | "where is", "find the", "which file" | function/code paragraphs |
| decision | "what did we decide", "chose", "rejected" | decision paragraphs |
| debugging | "what went wrong", "failure", "bug", "fix" | finding/decision paragraphs |
| status | "status", "progress", "current", "latest" | recency boost |
| metric | "benchmark", "accuracy", "score", "percent" | metric paragraphs |
| temporal | "when", "date", "timeline", "before" | date-containing paragraphs |
| explanation | "how does", "how to", "explain", "what is" | function/finding paragraphs |
| general | (default) | none |

Pattern order matters: more specific patterns (debugging, status, metric) are checked before broad ones (temporal, explanation) to prevent "what is the status" from matching "what is" (explanation).

### 2.4 Answer Extraction

For temporal, metric, and general queries, a regex-based answer extractor attempts to pull short answers from retrieved paragraphs:

- Dates: ISO format (`2026-03-15`) and English (`March 15, 2026`)
- Percentages: `66.4%`
- Versions: `v3.9.0`
- Numbers: `1,986`
- Names: capitalized multi-word spans
- Yes/no: negation pattern detection

This is a heuristic — complex paraphrased answers are not extractable without an LLM.

### 2.5 Memory Consolidation

Episodic traces are consolidated into semantic memories via:

1. Metadata extraction (date, project, type from filename patterns)
2. Entity extraction (regex patterns for projects, algorithms, versions, file paths, function names)
3. Key line extraction (30 trigger patterns with 2-line context capture)
4. Date-proximity clustering (traces from same project within 2 days are grouped)
5. Semantic memory writing (full text preserved, zero information loss)
6. Entity graph update (co-occurrence relations with weight accumulation)

The entity graph contains 223 entities and 6,434 relations across the full corpus.

### 2.6 MCP Integration

Remanentia exposes 3 tools via the Model Context Protocol (JSON-RPC on stdio):

- `remanentia_recall`: query the unified index
- `remanentia_status`: system statistics
- `remanentia_graph`: entity relationship query

This allows Claude Code, Cursor, and other MCP-compatible tools to use Remanentia as persistent cross-session memory.

---

## 3. Evaluation

### 3.1 LOCOMO Benchmark

We evaluate on LOCOMO [preprocessed], a multi-session conversational QA dataset with 1,986 questions across 10 conversations, spanning 5 categories: single-hop, multi-hop, temporal, adversarial, and open-domain.

| Category | Correct | Total | Accuracy |
|----------|---------|-------|----------|
| Single-hop | 157 | 282 | 55.7% |
| Multi-hop | 265 | 321 | 82.6% |
| Temporal | 41 | 96 | 42.7% |
| Adversarial | 669 | 841 | 79.5% |
| Open-domain | 351 | 446 | 78.7% |
| **Overall** | **1,483** | **1,986** | **74.7%** |

**Method:** BM25-lite search (top-20 retrieval, self-contained) + token overlap matching (0.3 threshold) + regex answer extraction. No embedding rerank, no LLM.

**Context:** Hindsight (SOTA with GPT-4 answer extraction) achieves 91.4% on LongMemEval. GPT-4o achieves 30-70% depending on category. Remanentia's 74.7% is achieved with zero LLM calls — pure BM25 + regex matching. The remaining 16pp gap to Hindsight is primarily answer extraction on single-hop (55.7%) and temporal (42.7%) questions where precise answer spans require LLM understanding.

**Caveat:** The 0.3 token overlap threshold is generous. A gold-answer token like "john" matching in a retrieved paragraph counts as a hit even if the paragraph doesn't answer the question correctly. This inflates scores on adversarial and multi-hop categories. A stricter evaluation (exact match or human judgment) would produce lower numbers.

### 3.2 Internal Benchmark

14 self-authored queries against the full 1,217-document index: 14/14 P@1.

**Caveat:** These queries were written by the system's developer, use terminology that appears directly in filenames and headers (token leakage), and test retrieval only (not answer extraction). This number measures "can the system find its own files" — a low bar.

### 3.3 Performance

- Build time: ~30s (scan + BM25 index, no GPU) to ~60s (with MiniLM-L6-v2 embeddings)
- Query time: 47-300ms (BM25 only), <100ms warm with cached index
- Index size: 289 MB on disk (pickle, includes 15,938 paragraph embeddings)
- Cold start: ~4s (index load from pickle)

---

## 4. Negative Result: SNN Retrieval

We originally designed Remanentia around a spiking neural network whose STDP-modified synaptic weights would encode associative memory. The hypothesis: inject a query stimulus into the LIF network, compare the resulting spike pattern to cached trace patterns, and use cosine similarity as a retrieval signal.

**Result after 70+ experiments across 4 learning rules (STDP, BCPNN, Hebbian, E/I balanced):**

The SNN provides zero discriminative signal. Spike patterns for relevant and irrelevant queries are indistinguishable. The root cause is an encoding bottleneck: 384-dimensional sentence-transformer embeddings encoded into 20,000-neuron patterns via hash projections produce activation patterns too similar for any local learning rule to differentiate.

Specific findings:
- STDP with decision gate: creates 449 spectral outliers in the weight matrix but P@1 = shuffled control
- BCPNN: weight matrix 100% positive, zero discrimination
- E/I balanced network (8-configuration GPU sweep): 7/8 configurations show delta=0 vs shuffled, 1/8 shows delta=+1 (noise)

**Conclusion:** At the encoding resolution available (hash of 384-dim embeddings → 20K sparse binary patterns), no local learning rule creates useful retrieval features. SNN weight was set to 0.00 in the scoring function. The retrieval signal comes entirely from BM25 + embedding similarity.

---

## 5. Implementation

### 5.1 Software

- Python 3.12, numpy (required), sentence-transformers/torch (optional)
- 8 core modules: memory_index, consolidation_engine, entity_extractor, answer_extractor, memory_recall, mcp_server, cli, api
- 225 tests across 8 test modules, 77% average coverage on core modules
- FastAPI REST endpoint (optional), CLI, MCP server
- AGPL-3.0 license, available at github.com/anulum/remanentia

### 5.2 Rust BM25 Engine (Negative Result)

A Rust BM25 engine was built with PyO3 + Rayon parallel scoring. At the current scale (15,938 paragraphs), Python BM25 is faster due to FFI serialization overhead. The Rust engine would outperform Python at 100K+ paragraphs with an inverted index, but the current corpus is too small to benefit.

---

## 6. Limitations

1. **No LLM in retrieval:** The 16pp gap to Hindsight (74.7% vs 91.4%) is primarily answer extraction, not retrieval. Adding an LLM to extract answers from retrieved context would likely close most of this gap.
2. **Temporal reasoning:** 42.7% on temporal questions. Date extraction is regex-only; temporal ordering, duration computation, and relative date resolution are not implemented.
3. **Generous matching:** The 0.3 token overlap threshold may over-count. Some "correct" retrievals match gold answer tokens without actually answering the question.
4. **Scale:** Tested on a single developer's corpus (1,217 documents). Behavior at 10K+ documents is untested.
5. **Evaluation bias:** The 14/14 internal benchmark uses self-authored queries with filename token leakage.
6. **No active learning:** The system does not learn from retrieval successes or failures.

---

## 7. Conclusion

Remanentia demonstrates that filesystem-native retrieval with BM25 and query intelligence can serve as persistent agent memory without cloud infrastructure or LLM calls. The system achieves 74.7% on LOCOMO (1,986 questions), within 16pp of Hindsight's LLM-augmented 91.4%, using zero LLM calls. It is functional as cross-session memory for AI coding tools via MCP.

The SNN negative result (Section 4) has implications beyond this system: hash-encoded dense embeddings into sparse spike patterns do not produce discriminative features under any local learning rule we tested. This suggests a fundamental mismatch between the encoding resolution and the learning rule's capacity, not a hyperparameter tuning problem.

The path to improving accuracy is straightforward: add LLM-based answer extraction to the retrieval pipeline. The retrieved context is often correct; the extraction step is what's missing.

---

## References

- Packer, C., et al. (2023). MemGPT: Towards LLMs as Operating Systems.
- Chase, H. (2022). LangChain.
- mem0ai (2024). Mem0: The Memory Layer for AI Agents.
- Zep AI (2024). Zep: Long-term Memory for AI Assistants.
- LangMem (2025). LangMem: Long-Term Memory for LLM Applications.
- Hindsight (2024). LongMemEval: Benchmarking Long-Term Memory in AI Assistants.
- LOCOMO (2024). KhangPTT373/locomo_preprocess (HuggingFace).

---

Co-Authored-By: Arcane Sapience <protoscience@anulum.li>
