# AI Agent Memory Systems: Cutting-Edge Research Survey
# Compiled: 2026-03-20 | For: Remanentia Product Development

---

## 1. Spiking Neural Networks for Memory

### 1.1 Bio-Inspired Spike-Based Content-Addressable Memory (Hippocampus CA3)

- **Title:** Bio-inspired computational memory model of the Hippocampus: An approach to a neuromorphic spike-based Content-Addressable Memory
- **Authors:** Daniel Casanueva-Morato, Alvaro Ayuso-Martinez, Juan P. Dominguez-Morales, Angel Jimenez-Fernandez, Gabriel Jimenez-Moreno
- **Year:** 2024
- **Key Finding:** First hardware implementation of a fully-functional bio-inspired spiking hippocampal content-addressable memory (CAM) model. Based on the CA3 region of the hippocampus, it learns, forgets, and recalls memories (both orthogonal and non-orthogonal) from any fragment. Implemented on the SpiNNaker neuromorphic hardware using SNNs with STDP.
- **URL:** https://www.sciencedirect.com/science/article/pii/S0893608024003988
- **DOI:** 10.1016/j.neunet.2024.106394
- **Remanentia Relevance:** Direct precedent for our SNN-based memory retrieval; validates that spike-based CAM on neuromorphic hardware is feasible and performant.

### 1.2 Spiking Representation Learning for Associative Memories

- **Title:** Spiking representation learning for associative memories
- **Authors:** Naresh Ravichandran, Anders Lansner, Pawel Herman
- **Year:** 2024
- **Key Finding:** SNN model grounded in the Bayesian Confidence Propagation Neural Network (BCPNN) framework learns sparse distributed representations and employs them for associative memory. Incorporates Hebbian plasticity, structural plasticity (activity-dependent rewiring), Poisson-spiking neurons, and neocortical columnar architecture. Demonstrates that sparsely spiking Poissonian neurons match non-spiking rate-based networks in representation learning and associative memory.
- **URL:** https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1439414/full
- **Remanentia Relevance:** Validates our approach of using sparse spike representations for memory encoding; the BCPNN framework offers an alternative learning rule to pure STDP.

### 1.3 SNN-Based Bidirectional Associative Learning

- **Title:** Spiking Neural Network-Based Bidirectional Associative Learning Circuit for Efficient Multibit Pattern Recall in Neuromorphic Systems
- **Authors:** (MDPI Electronics, 2024)
- **Year:** 2024
- **Key Finding:** Bidirectional associative learning system using paired multibit inputs with a synapse-neuron structure based on SNNs. Emulates biological learning for multimodal integration, fault tolerance, and energy efficiency.
- **URL:** https://www.mdpi.com/2079-9292/14/19/3971
- **Remanentia Relevance:** Bidirectional recall aligns with Remanentia's need for both forward (cue->memory) and reverse (memory->context) retrieval.

---

## 2. Memory Consolidation in AI Agents

### 2.1 Mem0: Production-Ready Long-Term Memory

- **Title:** Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory
- **Authors:** Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet Singh, Deshraj Yadav
- **Year:** 2025
- **Key Finding:** Two-phase architecture (Extraction + Update) that dynamically extracts, consolidates, and retrieves salient information. Graph variant (Mem0^g) stores memories as a directed labeled graph with Entity Extractor and Relations Generator. Conflict Detector flags overlapping/contradictory nodes. Achieves 26% improvement over OpenAI on LOCOMO, 91% lower p95 latency, 90%+ token cost savings.
- **URL:** https://arxiv.org/abs/2504.19413
- **Remanentia Relevance:** Mem0's extraction/update pipeline is the closest production analog to our memory consolidation; the graph variant validates our entity-relationship approach.

### 2.2 Letta (MemGPT): Self-Editing Memory Architecture

- **Title:** MemGPT: Towards LLMs as Operating Systems
- **Authors:** Charles Packer, Sarah Wooders, Kevin Lin, Vivian Fang, Shishir G. Patil, Ion Stoica, Joseph E. Gonzalez (UC Berkeley)
- **Year:** 2023 (paper), 2024-2025 (Letta platform evolution)
- **Key Finding:** Introduced self-editing memory where LLMs use tools to edit their own context window. Two-tier hierarchy: in-context memory (core blocks for "Human" and "Persona") and out-of-context memory (archival). Agent uses memory_replace, memory_insert, memory_rethink tools. Parallels human episodic-to-semantic conversion: specific experiences gradually decouple from contextual details through "semantization." Letta V1 (2025) deprecates heartbeat mechanism for modern ReAct-style loops.
- **URL:** https://arxiv.org/abs/2310.08560
- **Remanentia Relevance:** MemGPT's episodic-to-semantic "semantization" is exactly the consolidation mechanism Remanentia needs; self-editing memory is the paradigm we should evaluate against.

### 2.3 Memory in the Age of AI Agents (Survey)

- **Title:** Memory in the Age of AI Agents
- **Authors:** Yuyang Hu, Shichun Liu, Yanwei Yue, Guibin Zhang, Boyang Liu + 42 others
- **Year:** 2025
- **Key Finding:** Comprehensive survey with three-dimensional taxonomy: Forms (token-level, parametric, latent), Functions (factual, experiential, working), and Dynamics (formation, evolution, retrieval). Distinguishes agent memory from LLM memory, RAG, and context engineering. Covers 100+ systems. Frames memory as "a first-class primitive in the design of future agentic intelligence."
- **URL:** https://arxiv.org/abs/2512.13564
- **Remanentia Relevance:** The canonical reference for positioning Remanentia in the agent memory landscape; our SNN approach maps to a novel "latent + experiential" quadrant.

---

## 3. Entity Relationship Graphs for AI Memory

### 3.1 Zep/Graphiti: Temporal Knowledge Graph for Agent Memory

- **Title:** Zep: A Temporal Knowledge Graph Architecture for Agent Memory
- **Authors:** Preston Rasmussen, Pavlo Paliychuk, Travis Beauvais, Jack Ryan, Daniel Chalef
- **Year:** 2025
- **Key Finding:** Core component Graphiti is a temporally-aware KG engine. Ingests raw "Episodes" (message, text, JSON) and constructs entity-relationship graphs. Bitemporal approach: Event Time (T, when fact occurred) + Ingestion Time (T', when observed). Three search functions: cosine similarity, BM25 full-text, breadth-first graph traversal. Outperforms MemGPT on DMR benchmark (94.8% vs 93.4%) with 90% latency reduction.
- **URL:** https://arxiv.org/abs/2501.13956
- **Remanentia Relevance:** Graphiti's bitemporal approach and edge invalidation are directly applicable to Remanentia's temporal memory model; the Episode ingestion pattern matches our session-based architecture.

### 3.2 Microsoft GraphRAG

- **Title:** GraphRAG: Unlocking LLM discovery on narrative private data
- **Authors:** Microsoft Research (Darren Edge et al.)
- **Year:** 2024
- **Key Finding:** Automates extraction of rich knowledge graphs from text using LLMs. Uses Leiden community detection for hierarchical clustering. Local search combines structured KG data with unstructured documents. Global search uses community summaries for holistic reasoning. Open-source, production-grade.
- **URL:** https://microsoft.github.io/graphrag/
- **Remanentia Relevance:** GraphRAG's entity extraction + community detection pipeline could serve as the LLM-side complement to our SNN-side memory encoding.

### 3.3 LLM-Empowered Knowledge Graph Construction (Survey)

- **Title:** LLM-empowered knowledge graph construction: A survey
- **Authors:** (Multiple, 2025)
- **Year:** 2025
- **Key Finding:** Covers ChatIE (extraction as multi-turn dialogue), KGGEN (two-stage: entity detection then relation generation), KARMA (multi-agent schema-guided extraction), and ODKE+ (ontology snippets for context-aware prompts). Shows convergence toward LLM-based extraction replacing traditional NER+RE pipelines.
- **URL:** https://arxiv.org/html/2510.20345v1
- **Remanentia Relevance:** Maps the full landscape of entity extraction approaches we could integrate as the "input encoder" stage of Remanentia's pipeline.

---

## 4. Novelty Detection in Neural Networks

### 4.1 Predictive Coding Model Detects Novelty Hierarchically

- **Title:** Predictive Coding Model Detects Novelty on Different Levels of Representation Hierarchy
- **Authors:** T. Ed Li, Mufeng Tang, Rafal Bogacz
- **Year:** 2025
- **Key Finding:** Predictive coding networks detect novelty at different abstraction levels, from pixel arrangements to object identities. Prediction error neurons produce higher activity for novel stimuli. Unifies novelty detection, associative memory, and representation learning in a single framework. More robust than alternatives on correlated real-world patterns. Open-source code available.
- **URL:** https://direct.mit.edu/neco/article/37/8/1373/131383/
- **Remanentia Relevance:** Directly applicable to Remanentia's novelty scoring: hierarchical prediction errors can drive our "surprise signal" for memory prioritization, replacing ad-hoc heuristics.

### 4.2 Novelty Detection in RL with World Models

- **Title:** Novelty Detection in Reinforcement Learning with World Models
- **Authors:** (OpenReview, 2025)
- **Year:** 2025
- **Key Finding:** Uses misalignment between world model hallucinations and true observed states as a novelty score. When the world model's predictions diverge significantly from reality, the agent flags the experience as novel, triggering enhanced learning and exploration.
- **URL:** https://openreview.net/forum?id=xtlixzbcfV
- **Remanentia Relevance:** The "prediction-reality mismatch as novelty" paradigm maps directly to our SNN's ability to detect when new information doesn't match stored patterns.

### 4.3 Von Restorff Effect and Distinctiveness in Memory

- **Title:** What explains the von Restorff effect? Contrasting distinctive processing and retrieval cue efficacy
- **Authors:** (Multiple studies, compiled)
- **Year:** Various (theory ongoing through 2025)
- **Key Finding:** Two competing explanations: (1) distinctive processing during encoding, and (2) retrieval cue efficacy. Surprise-based explanations (Green 1956) are partially supported but insufficient alone -- items isolated early also show the effect before context is established. PFC activation during encoding of distinctive items signals surprise and enhances salience. Computational models of free recall include PFC mechanisms sensitive to both commonality and distinctiveness.
- **URL:** https://www.sciencedirect.com/science/article/abs/pii/S0749596X17300864
- **Remanentia Relevance:** Our SNN novelty detection should implement both distinctiveness-at-encoding AND retrieval-cue-efficacy mechanisms, not just surprise alone.

---

## 5. Modern Hopfield Networks and Attention-as-Memory

### 5.1 Hopfield Networks is All You Need (Foundation)

- **Title:** Hopfield Networks is All You Need
- **Authors:** Hubert Ramsauer, Bernhard Schafl, Johannes Lehner, Philipp Seidl, Michael Widrich, Thomas Adler, Lukas Gruber, Markus Holzleitner, Milena Pavlovic, Geir Kjetil Sandve, Victor Greiff, David Kreil, Michael Kopp, Gunter Klambauer, Johannes Brandstetter, Sepp Hochreiter
- **Year:** 2021
- **Key Finding:** Modern Hopfield networks with continuous states store exponentially many patterns and retrieve them with one update step. The update rule is equivalent to the attention mechanism in transformers. Exponential capacity (in dimension of associative space) with exponentially small retrieval errors. Establishes the formal bridge between Hopfield networks and transformer attention heads.
- **URL:** https://arxiv.org/abs/2008.02217
- **Remanentia Relevance:** Foundational theory connecting associative memory to attention; Remanentia could use Hopfield layers as a differentiable memory module that complements SNN-based storage.

### 5.2 Modern Hopfield Networks with Continuous-Time Memories

- **Title:** Modern Hopfield Networks with Continuous-Time Memories
- **Authors:** Saul Santos, Antonio Farinhas, Daniel C. McNamee, Andre F.T. Martins
- **Year:** 2025
- **Key Finding:** Replaces discrete memory slots with continuous-time probability density functions, inspired by psychological theories of continuous neural resource allocation. Modifies the Hopfield energy function to use probability density over continuous memory instead of softmax PMF. Achieves competitive retrieval performance with smaller memory footprint. Validated on synthetic and video datasets.
- **URL:** https://arxiv.org/abs/2502.10122
- **Remanentia Relevance:** Continuous-time memory compression directly relevant to Remanentia's need to consolidate many discrete experiences into compact, retrievable representations.

### 5.3 Provably Optimal Memory Capacity (Spherical Codes)

- **Title:** Provably Optimal Memory Capacity for Modern Hopfield Models: Transformer-Compatible Dense Associative Memories as Spherical Codes
- **Authors:** Jerry Yao-Chieh Hu, Dennis Wu, Han Liu
- **Year:** 2024
- **Key Finding:** First tight and optimal asymptotic memory capacity bound for modern Hopfield models. Connects memory configurations to spherical codes (optimal point arrangements on hyperspheres). Optimal capacity occurs when feature space allows memories to form an optimal spherical code. Provides U-Hop+ sub-linear time algorithm achieving optimal capacity.
- **URL:** https://arxiv.org/abs/2410.23126
- **Remanentia Relevance:** Theoretical ceiling on how much our Hopfield-based memory layers can store; spherical code insight could guide our embedding space design.

### 5.4 Dynamic Manifold Hopfield Networks

- **Title:** Dynamic Manifold Hopfield Networks for Context-Dependent Associative Memory
- **Authors:** (arXiv, 2025)
- **Year:** 2025
- **Key Finding:** Continuous dynamical models where contextual modulation reshapes attractor geometry dynamically. When storing 2N patterns in N neurons, DMHN achieves 64% reliable retrieval vs 1% (classical) and 13% (modern Hopfield). Enables context-dependent memory retrieval where the same network retrieves different patterns based on context.
- **URL:** https://arxiv.org/abs/2506.01303
- **Remanentia Relevance:** Context-dependent retrieval is exactly what Remanentia needs: the same memory should be accessible differently depending on the current conversational context.

### 5.5 Sparse Quantized Hopfield Network for Online-Continual Memory

- **Title:** A Sparse Quantized Hopfield Network for Online-Continual Memory
- **Authors:** (Nature Communications, 2024)
- **Year:** 2024
- **Key Finding:** Energy-based model using online MAP learning with neurogenesis and local learning rules. Outperforms SOTA on associative memory tasks in online, continual, and noisy settings. Better than baselines on episodic memory tasks. Bridges the gap between biological (online, noisy, non-i.i.d.) and artificial (offline, clean, i.i.d.) learning paradigms.
- **URL:** https://www.nature.com/articles/s41467-024-46976-4
- **Remanentia Relevance:** The online-continual learning paradigm with neurogenesis is the closest existing work to what Remanentia aims to achieve: adding new memories without forgetting old ones, in real-time.

---

## 6. Hippocampal Replay and Memory Consolidation

### 6.1 Generative Model of Memory Construction and Consolidation

- **Title:** A generative model of memory construction and consolidation
- **Authors:** Eleanor Spens, Neil Burgess
- **Year:** 2024
- **Key Finding:** Hippocampal replay from an autoassociative network trains generative models (VAEs) to recreate sensory experiences from latent variable representations. Explains semantic memory, imagination, episodic future thinking, relational inference, and schema-based distortions. Unique sensory elements and predictable conceptual elements are stored/reconstructed by efficiently combining hippocampal and neocortical systems, optimizing limited hippocampal storage for new and unusual information.
- **URL:** https://www.nature.com/articles/s41562-023-01799-z
- **Remanentia Relevance:** The hippocampus-as-autoassociative-network training neocortex-as-VAE paradigm is a biological blueprint for Remanentia's two-stage consolidation: SNN (hippocampus) training compressed representations (neocortex).

### 6.2 Memory Consolidation from a Reinforcement Learning Perspective

- **Title:** Memory consolidation from a reinforcement learning perspective
- **Authors:** (Frontiers in Computational Neuroscience, 2024)
- **Year:** 2024
- **Key Finding:** CA3 generates diverse activity patterns while CA1 selectively reinforces high-value ones. Reward enhances rate and fidelity of awake replays, facilitating consolidation. Priority-based replay: memories most likely to be relevant in the future are replayed first and most often during sleep. Frames memory consolidation as offline reinforcement learning.
- **URL:** https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2024.1538741/full
- **Remanentia Relevance:** Value-driven priority replay is the biological basis for our novelty-weighted consolidation: high-surprise memories should get more replay cycles.

### 6.3 HiCL: Hippocampal-Inspired Continual Learning

- **Title:** HiCL: Hippocampal-Inspired Continual Learning
- **Authors:** Kushal Kapoor, Wyatt Mackey, Yiannis Aloimonos, Xiaomin Lin
- **Year:** 2025
- **Key Finding:** Dual-memory architecture with: grid-cell-like encoding, dentate gyrus pattern separation (top-k sparsity), CA3 autoassociative episodic memory, DG-gated mixture-of-experts routing. Consolidation via Elastic Weight Consolidation weighted by inter-task similarity + prioritized replay. Near SOTA on continual learning benchmarks at lower computational cost.
- **URL:** https://arxiv.org/abs/2508.16651
- **Remanentia Relevance:** HiCL is the most complete hippocampal architecture in ML; its DG pattern separation + CA3 autoassociation + EWC consolidation provides a direct reference implementation for Remanentia's memory hierarchy.

### 6.4 Hybrid Neural Networks for Corticohippocampal Continual Learning

- **Title:** Hybrid neural networks for continual learning inspired by corticohippocampal circuits
- **Authors:** (Nature Communications, 2025)
- **Year:** 2025
- **Key Finding:** Bidirectional interaction between memory systems during offline periods. Cortical reactivation triggers time-compressed sequential replays in hippocampus, which drive cortical consolidation. Demonstrates the mutual dependency between fast (hippocampal) and slow (cortical) learning systems.
- **URL:** https://www.nature.com/articles/s41467-025-56405-9
- **Remanentia Relevance:** Validates our two-speed architecture where the SNN (fast hippocampal) and embedding store (slow cortical) interact bidirectionally during consolidation.

---

## 7. Paragraph-Level Retrieval

### 7.1 ColBERTv2: Lightweight Late Interaction

- **Title:** ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction
- **Authors:** Keshav Santhanam, Omar Khattab, Jon Saad-Falcon, Christopher Potts, Matei Zaharia
- **Year:** 2022 (NAACL)
- **Key Finding:** Multi-vector representations at token granularity with late interaction scoring (MaxSim). Residual compression reduces vector size from 256 to 36 bytes (2-bit) while maintaining accuracy. Denoised supervision via teacher distillation + hard negative mining. 6-10x storage reduction over ColBERTv1. PLAID acceleration achieves 45x lower latency.
- **URL:** https://arxiv.org/abs/2112.01488
- **Remanentia Relevance:** ColBERTv2's token-level multi-vector approach is the retrieval backbone our paragraph-level finding (85.7% vs 50.0%) validates; late interaction could replace dense single-vector retrieval in Remanentia.

### 7.2 Dense X Retrieval: Proposition-Level Granularity

- **Title:** Dense X Retrieval: What Retrieval Granularity Should We Use?
- **Authors:** Tong Chen, Hongwei Wang, Sihao Chen, Wenhao Yu, Kaixin Ma, Xinran Zhao, Hongming Zhang, Dong Yu
- **Year:** 2024 (EMNLP)
- **Key Finding:** Introduces "propositions" as retrieval unit: atomic expressions each encapsulating a distinct factoid. Proposition-level indexing outperforms passage-level by +10.1 Recall@20 (unsupervised) and +2.7 (supervised). With 500-token budget, propositions yield +2.7 to +4.1 EM point gains. Fine-grained units enable precise retrieval while maintaining context.
- **URL:** https://arxiv.org/html/2312.06648
- **Remanentia Relevance:** Our "best paragraph" finding at 85.7% aligns with this literature -- proposition-level granularity could push accuracy even further. Remanentia should index at proposition level, not passage level.

### 7.3 ColPali/ColQwen: Multimodal Late Interaction

- **Title:** ColPali: Efficient Document Retrieval with Vision Language Models
- **Authors:** Manuel Faysse, Hugues Sibille, Tony Wu, Bilel Omrani, Gautier Viaud, Celine Hudelot, Pierre Colombo
- **Year:** 2024
- **Key Finding:** Extends late interaction to vision: treats entire PDFs as images and performs token-level similarity between text queries and image patches. ColQwen2 (using Qwen2-VL) improves +5.3 nDCG@5 over ColPali. Eliminates need for OCR and complex chunking pipelines.
- **URL:** https://arxiv.org/abs/2407.01449
- **Remanentia Relevance:** Demonstrates late interaction generalizes beyond text; relevant if Remanentia ever needs to handle multimodal memory (images, diagrams, code screenshots).

### 7.4 Jina-ColBERT-v2: Multilingual Late Interaction

- **Title:** Jina-ColBERT-v2: A General-Purpose Multilingual Late Interaction Retriever
- **Authors:** (Jina AI, 2024)
- **Year:** 2024
- **Key Finding:** Extends ColBERT to 89 languages with flexible output dimensions. Strong cross-lingual retrieval performance. Production-ready multilingual late interaction model.
- **URL:** https://arxiv.org/abs/2408.16672
- **Remanentia Relevance:** If Remanentia serves multilingual users, Jina-ColBERT-v2 provides the retrieval backbone without per-language fine-tuning.

---

## 8. Kuramoto Oscillators for Computation

### 8.1 Artificial Kuramoto Oscillatory Neurons (AKOrN)

- **Title:** Artificial Kuramoto Oscillatory Neurons
- **Authors:** Takeru Miyato, Sindy Lowe, Andreas Geiger, Max Welling
- **Year:** 2025 (ICLR Oral)
- **Key Finding:** Replaces threshold units with Kuramoto oscillators as a dynamical alternative. Binding through synchronization: generalized Kuramoto updates align/anti-align connected oscillators, performing distributed continuous clustering. Solves unsupervised object discovery, adversarial robustness, uncertainty quantification, and Sudoku reasoning. Can be combined with fully connected, convolutional, or attention mechanisms.
- **URL:** https://arxiv.org/abs/2410.13821
- **Remanentia Relevance:** AKOrN validates that Kuramoto oscillators work as neural network building blocks beyond pure synchronization. Our Kuramoto layer could adopt this architecture for binding related memories through phase coherence.

### 8.2 Computing with Oscillators: Theory to Applications

- **Title:** Computing with oscillators from theoretical underpinnings to applications and demonstrators
- **Authors:** (npj Unconventional Computing, 2024)
- **Year:** 2024
- **Key Finding:** Comprehensive review of oscillatory computing paradigms. ONNs encode information in phase differences of coupled oscillators. Applications: combinatorial optimization (Ising machines), classification, pattern recognition. Phase transition functions for the Kuramoto model can be simulated in software. Hardware demonstrations include CMOS ring oscillator circuits (1,968 nodes).
- **URL:** https://www.nature.com/articles/s44335-024-00015-z
- **Remanentia Relevance:** Establishes the theoretical foundation for phase-based memory encoding in Remanentia; validates our Kuramoto synchronization detection as a legitimate computational primitive.

### 8.3 Oscillatory Ising Machines and Combinatorial Optimization

- **Title:** Oscillatory Neural Network-Based Ising Machine Using 2D Memristors
- **Authors:** (ACS Nano, 2024)
- **Year:** 2024
- **Key Finding:** Kuramoto oscillators solve combinatorial optimization (Max-Cut, k-SAT) by mapping to continuous Ising models. 2D memristors enable in-memory computing with scalability. Recent work extends to inverse matrix computation via linearized Kuramoto models. Ultralow power consumption with rapid computational performance.
- **URL:** https://pubs.acs.org/doi/10.1021/acsnano.3c10559
- **Remanentia Relevance:** The Ising machine connection means Kuramoto networks can solve optimization problems -- relevant for finding optimal memory retrieval paths or resolving conflicting memories.

### 8.4 Multifrequency Oscillatory Neural Networks

- **Title:** Harnessing Phase Dynamics Across Diverse Frequencies with Multifrequency Oscillatory Neural Networks
- **Authors:** Dinc et al.
- **Year:** 2025
- **Key Finding:** Addresses practical challenge of frequency mismatches in hardware. Introduces multifrequency ONN computing approach validated in both hardware and software simulations. Enables oscillatory computing to be robust against device variability.
- **URL:** https://advanced.onlinelibrary.wiley.com/doi/abs/10.1002/aidi.202500152
- **Remanentia Relevance:** If Remanentia deploys on heterogeneous hardware, multifrequency robustness is essential for reliable oscillatory memory.

---

## 9. Contrastive Learning for Spiking Networks

### 9.1 Contrastive Signal-Dependent Plasticity (CSDP)

- **Title:** Contrastive signal-dependent plasticity: Self-supervised learning in spiking neural circuits
- **Authors:** Alexander G. Ororbia
- **Year:** 2024 (Science Advances)
- **Key Finding:** Generalizes self-supervised contrastive learning to spiking circuits. Integrates "goodness principle" from forward-forward learning into spiking neuron dynamics: raise probability for real sensory input ("positive"), lower for fake/OOD input ("negative"). Side-steps need for feedback synapses (unlike backprop). Both unsupervised and supervised variants produce generative and discriminative capabilities. Viable for memristive hardware: requires only pre-synaptic spike and post-synaptic spike/trace information.
- **URL:** https://www.science.org/doi/10.1126/sciadv.adn6076
- **Remanentia Relevance:** CSDP is the strongest candidate to replace STDP in Remanentia's SNN: it produces discriminative features without backpropagation, is biologically plausible, and is hardware-friendly.

### 9.2 Stabilized Supervised STDP (S2-STDP) with Paired Competing Neurons

- **Title:** Paired competing neurons improving STDP supervised local learning in spiking neural networks
- **Authors:** Goupy, Tirilly, Bilasco
- **Year:** 2024 (Frontiers in Neuroscience)
- **Key Finding:** S2-STDP integrates error-modulated weight updates aligning spikes with desired timestamps. Paired Competing Neurons (PCN) architecture: each class gets paired neurons that specialize toward target/non-target via intra-class competition. Outperforms SOTA supervised STDP rules on MNIST, Fashion-MNIST, CIFAR-10 without additional hyperparameters.
- **URL:** https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1401690/full
- **Remanentia Relevance:** PCN's intra-class competition could improve Remanentia's memory discrimination: paired neurons for "this is a relevant memory" vs "this is not" classification.

### 9.3 Temporal Contrastive Learning for SNNs

- **Title:** Temporal Contrastive Learning for Spiking Neural Networks
- **Authors:** (arXiv, 2023-2024)
- **Year:** 2024
- **Key Finding:** TCL framework obtains low-latency, high-performance SNNs by incorporating contrastive supervision with temporal domain information. Bridges the gap between energy-efficient SNN execution and discriminative feature learning.
- **URL:** https://arxiv.org/abs/2305.13909
- **Remanentia Relevance:** Temporal contrastive learning could enable Remanentia's SNN to learn time-aware memory representations that distinguish recent from old information.

---

## 10. Production Memory Systems Benchmarks

### 10.1 LongMemEval

- **Title:** LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory
- **Authors:** Di Wu, Hongwei Wang, Wenhao Yu, Yuwei Zhang, Kai-Wei Chang, Dong Yu
- **Year:** 2025 (ICLR)
- **Key Finding:** 500 curated questions testing 5 abilities: information extraction, multi-session reasoning, temporal reasoning, knowledge updates, abstention. Two configs: LongMemEval_S (~115k tokens) and LongMemEval_M (~1.5M tokens, 500 sessions). Commercial assistants show 30% accuracy drop; GPT-4o achieves only 30-70%. Proposes session decomposition, fact-augmented key expansion, time-aware query expansion as optimizations.
- **URL:** https://arxiv.org/abs/2410.10813
- **Remanentia Relevance:** THE benchmark for evaluating Remanentia's memory system. Our target: beat GPT-4o's 30-70% on all five abilities, especially temporal reasoning and knowledge updates.

### 10.2 LOCOMO: Long-Term Conversational Memory

- **Title:** Evaluating Very Long-Term Conversational Memory of LLM Agents
- **Authors:** Maharana et al.
- **Year:** 2024
- **Key Finding:** Dataset of very long conversations: 300 turns, 9K tokens avg, up to 35 sessions. Tests question answering, event summarization, multi-modal dialogue generation. Up to 32 sessions with ~600 turns (~16K tokens) incorporating images. Standard benchmark used by Mem0, Zep, and other memory systems.
- **URL:** https://snap-research.github.io/locomo/
- **Remanentia Relevance:** LOCOMO is the easier benchmark (vs LongMemEval) where we should first demonstrate competitiveness; Mem0 already achieves strong results here.

### 10.3 MEMTRACK: Multi-Platform Dynamic Environments

- **Title:** MEMTRACK: Evaluating Long-Term Memory and State Tracking in Multi-Platform Dynamic Agent Environments
- **Authors:** Darshan Deshpande, Varun Gangal, Hersh Mehta, Anand Kannappan, Rebecca Qian, Peng Wang
- **Year:** 2025
- **Key Finding:** Tests memory across Slack, Linear, and Git platforms with chronologically interleaved timelines, conflicting information, and cross-references. Metrics: Correctness, Efficiency, Redundancy. GPT-5 achieves only 60% Correctness. Existing memory systems (Zep, Mem0) do not significantly improve performance, highlighting fundamental limitations.
- **URL:** https://arxiv.org/abs/2510.01353
- **Remanentia Relevance:** MEMTRACK exposes the failure of current memory systems in enterprise settings -- the exact gap Remanentia aims to fill with biologically-grounded consolidation.

### 10.4 MemoryAgentBench: Incremental Multi-Turn Evaluation

- **Title:** Evaluating Memory in LLM Agents via Incremental Multi-Turn Interactions
- **Authors:** (ICLR 2026)
- **Year:** 2025-2026
- **Key Finding:** Four core competencies: accurate retrieval, test-time learning, long-range understanding, selective forgetting. Transforms long-context datasets into multi-turn format simulating incremental information processing. Based on cognitive science memory theories.
- **URL:** https://arxiv.org/abs/2507.05257
- **Remanentia Relevance:** The "selective forgetting" competency aligns with Remanentia's decay mechanism; this benchmark tests exactly the cognitive capabilities we claim to implement.

### 10.5 5-System Benchmark Comparison (2026)

- **Title:** 5 AI Agent Memory Systems Compared: Mem0, Zep, Letta, Supermemory, SuperLocalMemory
- **Authors:** (DEV Community comparison, 2026)
- **Year:** 2026
- **Key Finding:** Comparative benchmark across production systems. Mem0 leads on graph-based relational memory. Zep excels on temporal reasoning. Letta strongest on self-editing and context management. No system dominates all categories. All struggle with knowledge update conflicts.
- **URL:** https://dev.to/varun_pratapbhardwaj_b13/5-ai-agent-memory-systems-compared-mem0-zep-letta-supermemory-superlocalmemory-2026-benchmark-59p3
- **Remanentia Relevance:** Competitive landscape map showing the gaps Remanentia can exploit: no existing system combines SNN-based consolidation with graph memory.

---

## Cross-Cutting Synthesis for Remanentia

### Architecture Implications

| Component | Best Reference | Key Insight |
|-----------|---------------|-------------|
| SNN Memory Layer | Casanueva-Morato 2024, BCPNN 2024 | Spike-based CAM on neuromorphic hardware is proven; BCPNN offers alternative to pure STDP |
| Learning Rule | CSDP (Ororbia 2024) | Replace STDP with contrastive signal-dependent plasticity for discriminative features |
| Novelty Detection | Li/Tang/Bogacz 2025 | Hierarchical predictive coding errors as surprise signal; unifies with associative memory |
| Memory Consolidation | Spens & Burgess 2024, HiCL 2025 | Hippocampal autoassociation -> VAE-like generative consolidation; DG pattern separation + CA3 |
| Entity Graphs | Zep/Graphiti 2025 | Bitemporal KG with episode ingestion; edge invalidation for knowledge updates |
| Retrieval | ColBERTv2, Dense X Retrieval 2024 | Proposition-level granularity (+10.1 Recall@20); multi-vector late interaction |
| Oscillator Layer | AKOrN (ICLR 2025) | Kuramoto oscillators as neural building blocks; binding through synchronization |
| Associative Memory | Modern Hopfield + SQHN 2024 | Exponential capacity, continuous-time compression, online-continual learning |
| Evaluation | LongMemEval, MEMTRACK | 5 abilities benchmark (LongMemEval); enterprise multi-platform (MEMTRACK) |

### Key Gaps in Literature (Remanentia's Opportunity)

1. **No system combines SNN-based memory encoding with LLM-based entity extraction** -- Remanentia bridges biological and artificial approaches
2. **Existing benchmarks show 30-60% accuracy for SOTA** -- massive room for improvement
3. **Episodic-to-semantic conversion is hand-waved in production systems** -- Remanentia can implement it biologically via hippocampal replay
4. **Kuramoto oscillators have never been used for memory binding in agent systems** -- novel contribution
5. **Novelty detection is not integrated into any production memory system** -- Remanentia's surprise-weighted consolidation is unique

---

# ROUND 3: PRODUCT LANDSCAPE & COMPETITIVE ANALYSIS

## Compiled: 2026-03-20 | Deep product research for Remanentia positioning

---

## 11. Mem0 — Production Architecture & Product Analysis

### 11.1 Technical Architecture

- **Dual-Storage Design:** Vector store (20+ backends: Qdrant, ChromaDB, Milvus, Pinecone, PGVector) for semantic search + SQLite for version tracking and audit trails
- **Self-Hosted Stack:** Three Docker containers — FastAPI REST API, PostgreSQL with pgvector for embeddings, Neo4j for entity relationships
- **API Surface:** Simple two-method core: `memory.add(messages, user_id)` and `memory.search(query, user_id, limit)`. Graph variant (Mem0^g) adds entity extraction and relation generation
- **Memory Pipeline:** Two-phase (Extraction + Update) — LLM extracts salient facts from conversations, deduplicates against existing memories, resolves conflicts
- **Default LLM:** GPT-4.1-nano-2025-04-14 for extraction
- **SDKs:** Python (`pip install mem0ai`), TypeScript (`npm install mem0ai`)

### 11.2 Product & Pricing

| Tier | Price | Memories | Retrieval Calls | Key Features |
|------|-------|----------|-----------------|--------------|
| Hobby | Free | 10K | 1K/month | Basic vector search |
| Starter | $19/month | 50K | — | Standard features |
| Pro | $249/month | Unlimited | Unlimited | Graph memory, analytics |
| Enterprise | Custom | Custom | Custom | On-prem, SSO, SLA, BYOK |

- **Funding:** YC-backed, $24M Series A (Oct 2025) from Peak XV and Basis Set Ventures
- **GitHub:** 50.4K stars, 5.6K forks, 1,957 commits, 251 open issues
- **Compliance:** SOC 2, HIPAA, BYOK

### 11.3 User Complaints & Weaknesses

1. **Latency:** "Super bad" latency reported in production; imports dragging research flow
2. **Indexing reliability:** Memory additions silently fail; "every time it was indexing something, it wouldn't add it to the memories"
3. **Scaling degradation:** Performance drops as users add more data
4. **Context extraction accuracy:** Misinterprets nuanced preferences (e.g., "I hate mushrooms" applied too broadly)
5. **Self-hosting:** "You're mostly on your own" — the hosted platform is the real product
6. **Pricing cliff:** $19 -> $249 jump for graph features is steep for teams that want to test graph retrieval before committing
7. **Free tier:** 10K memories is barely enough to prototype

### 11.4 Remanentia Differentiation

- Mem0 uses LLM-based extraction (expensive, latency-bound); Remanentia uses SNN-based consolidation (local, constant-time, no LLM calls for memory operations)
- Mem0's memory is flat key-value with optional graph overlay; Remanentia has biologically-grounded episodic-to-semantic conversion
- Mem0 has no novelty detection or surprise-weighted consolidation
- Mem0 requires external vector DB + graph DB; Remanentia runs from JSONL files with optional embedding upgrade

---

## 12. Letta/MemGPT — Production Architecture & Product Analysis

### 12.1 Technical Architecture

- **Self-Editing Memory:** Agents use tools (memory_replace, memory_insert, memory_rethink) to edit their own context window. The LLM decides what to remember and what to forget
- **Memory Hierarchy:** Core Memory (always in-context: Human block + Persona block) + Archival Memory (out-of-context, searchable) + Recall Memory (conversation history)
- **Persistence Layer:** SQLite (development default, ~/.letta/letta.db) or PostgreSQL (production). Aurora PostgreSQL for enterprise
- **Multi-User:** One agent per user pattern. Each agent maintains isolated persistent memory. Horizontal scaling via Kubernetes
- **Deployment:** REST API server. Self-hosted or Letta Cloud. AWS Marketplace AMI available
- **Conversations API (Jan 2026):** Shared memory across parallel user interactions

### 12.2 Product & Pricing

| Tier | Price | Credits | Agents | Storage | Key Features |
|------|-------|---------|--------|---------|--------------|
| Free | $0 | Limited | Limited | — | Rotating free models |
| Pro | ~$20/month | 5,000 | Unlimited | 1 GB | 2 agent templates |
| Max | Higher | 20,000 | Unlimited | 10 GB | Pay-as-you-go overage, 20 templates |
| Enterprise | Custom | Custom | Custom | Custom | RBAC, SSO, private models, dedicated support |

- **Funding:** $10M seed (Sep 2024), UC Berkeley origin
- **GitHub:** Major open-source project (letta-ai/letta)
- **DeepLearning.AI:** Official course on Letta with Andrew Ng's platform

### 12.3 Strengths & Weaknesses

**Strengths:**
- Self-editing memory is the most advanced paradigm — the agent manages its own memory
- Strong academic foundation (MemGPT paper, UC Berkeley)
- Production-grade with Kubernetes scaling, Aurora PostgreSQL
- Open source with commercial cloud overlay

**Weaknesses:**
- Self-editing memory depends entirely on LLM quality — hallucinations corrupt memory
- Every memory operation requires an LLM call (expensive at scale)
- Memory consolidation is implicit (agent decides) rather than systematic
- No biological grounding or novelty detection
- Cloud pricing details still murky

### 12.4 Remanentia Differentiation

- Letta's memory is LLM-managed (expensive, hallucination-prone); Remanentia's is SNN-managed (deterministic, local)
- Letta has no temporal knowledge graph; Remanentia has bitemporal entity relationships
- Letta's "self-editing" means the agent rewrites its own context — powerful but unreliable. Remanentia consolidates through biologically-grounded mechanisms
- Letta targets agent builders; Remanentia can serve as a memory backend for any agent framework

---

## 13. Zep — Production Architecture & Product Analysis

### 13.1 Technical Architecture (Graphiti)

- **Temporal Knowledge Graph:** Three-tier subgraph hierarchy — Episode subgraph, Semantic Entity subgraph, Community subgraph
- **Bitemporal Model:** Every edge tracks Event Time (when fact occurred) and Ingestion Time (when observed). Edges have explicit validity intervals (t_valid, t_invalid)
- **Entity Extraction:** Uses last N messages for context during NER. Predefined Cypher queries (not LLM-generated) for schema consistency
- **Entity Resolution:** Embedding-based deduplication. Hybrid search (cosine similarity + BM25 full-text) constrained to same entity pairs
- **Custom Ontologies:** Pydantic models for custom entity types and edge types with attribute extraction from text
- **Edge Invalidation:** Dynamic information updates through temporal extraction — old facts are invalidated, not deleted
- **Storage:** Neo4j graph database backend
- **Open Source:** Graphiti library is open source (github.com/getzep/graphiti)

### 13.2 Product & Pricing

| Component | Detail |
|-----------|--------|
| Credit Cost | 1 credit per Episode (chat message, JSON, or text block) |
| Episode Billing | Episodes >350 bytes billed in multiples (640 bytes = 2 credits) |
| Free Tier | 1,000 credits/month, lower priority, rate limited |
| Flex | $25/month for teams needing graph features |
| Enterprise | Custom, managed/BYOK/BYOM/BYOC options |
| Auto-Replenish | When balance drops below 20%, auto-adds 20,000 credits |
| Rollover | Unused credits roll over for 60 days (2 billing cycles) |
| Compliance | SOC 2 Type II, HIPAA BAA on Enterprise |

### 13.3 Strengths & Weaknesses

**Strengths:**
- Best-in-class temporal reasoning — bitemporal model is architecturally superior
- Graphiti is open source and independently usable
- Strong academic foundation (arxiv paper with formal evaluation)
- Knowledge graph MCP server available
- Episode-based ingestion matches real-world data flow

**Weaknesses:**
- Requires Neo4j (heavy infrastructure dependency)
- LLM-dependent extraction pipeline (cost scales with volume)
- Free tier (1,000 credits) is barely enough to evaluate
- Community/open-source version has limited features vs cloud
- No memory consolidation beyond graph updates — no decay, no surprise weighting

### 13.4 Remanentia Differentiation

- Zep's temporal model is pure-software; Remanentia's temporal encoding uses SNN spike timing (biologically motivated)
- Zep requires Neo4j; Remanentia runs from flat files with optional upgrade path
- Zep has no consolidation mechanism — old information is invalidated but not synthesized. Remanentia performs hippocampal-style episodic-to-semantic conversion
- Zep's Graphiti could serve as a complementary component to Remanentia's SNN backend

---

## 14. LangMem/LangGraph — Product Analysis

### 14.1 Technical Architecture

- **LangMem SDK:** Library (not a service) providing memory management tools for LangGraph agents
- **Core Tools:** `create_manage_memory_tool` (store/update) and `create_search_memory_tool` (retrieve)
- **Two Operational Modes:**
  - Hot Path: Agent decides what/when to store during real-time interactions
  - Background Processing: Automatic extraction, consolidation, and update asynchronously
- **Storage Backends:** InMemoryStore (development) or AsyncPostgresStore (production). Works with any BaseStore implementation
- **LangGraph Integration:** Native integration with LangGraph's Long-term Memory Store. Available by default in all LangGraph Platform deployments
- **Deployment:** `langgraph deploy` builds Docker image + auto-provisions Postgres + Redis (as of March 2026)
- **Setup:** ~15 lines of code for functional agent with memory

### 14.2 Product Model

- **Not a standalone service** — a library within the LangChain ecosystem
- **No independent pricing** — included in LangGraph Platform (LangSmith pricing)
- **Vendor lock-in:** Deeply tied to LangGraph. Using LangMem outside LangGraph requires implementing BaseStore yourself
- **Open source:** langchain-ai/langmem on GitHub

### 14.3 Strengths & Weaknesses

**Strengths:**
- Ecosystem integration — if you're already on LangGraph, memory is built in
- Background processing mode is unique — memory extraction without agent intervention
- Simple API surface (~15 lines to get started)
- MongoDB, Postgres, Redis backends all supported

**Weaknesses:**
- Not a standalone product — cannot use without LangGraph
- No knowledge graph — pure vector/key-value storage
- No temporal reasoning or bitemporal tracking
- No consolidation beyond simple update/overwrite
- Memory quality depends on LLM extraction (same limitation as Mem0)

### 14.4 Remanentia Differentiation

- LangMem is a library; Remanentia is a standalone system with API, MCP, and CLI
- LangMem has no entity graph, no temporal model, no consolidation engine
- LangMem is LangGraph-only; Remanentia is framework-agnostic
- Remanentia can serve as a LangGraph-compatible BaseStore implementation (potential integration path)

---

## 15. Emerging Competitors

### 15.1 Hindsight (by Vectorize)

- **Architecture:** Three core operations — retain, recall, reflect. Multi-strategy retrieval with cross-encoder reranking
- **MCP-First:** Designed as native MCP server, no custom glue code
- **Tools:** retain, recall, reflect, getMentalModel, getDocument
- **Benchmark:** 91% on LongMemEval (state-of-the-art as of March 2026)
- **Funding:** $3.5M (April 2024)
- **Differentiation from Mem0:** Purpose-built for institutional knowledge, not just chat personalization
- **Open source:** github.com/vectorize-io/hindsight

### 15.2 Supermemory

- **Architecture:** User-profile-centric — automatically infers, updates, retrieves user context from interactions
- **Multi-source:** Tweets, web pages, documents imported as memory sources
- **Ranking:** #1 on LongMemEval, LoCoMo, and ConvoMem benchmarks
- **API Surface:** Three lines to add memory (TypeScript, Python, REST)
- **Model:** Cloud-hosted, closed source. Generous free tier
- **Trade-off:** Simplicity over control. No custom entity types, no graph access

### 15.3 Mengram

- **Architecture:** Three memory types — semantic (facts), episodic (events), procedural (learned workflows)
- **Auto-extraction:** One API call extracts all three types automatically
- **Procedural memory:** Agent completes task -> Mengram saves steps -> next time it knows the optimal path with success/failure tracking
- **Differentiation:** Only product with explicit procedural memory support

### 15.4 SuperLocalMemory V2

- **Architecture:** Local-first AI memory across 17+ AI tools via MCP
- **Features:** Knowledge graphs, hybrid search, pattern learning
- **Zero cloud dependency** — fully local
- **Trade-off:** Limited scale, manual setup

---

## 16. Claude Code Memory System — Architecture & Enhancement Opportunity

### 16.1 Architecture (as of v2.1.59+)

**Two complementary systems:**

1. **CLAUDE.md Files (Manual Memory)**
   - Markdown files with persistent instructions
   - Three-level hierarchy: Managed Policy > Project > User
   - Project CLAUDE.md: `./CLAUDE.md` or `./.claude/CLAUDE.md`
   - User CLAUDE.md: `~/.claude/CLAUDE.md`
   - Managed: `/Library/Application Support/ClaudeCode/CLAUDE.md` (macOS) / `C:\Program Files\ClaudeCode\CLAUDE.md` (Windows)
   - Supports `@path/to/import` syntax for file imports (max 5 hops)
   - Path-specific rules via `.claude/rules/` with YAML frontmatter globs
   - Optimal at <200 lines (92% rule adherence vs 71% at 400+ lines)

2. **Auto Memory (MEMORY.md)**
   - Location: `~/.claude/projects/<project>/memory/`
   - Derived from git repository path (shared across worktrees)
   - MEMORY.md entrypoint + topic files (debugging.md, api-conventions.md, etc.)
   - First 200 lines loaded at session start; topic files loaded on demand
   - Machine-local, not shared across environments
   - Claude decides what to save based on future-conversation utility

**Key Characteristics:**
- Context-based, not enforced — Claude reads instructions as a user message, not system prompt
- Survives `/compact` — re-reads from disk after compaction
- No semantic search — pure file-based, loaded into context window
- No consolidation — MEMORY.md grows until Claude manually reorganizes it
- No entity graph — flat markdown structure
- No decay or forgetting — everything persists forever unless manually deleted

### 16.2 How Remanentia Enhances Claude Code Memory

| Claude Code Limitation | Remanentia Enhancement |
|----------------------|----------------------|
| Flat markdown, no structure | Entity graph with typed relationships |
| No semantic search | TF-IDF + optional embedding-based retrieval |
| 200-line MEMORY.md cap | Unlimited memory with SNN-orchestrated recall |
| No consolidation | Automatic episodic-to-semantic conversion |
| No temporal awareness | Bitemporal tracking (event time + ingestion time) |
| No novelty detection | Surprise-weighted consolidation prioritizes novel information |
| No cross-session reasoning | Entity graph connects knowledge across sessions |
| Machine-local only | API server enables distributed access |

**Integration Path:** Remanentia's MCP server (`remanentia_recall`, `remanentia_status`, `remanentia_graph`) plugs directly into Claude Code's `.mcp.json` configuration, running alongside Claude's native memory as a deep contextual backend.

---

## 17. MCP Memory Server Landscape

### 17.1 Anthropic's Official Memory MCP Server

- **Package:** `@modelcontextprotocol/server-memory` (npm, v2026.1.26)
- **Storage:** Local JSONL file (`memory.jsonl`)
- **Data Model:** Entities (nodes) + Relations (directed, active voice) + Observations (facts about entities, max 500 chars)
- **Entity Constraints:** Lowercase, alphanumeric + hyphens, max 100 chars, unique names
- **Capabilities:** Create/read/update entities, add/remove observations, create/delete relations, search by name/type/content
- **Safety:** Marker to prevent accidental overwrites of unrelated JSONL files
- **Limitations:** No semantic search, no embeddings, no temporal tracking, no consolidation, no decay

### 17.2 Zep Knowledge Graph MCP Server

- Powered by Graphiti engine
- Persistent local graph memory with temporal tracking
- Relationship evolution over time
- More sophisticated than Anthropic's official server

### 17.3 MCP Memory Ecosystem (March 2026)

- **Scale:** 17,000+ MCP servers total; "Knowledge & Memory" is the largest category at 283 servers
- **Key players:** Anthropic (official), Zep (Graphiti-backed), Hindsight (MCP-first), SuperLocalMemory, various community servers
- **Benchmark:** AIMultiple evaluated 4 implementations — Handrails, Knowledge Graph, Basic Memory, Zine

### 17.4 Remanentia's MCP Position

Remanentia's MCP server exposes three tools:
- `remanentia_recall`: Deep contextual recall (traces + semantic memories + entity graph + temporal context + cross-project insights)
- `remanentia_status`: System status (daemon, memory counts, disk usage)
- `remanentia_graph`: Entity relationship queries

**Differentiation from standard MCP memory servers:**
- SNN-orchestrated recall vs flat JSONL lookup
- Consolidation engine vs static storage
- Cross-project knowledge synthesis vs isolated memory
- Novelty scoring on recall results vs raw retrieval
- Temporal before/after context vs stateless queries

---

## 18. Neuromorphic Hardware — Business Angle

### 18.1 Market Size & Growth

- **2024:** ~$54M (Consegic estimate)
- **2025:** ~$1B (StellarMR estimate)
- **2030 projection:** $45B (StellarMR) to $298M (Consegic) — estimates vary wildly depending on scope definition
- **Growth driver:** AI edge inference, robotics, IoT sensor networks, medical devices

### 18.2 Intel Loihi 3 (2026)

- **Process:** 4nm, 8x density increase over Loihi 2
- **Capacity:** 8M neurons, 64B synapses per chip
- **Power:** 1.2W peak load — 250x reduction vs GPU-based inference for robotics
- **Efficiency:** Up to 15 TOPS/W without batching (Hala Point results)
- **Software:** Lava framework (open source, platform-agnostic, prototype on CPU/GPU, deploy to Loihi)
- **Community:** INRC — ~150 members including Ford, Georgia Tech, SwRI, Teledyne-FLIR

### 18.3 SpiNNaker 2 (Dresden)

- **Scale:** 5.2M cores, 5B neurons across 720 boards, 8 server racks
- **Fabrication:** 34,500+ chips taped out
- **Access:** Through EBRAINS research infrastructure (academic, not commercial)

### 18.4 BrainScaleS 2 (Heidelberg)

- **Architecture:** Analog/mixed-signal neuromorphic system
- **Speed:** Up to 10,000x faster than biological real-time
- **Capacity:** 512 adaptive integrate-and-fire neurons, 131K plastic synapses per chip
- **Access:** Through EBRAINS (academic)

### 18.5 Commercial Applications (2026)

- **Robotics:** BMW (traffic sign recognition), Lockheed Martin (drone navigation)
- **Medical:** FDA-track seizure prediction wearables
- **Telecom:** Ericsson (network optimization)
- **IoT:** 40% of IoT sensor nodes projected to use neuromorphic chips by 2030 (industry forecast)
- **IEEE P2800:** Standardized neuromorphic benchmarks in development

### 18.6 Remanentia on Neuromorphic Hardware — The Business Case

**Feasibility:**
- Remanentia's SNN backend (snn_daemon.py, snn_backend.py) already implements LIF neurons, STDP, and spike-based retrieval
- Intel's Lava framework is platform-agnostic: prototype on CPU/GPU, deploy to Loihi
- Remanentia's architecture maps naturally to neuromorphic hardware: sparse spiking, event-driven consolidation, local plasticity rules

**Business Angle:**
1. **R&D Partnership:** Join Intel INRC to get Loihi access, demonstrate memory consolidation on neuromorphic hardware
2. **Edge Deployment:** Remanentia running on Loihi = persistent AI memory at 1.2W power — viable for edge agents, autonomous systems, wearables
3. **Academic Credibility:** First production memory system demonstrated on neuromorphic hardware would be a major differentiator and publication opportunity
4. **Long-term Moat:** If neuromorphic hardware goes mainstream (as 2026 indicators suggest), Remanentia's SNN-native architecture becomes a fundamental advantage that software-only competitors cannot replicate without major rewrites

**Risk:** Neuromorphic hardware is still primarily research-access. Commercial Loihi 3 availability is not yet confirmed for general purchase.

---

## 19. AI Agent Memory — IP Landscape

### 19.1 Patent Activity (2024-2026)

- **Volume:** US generative AI patent applications surged 56% in 14 months (Feb 2024 - Apr 2025) to 51,487 applications; granted patents rose 32%
- **Google:** Dominant filer. Patents on agent orchestration, collaborative agent networks, hierarchical computing ecosystems
- **OpenAI:** 110 patent applications filed, clustering from mid-2022 through 2024. Focus on speech, generation, fine-tuning rather than memory specifically
- **Microsoft:** GraphRAG and agent frameworks patented/patent-pending

### 19.2 Memory-Specific IP

- **ChatGPT Memory:** OpenAI's memory feature (Feb 2024, expanded Sep 2024) — likely trade-secret rather than patented
- **Claude Memory:** Anthropic's auto-memory (2025) — file-based, likely not patented
- **No specific "agent memory system" patents found** from major players — the field is still emerging
- **USPTO Guidance (Nov 2025):** AI is treated as a "tool that assists" — human inventors required for patentability
- **2026 Outlook:** New USPTO Director Squires is more favorable to AI-related patents

### 19.3 Remanentia IP Strategy

- **SNN-based memory consolidation for AI agents** is a novel combination not covered by existing patents
- **AGPL-3.0 + Commercial dual license** protects the code while enabling enterprise licensing
- **Patent-worthy elements:**
  - SNN-orchestrated episodic-to-semantic memory conversion for AI agents
  - Surprise-weighted consolidation using novelty detection in spiking networks
  - Kuramoto oscillator-based memory binding for agent context management
- **Defensive publication** (via arxiv/JOSS) establishes prior art to prevent others from patenting these approaches
- **Timing:** File provisional patent applications before any public benchmark results or production deployment announcements

---

## 20. Memory API — Developer Experience Analysis

### 20.1 What Makes a Memory API Great

**From industry analysis:**
1. **Three-line integration:** Supermemory's approach — `init()`, `add()`, `recall()`. Developers adopt what they can try in 5 minutes
2. **Typed SDKs:** Autocompletion in IDE, precise types eliminate an entire class of errors
3. **Agent Experience (AX):** The next frontier after DX — APIs must work well when called by LLMs, not just humans. Clear descriptions, consistent patterns, logged retries
4. **Lazy loading:** 7.5x faster initialization, 80% less memory. Don't load what you don't need
5. **Multi-language:** Python + TypeScript at minimum. REST as universal fallback
6. **Opinionated defaults:** Works out of the box with zero configuration
7. **Escape hatches:** Power users need access to underlying primitives

### 20.2 Current Remanentia DX Assessment

| Aspect | Current State | Target |
|--------|--------------|--------|
| Installation | `pip install remanentia[all]` | Good — single package |
| First use | Requires daemon setup, directory structure | Bad — should work on first API call |
| API surface | 7 endpoints (health, recall, consolidate, status, entities, graph, entity detail) | Good — comprehensive but may need simplification |
| MCP integration | 3 tools (recall, status, graph) | Good — focused |
| CLI | Full CLI (cli.py) | Good |
| SDK | No standalone SDK | Need Python + TypeScript SDKs |
| Documentation | README.md only | Need quickstart, API reference, tutorials |
| Onboarding time | ~30 minutes (estimated) | Target: <5 minutes |

### 20.3 Recommended DX Improvements for Remanentia

1. **Zero-config start:** `from remanentia import Memory; m = Memory(); m.add("fact"); m.recall("query")` — no daemon required for basic use
2. **Auto-initialization:** First call creates directory structure, starts background consolidation
3. **TypeScript SDK:** Critical for Claude Code, Cursor, and web agent ecosystems
4. **MCP server as pip extra:** `pip install remanentia[mcp]` auto-configures the server
5. **Interactive playground:** Web UI for exploring memory graph (web/ directory exists but needs polish)
6. **Benchmark suite:** Ship with LongMemEval and LOCOMO test harness so users can verify performance

---

## 21. Memory-as-a-Service Market Analysis

### 21.1 Market Structure

The AI agent memory market is consolidating around three business models:

| Model | Examples | Revenue Source |
|-------|----------|----------------|
| **Hosted SaaS** | Mem0 Cloud, Supermemory, Zep Cloud | Per-credit or per-tier subscription |
| **Open Core** | Letta, Graphiti, Hindsight | Free OSS + paid cloud/enterprise |
| **Framework Library** | LangMem, LangGraph Memory | Platform lock-in, indirect revenue |

### 21.2 Market Sizing

- **AI agent market:** $7.84B (2025) -> $52.62B (2030), 46.3% CAGR
- **Memory layer:** ~5-10% of agent infrastructure spend (estimated) = $400M-$800M by 2030
- **Mem0 validation:** $24M Series A at ~$100M+ valuation suggests VCs see memory as a standalone category

### 21.3 Remanentia's Business Model Options

| Model | Pros | Cons | Fit |
|-------|------|------|-----|
| **AGPL + Commercial License** | Protects IP, proven model (MongoDB, WURFL), forces enterprise licensing | Deters casual contributors | **Strong** — matches existing license |
| **Hosted SaaS** | Recurring revenue, easy metrics | Infrastructure costs, competition with Mem0/Zep | Medium — later phase |
| **MCP Marketplace** | Distribution via Claude Code, Cursor ecosystems | Unclear monetization, MCP servers mostly free | Medium — for awareness |
| **Research License** | Academic credibility, JOSS/JMLR publications | No direct revenue | **Strong** — for differentiation |
| **Neuromorphic Partnership** | First-mover on hardware, Intel INRC access | Slow to monetize | Long-term moat |

### 21.4 Recommended Go-to-Market

**Phase 1 (Current — v0.2.0):** Open source with AGPL. Publish benchmarks. MCP server for Claude Code adoption.

**Phase 2 (v0.5.0):** Commercial license for enterprises. Managed hosting (optional). TypeScript SDK. LangGraph integration.

**Phase 3 (v1.0.0):** Production SaaS. Neuromorphic hardware demo. JOSS paper submission. Enterprise pilot customers.

**Phase 4 (v2.0.0):** Neuromorphic edge deployment. Hardware partnerships. Agent-as-a-Service integration.

---

## 22. Competitive Position Matrix

### 22.1 Feature Comparison

| Feature | Mem0 | Letta | Zep | LangMem | Hindsight | Remanentia |
|---------|------|-------|-----|---------|-----------|------------|
| **Vector Search** | Yes (20+ backends) | Yes (archival) | Yes (hybrid) | Yes | Yes (multi-strategy) | Yes (TF-IDF + optional embeddings) |
| **Entity Graph** | Yes (Pro tier) | No | Yes (Graphiti) | No | No | Yes (JSONL-based) |
| **Temporal Tracking** | No | No | Yes (bitemporal) | No | No | Yes (spike timing) |
| **Memory Consolidation** | Dedup/conflict | Self-editing | Edge invalidation | Overwrite | Reflect | SNN episodic-to-semantic |
| **Novelty Detection** | No | No | No | No | No | Yes (surprise weighting) |
| **SNN Backend** | No | No | No | No | No | Yes |
| **MCP Server** | Community | No | Yes | No | Yes (native) | Yes |
| **Self-Hosted** | Docker (3 containers) | Docker/pip | Docker | Library only | Docker | Single process |
| **LLM Required** | Yes (extraction) | Yes (everything) | Yes (extraction) | Yes (extraction) | Yes (extraction) | No (SNN-native) |
| **Neuromorphic Ready** | No | No | No | No | No | Yes (Lava-compatible architecture) |
| **Pricing** | Free-$249/mo | Free-Enterprise | Free-$25/mo+ | Included in LangGraph | Open source | AGPL + Commercial |
| **Minimum Deps** | Vector DB + LLM | PostgreSQL + LLM | Neo4j + LLM | LangGraph + Postgres | Docker | numpy only |

### 22.2 Unique Selling Points — What Only Remanentia Has

1. **SNN-orchestrated memory:** No competitor uses spiking neural networks for memory consolidation. This is not a feature — it's a fundamentally different computational paradigm
2. **LLM-free memory operations:** Every competitor requires LLM calls for extraction/retrieval. Remanentia's SNN backend performs consolidation and retrieval locally, with zero API costs and constant-time operations
3. **Biologically-grounded architecture:** Hippocampal replay, STDP, novelty detection, Kuramoto oscillators — grounded in neuroscience, not prompt engineering
4. **Neuromorphic deployment path:** Architecture maps to Intel Loihi/Lava. No competitor can run on neuromorphic hardware without a complete rewrite
5. **Minimal infrastructure:** Runs from JSONL files with numpy as the only required dependency. No Docker, no database, no LLM API key required for core functionality
6. **Dual scientific/commercial identity:** Academic paper + production system. Competitors are either research (no product) or product (no science)

### 22.3 Where Competitors Are Stronger (Honest Assessment)

1. **Mem0:** Larger ecosystem (50K stars), more vector DB integrations, enterprise compliance (SOC2, HIPAA)
2. **Letta:** Deeper agent integration, self-editing paradigm is more flexible for agent-driven use cases
3. **Zep/Graphiti:** Superior temporal knowledge graph with formal bitemporal model and Neo4j backing
4. **Hindsight:** Better benchmark scores (91% LongMemEval), MCP-first design
5. **Supermemory:** Simpler DX (3 lines), better benchmarks, user-profile-centric approach

### 22.4 Strategic Gaps to Close

1. **Benchmark scores:** Must publish LongMemEval and LOCOMO results to compete credibly
2. **SDK coverage:** Need TypeScript SDK for web/agent ecosystems
3. **Compliance:** SOC 2 and HIPAA are table stakes for enterprise
4. **Documentation:** Comprehensive docs, quickstart guide, API reference
5. **Community:** Build contributor base around the scientific differentiation

---

## Sources

### Mem0
- [Mem0 Official Site](https://mem0.ai/)
- [Mem0 Pricing](https://mem0.ai/pricing)
- [Mem0 GitHub](https://github.com/mem0ai/mem0)
- [Mem0 Self-Host Docker Guide](https://mem0.ai/blog/self-host-mem0-docker)
- [Mem0 Storage Backends (DeepWiki)](https://deepwiki.com/mem0ai/mem0/5-vector-stores)
- [Mem0 Honest Review (Medium)](https://medium.com/@reliabledataengineering/mem0-do-ai-agents-really-need-memory-honest-review-6760b5288f37)
- [Mem0 $24M Series A](https://startupwired.com/2025/10/29/mem0-raises-24-million-series-a-to-build-the-memory-layer/)
- [Why Scira AI Switched from Mem0 to Supermemory](https://supermemory.ai/blog/why-scira-ai-switched/)

### Letta/MemGPT
- [Letta Official Site](https://www.letta.com/)
- [Letta GitHub](https://github.com/letta-ai/letta)
- [Letta Docs — MemGPT Concepts](https://docs.letta.com/concepts/memgpt/)
- [Letta Pricing](https://www.letta.com/pricing)
- [Letta on Aurora PostgreSQL (AWS)](https://aws.amazon.com/blogs/database/how-letta-builds-production-ready-ai-agents-with-amazon-aurora-postgresql/)
- [Letta $10M Stealth Emergence](https://www.hpcwire.com/bigdatawire/this-just-in/letta-emerges-from-stealth-with-10m-to-build-ai-agents-with-advanced-memory/)
- [Stateful AI Agents Deep Dive (Medium)](https://medium.com/@piyush.jhamb4u/stateful-ai-agents-a-deep-dive-into-letta-memgpt-memory-models-a2ffc01a7ea1)

### Zep
- [Zep Official Site](https://www.getzep.com/)
- [Zep Pricing](https://www.getzep.com/pricing/)
- [Zep Paper (arXiv)](https://arxiv.org/html/2501.13956v1)
- [Graphiti GitHub](https://github.com/getzep/graphiti)
- [Zep Knowledge Graph MCP Server](https://www.getzep.com/product/knowledge-graph-mcp/)
- [Mem0 vs Zep Comparison](https://vectorize.io/articles/mem0-vs-zep)

### LangMem/LangGraph
- [LangMem Documentation](https://langchain-ai.github.io/langmem/)
- [LangMem GitHub](https://github.com/langchain-ai/langmem)
- [LangMem SDK Launch Blog](https://blog.langchain.com/langmem-sdk-launch/)
- [LangGraph Deploy CLI](https://blockchain.news/news/langchain-deploy-cli-langgraph-agent-deployment)
- [MongoDB + LangGraph Memory](https://www.mongodb.com/company/blog/product-release-announcements/powering-long-term-memory-for-agents-langgraph)

### Emerging Competitors
- [Hindsight GitHub](https://github.com/vectorize-io/hindsight)
- [Hindsight 91% LongMemEval](https://topaiproduct.com/2026/03/14/hindsight-by-vectorize-hits-91-on-longmemeval-the-case-for-giving-ai-agents-human-like-memory/)
- [Hindsight MCP Server](https://hindsight.vectorize.io/blog/2026/03/04/mcp-agent-memory)
- [Mengram on Product Hunt](https://www.producthunt.com/products/mengram)
- [Supermemory](https://supermemory.ai/)
- [Supermemory GitHub](https://github.com/supermemoryai/supermemory)
- [Top 10 AI Memory Products 2026 (Medium)](https://medium.com/@bumurzaqov2/top-10-ai-memory-products-2026-09d7900b5ab1)
- [5 AI Agent Memory Systems Compared (DEV)](https://dev.to/varun_pratapbhardwaj_b13/5-ai-agent-memory-systems-compared-mem0-zep-letta-supermemory-superlocalmemory-2026-benchmark-59p3)
- [Best AI Agent Memory Systems 2026 (Vectorize)](https://vectorize.io/articles/best-ai-agent-memory-systems)
- [Mem0 Alternatives (Vectorize)](https://vectorize.io/articles/mem0-alternatives)

### Claude Code Memory
- [Claude Code Memory Docs](https://code.claude.com/docs/en/memory)
- [Claude Code Memory Explained (Substack)](https://joseparreogarcia.substack.com/p/claude-code-memory-explained)
- [CLAUDE.md Deep Dive (SFEIR)](https://institute.sfeir.com/en/claude-code/claude-code-memory-system-claude-md/deep-dive/)
- [Claude Code Memory Architecture (Ian Paterson)](https://ianlpaterson.com/blog/claude-code-memory-architecture/)
- [Anthropic Auto-Memory Tested (Medium)](https://medium.com/@joe.njenga/anthropic-just-added-auto-memory-to-claude-code-memory-md-i-tested-it-0ab8422754d2)
- [claude-mem Plugin (GitHub)](https://github.com/thedotmack/claude-mem)

### MCP Memory Servers
- [MCP Memory Server (npm)](https://www.npmjs.com/package/@modelcontextprotocol/server-memory)
- [MCP Knowledge Graph Memory Server](https://github.com/modelcontextprotocol/servers/tree/main/src/memory)
- [MCP Specification (2025-11-25)](https://modelcontextprotocol.io/specification/2025-11-25)
- [2026 MCP Roadmap](http://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/)
- [Knowledge & Memory MCP Servers (Glama)](https://glama.ai/mcp/servers/categories/knowledge-and-memory)
- [MCP Memory Benchmark (AIMultiple)](https://aimultiple.com/memory-mcp)
- [Awesome MCP Servers — Memory](https://github.com/TensorBlock/awesome-mcp-servers/blob/main/docs/knowledge-management--memory.md)

### Neuromorphic Hardware
- [Intel Neuromorphic Computing](https://www.intel.com/content/www/us/en/research/neuromorphic-computing.html)
- [Intel Hala Point Announcement](https://newsroom.intel.com/artificial-intelligence/intel-builds-worlds-largest-neuromorphic-system-to-enable-more-sustainable-ai)
- [Loihi 2 (Open Neuromorphic)](https://open-neuromorphic.org/neuromorphic-computing/hardware/loihi-2-intel/)
- [Lava Framework](https://lava-nc.org/)
- [SpiNNaker 2 (Open Neuromorphic)](https://open-neuromorphic.org/neuromorphic-computing/hardware/spinnaker-2-university-of-dresden/)
- [EBRAINS Neuromorphic Computing](https://ebrains.eu/data-tools-services/computing-infrastructure/neuromorphic-computing)
- [Neuromorphic Computing Goes Mainstream 2026](https://markets.chroniclejournal.com/chroniclejournal/article/tokenring-2026-1-21-the-brain-inspired-revolution-neuromorphic-computing-goes-mainstream-in-2026)
- [Neuromorphic Market Analysis (Grand View)](https://www.grandviewresearch.com/industry-analysis/neuromorphic-computing-market)

### AI Memory Market & Business Model
- [Memory for AI Agents: New Paradigm (The New Stack)](https://thenewstack.io/memory-for-ai-agents-a-new-paradigm-of-context-engineering/)
- [AI Agents Memory Ecosystem (Business Engineer)](https://businessengineer.ai/p/the-ai-agents-memory-ecosystem)
- [Memory as Asset (arXiv)](https://arxiv.org/pdf/2603.14212)
- [Memory in the Age of AI Agents (arXiv)](https://arxiv.org/abs/2512.13564)
- [Agent as a Service (AaaS)](https://www.ema.ai/additional-blogs/addition-blogs/agent-as-service-future-beyond-saas)
- [Google Always On Memory Agent (VentureBeat)](https://venturebeat.com/orchestration/google-pm-open-sources-always-on-memory-agent-ditching-vector-databases-for)

### IP Landscape
- [AI Patent Outlook 2026 (Greenberg Traurig)](https://www.gtlaw.com/en/insights/2026/01/ai-patent-outlook-for-2026)
- [Google Dominates AI Patents (Axios)](https://www.axios.com/2025/05/15/ai-patents-google-agents)
- [OpenAI Patents (GreyB)](https://insights.greyb.com/openai-patents/)
- [OpenAI Patent List (Originality.AI)](https://originality.ai/blog/openai-patent-list)
- [AGPL as Non-Starter for Companies](https://www.opencoreventures.com/blog/agpl-license-is-a-non-starter-for-most-companies)
- [Can You Patent an Algorithm in 2026](https://patentailab.com/how-to-patent-ai-algorithms-in-2025-navigating-section-101-without-rejection/)

### Developer Experience
- [API Design Principles for Agentic Era (Apideck)](https://www.apideck.com/blog/api-design-principles-agentic-era)
- [DX Best Practices in REST API Design (Speakeasy)](https://www.speakeasy.com/api-design/developer-experience)
- [Building Great SDKs (Pragmatic Engineer)](https://newsletter.pragmaticengineer.com/p/building-great-sdks)
- [AI SDK Agents Memory](https://ai-sdk.dev/docs/agents/memory)
- [How to Build a Great DX (Apideck)](https://www.apideck.com/blog/how-to-build-a-great-developer-experience)

---

*Round 3 compiled by Arcane Sapience for Remanentia development. All sources verified as of 2026-03-20.*

---

# ROUND 4: IMPLEMENTATION PATTERNS — Deep Technical Research

## Compiled: 2026-03-20 | Implementation-focused deep dive for Remanentia consolidation pipeline

---

## 23. Zep/Graphiti — Complete Implementation Architecture

### 23.1 Data Model Schema (from arXiv 2501.13956)

The knowledge graph G = (N, E, phi) contains three node types and three edge types.

**Episode Nodes (N_e):**
- Text content (raw message, JSON, or document fragment)
- Actor/speaker identifier
- Reference timestamp (t_ref) — the wall-clock time of the episode
- Source type: `EpisodeType.text`, `EpisodeType.json`, `EpisodeType.message`
- `group_id` — namespace for multi-tenant isolation (default: "main")
- `source_description` — provenance string (e.g., "Customer support chat — Session #47")

**Entity Nodes (N_s):**
- `uuid` — unique identifier
- `name` — full explicit string (e.g., "Miroslav Sotek", not "he")
- `summary` — LLM-generated contextual description, updated on re-encounters
- `name_embedding` — 1024-dimensional dense vector for cosine search
- `group_id` — namespace
- `labels` — Neo4j node labels
- `created_at` — system timestamp
- `attributes` — extensible via Pydantic custom entity types

**Community Nodes (N_c):**
- Community summary — generated via iterative map-reduce over member entities
- Community name — key terms extracted from summary
- Embedded name vector — enables cosine similarity search over communities

**Episodic Edges (E_e):**
- Connect episodes to the entities they mention
- Bidirectional indices: "which episodes mention entity X?" and "which entities appear in episode Y?"
- Enable provenance tracking from any derived fact back to source episode

**Semantic Edges (E_s) — THE CRITICAL PIECE:**
- `fact` — natural language statement of the relationship (e.g., "Miroslav works on Remanentia")
- `relation_type` — concise all-caps label (e.g., "WORKS_ON", "LIVES_IN")
- `t_created` (T') — when this edge was created in the system
- `t_expired` (T') — when this edge was invalidated in the system
- `t_valid` (T) — when this fact became true in reality
- `t_invalid` (T) — when this fact stopped being true in reality
- Embedding vector for the fact description

**Community Edges (E_c):**
- Connect community nodes to their member entity nodes

### 23.2 Bitemporal Model — How It Works

Two independent timelines:
- **T (Event Timeline):** Chronological ordering of real-world events. Extracted from episode content by analyzing message text against reference timestamp. Handles both absolute dates ("June 23, 2024") and relative expressions ("two weeks ago") by computing against t_ref.
- **T' (Transactional Timeline):** System ingestion order. Used for database auditing.

Each semantic edge carries four timestamps: `t_created`, `t_expired` (system time), `t_valid`, `t_invalid` (real-world time). This enables queries like:
- "What was true at time X?" — filter by t_valid <= X < t_invalid
- "What did we believe at time X?" — filter by t_created <= X < t_expired
- "What changed between session A and session B?" — diff by t_created range

### 23.3 Contradiction Detection and Resolution

1. New edge extracted from episode
2. System performs semantic search (cosine + BM25) for existing edges with related entities
3. LLM compares new edge against semantically related existing edges
4. If contradiction detected with temporal overlap: old edge's `t_invalid` is set to new edge's `t_valid`
5. Priority rule: **new information always wins** — Graphiti consistently prioritizes recent data
6. Old edges are preserved (never deleted), maintaining full history

### 23.4 Entity Extraction Pipeline (5 phases)

**Phase 1 — Initial Extraction:** Process current message + last N=4 messages (two conversation turns). Auto-extract speaker as entity. Uses reflexion technique to minimize hallucinations.

**Phase 2 — Embedding:** Create 1024-dim vectors for all extracted entity names.

**Phase 3 — Candidate Retrieval:** Parallel full-text search on existing entity names and summaries. Combine vector and text search results.

**Phase 4 — Resolution:** Pass candidates + episode context through LLM entity resolution prompt. Identify duplicates, generate updated name/summary on match.

**Phase 5 — Integration:** Predefined Cypher queries (NOT LLM-generated) for schema consistency. Prevents hallucinated graph modifications.

### 23.5 Hybrid Search Architecture

Three search functions composed into unified retrieval:

- **phi_cos** — Cosine similarity via Neo4j Lucene index. Searches entity name embeddings.
- **phi_bm25** — BM25 full-text search via Neo4j Lucene. Searches edge fact descriptions.
- **phi_bfs** — Breadth-first graph traversal from initial results. Accepts nodes as seeds. Reveals contextual proximity.

Results combined into 3-tuple: (semantic_edges, entity_nodes, community_nodes).

Reranking options (configurable): Reciprocal Rank Fusion, MMR, episode-mentions frequency, node distance from centroid, cross-encoder LLM reranking.

### 23.6 Community Detection

Uses **label propagation** (not Leiden) — when adding new entity n_i, surveys neighboring nodes' communities, assigns n_i to the plurality community. Single recursive step enables efficient incremental updates without full recompute. Periodic full refreshes needed as divergence accumulates.

### 23.7 `add_episode` API Example

```python
await client.add_episode(
    name="support_chat_42",
    episode_body="User reports that the consolidation pipeline crashes on files > 10MB",
    source_description="Customer support ticket #42",
    source=EpisodeType.text,
    entity_types={"User": UserModel, "Bug": BugModel},
    edge_types={"REPORTS": ReportsEdge},
    reference_time=datetime.now(timezone.utc),
    group_id="support_team"
)
```

- **Remanentia Relevance:** Graphiti's schema maps almost 1:1 to what we need. The four-timestamp bitemporal model on edges is the gold standard for temporal memory. Our JSONL-based entity graph should adopt t_valid/t_invalid fields. The label propagation community detection is cheaper than Leiden and viable for incremental updates — directly applicable to our consolidation pipeline. The key difference: Graphiti requires LLM for entity extraction; Remanentia should use spaCy/GLiNER + SNN for the same.

- **Sources:**
  - [Zep Paper (arXiv)](https://arxiv.org/abs/2501.13956)
  - [Zep Paper HTML](https://arxiv.org/html/2501.13956v1)
  - [Graphiti GitHub](https://github.com/getzep/graphiti)
  - [Graphiti Neo4j Blog](https://neo4j.com/blog/developer/graphiti-knowledge-graph-memory/)

---

## 24. Microsoft GraphRAG — Complete Pipeline Architecture

### 24.1 Five-Phase Dataflow

**Phase 1 — Compose TextUnits:** Documents chunked into configurable-size text units with overlap. Chunk boundaries respect sentence/paragraph structure when possible.

**Phase 2 — Graph Extraction:** Each text unit processed by LLM to extract entities (with title, type, description) and relationships (with source, target, description). The extraction prompt has four sections:
1. Extraction instructions (entity types, relationship types to look for)
2. Few-shot examples (default: 15 entity examples, 12 relationship examples for org/geo/person types)
3. Real data placeholder (the text chunk)
4. Gleanings — multi-turn continuation where the LLM assesses whether all entities were found, and if not, a more aggressive prompt forces additional extraction

**Phase 3 — Graph Augmentation:** Entity resolution via exact string matching. Leiden community detection (hierarchical, using graspologic library) with configurable GRAPHRAG_MAX_CLUSTER_SIZE (default: 10). Recursive sub-community detection until leaf communities.

**Phase 4 — Community Summarization:** LLM generates summary for each community containing executive overview + key entities + relationships + claims. Map-reduce: summaries at higher hierarchy levels incorporate lower-level summaries. Fixed 8K token context window.

**Phase 5 — Document Processing:** Final export to parquet files: text_units.parquet, entities.parquet, relationships.parquet, community_reports.parquet.

### 24.2 Local vs Global Search

**Local Search:** Combines structured KG data (entities, relationships) with unstructured source documents. Good for specific entity queries. Uses entity/relationship subgraph + source text chunks as context.

**Global Search (Map-Reduce):**
1. Map step: Each community summary independently answers the query (parallel)
2. Reduce step: Partial answers summarized into final global answer
3. Good for holistic/thematic queries spanning entire corpus

### 24.3 Gleaning Process — Step by Step

1. LLM extracts entities and relationships from text chunk
2. System prompts LLM: "Did you extract all entities? Are there any you missed?"
3. If LLM says yes (entities missed): continuation prompt states "MANY entities were missed in the last extraction" — deliberately aggressive phrasing
4. LLM extracts additional entities
5. Repeat up to configurable maximum gleanings
6. Each gleaning round is a separate LLM call (cost scales linearly)

### 24.4 Incremental Updates (as of 2025)

`get_delta_docs` function compares input dataset with existing final documents. Only new documents are processed (chunked, extracted). Existing documents are fetched from cache (no re-extraction). Graph construction rebuilds to include new nodes/edges. Community detection recomputed (expensive step). Planned `graphrag.append` command to minimize community recomputes via threshold-based triggering.

### 24.5 Entity Resolution Limitation

GraphRAG uses **exact string matching** for entity deduplication. This means "Miroslav Sotek" and "Miroslav" would remain as separate entities unless post-processing consolidates them. Graphiti's embedding-based resolution is superior here.

- **Remanentia Relevance:** GraphRAG's gleaning technique is worth adopting for our LLM-assisted extraction mode — asking "did you miss anything?" catches 15-20% more entities. The Leiden community detection creates useful hierarchical summaries but is expensive; for our incremental pipeline, Graphiti's label propagation is better. The parquet output format is a good reference for structured memory export. The key takeaway: GraphRAG is batch-oriented (process a corpus), not streaming (process one episode at a time). Remanentia needs streaming.

- **Sources:**
  - [GraphRAG Paper](https://arxiv.org/abs/2404.16130)
  - [GraphRAG Docs — Dataflow](https://microsoft.github.io/graphrag/index/default_dataflow/)
  - [GraphRAG GitHub](https://github.com/microsoft/graphrag)
  - [GraphRAG Auto-Tuning (Microsoft Research)](https://www.microsoft.com/en-us/research/blog/graphrag-auto-tuning-provides-rapid-adaptation-to-new-domains/)

---

## 25. Temporal Knowledge Graph Embedding — Method Taxonomy

### 25.1 Ten Categories of TKG Methods (from survey arXiv 2403.04782)

| Category | Key Models | Time Representation | Best For |
|----------|-----------|-------------------|----------|
| Translation-based | TTransE, TA-TransE, HyTE | Concatenation or hyperplane projection | Simple temporal facts |
| Rotation-based | TeRo, ChronoR, RotateQVS | Complex/quaternion space rotation | Capturing periodic patterns |
| CP Decomposition | DE-SimplE, TComplEx, TNTComplEx | 4th-order tensor with time dimension | Dense temporal data |
| Tucker Decomposition | TuckERT, TLT-KGE | Core tensor + per-mode embeddings | Separating semantic vs temporal |
| GNN-based | TEA-GNN, TREA, T2TKG | Attention over temporal neighbor graphs | Structural + temporal features |
| Capsule Networks | TempCaps, BiQCap, DuCape | Sliding windows + dynamic routing | Small-scale, high accuracy |
| Autoregressive | RE-NET, RE-GCN, TiRGN | GRU/LSTM over snapshot sequences | Extrapolation (future prediction) |
| Temporal Point Process | Know-Evolve, GHNN, EvoKG | Continuous-time intensity functions | Event timing prediction |
| Interpretable | xERTE, CluSTeR, TITer | RL or attention over subgraphs | Explainable temporal reasoning |
| Language Model | ICLTKG, zrLLM, GenTKG | In-context learning or fine-tuning | Zero-shot temporal QA |

### 25.2 Interpolation vs Extrapolation

- **Interpolation:** Predict missing facts within observed time range. Standard link prediction: given (h, r, ?, t), find tail entity. Models: DE-SimplE, TComplEx, TeRo.
- **Extrapolation:** Predict future facts beyond observed timestamps. Given history up to t, predict events at t+1. Models: RE-NET, RE-GCN, CluSTeR. Harder task, requires capturing temporal dynamics.

### 25.3 Time Representation Strategies

| Strategy | How It Works | Example |
|----------|-------------|---------|
| Concatenation | Append time embedding to relation vector | TTransE: score = h + r + tau - t |
| Temporal vectors | Learn separate embedding for each timestamp | DE-SimplE: entity embedding is f(entity, time) |
| Rotation | Time as rotation operator in complex/quaternion space | ChronoR: k-dimensional rotation |
| Projection | Time defines a hyperplane; entities projected onto it | HyTE: project onto w_tau plane |
| Separation | Imaginary components for temporal, real for semantic | TLT-KGE |
| Piecewise linear | Time modulates entity embeddings element-wise | DE-SimplE: diachronic entity embedding |
| Continuous ODE | Solve neural ODE to evolve embeddings over time | TANGO |

### 25.4 Benchmark Datasets

| Dataset | Entities | Relations | Timestamps | Facts |
|---------|----------|-----------|-----------|-------|
| ICEWS14 | 7,128 | 230 | 365 | 90,730 |
| ICEWS05-15 | 10,488 | 251 | 4,017 | 461,329 |
| GDELT | 7,691 | 240 | 2,751 | 2,278,405 |
| Wikidata | 12,554 | 24 | 232 | 669,934 |

### 25.5 Most Practical for Remanentia

For Remanentia's use case (agent memory, not academic KG completion), the most applicable approaches are:

1. **DE-SimplE's diachronic embeddings** — entity representations that naturally evolve with time. Lightweight, no complex architecture. Could be applied to our entity nodes to encode temporal context.
2. **Autoregressive snapshot models (RE-GCN)** — process session snapshots sequentially via GRU. Maps to our session-based architecture where each session is a "snapshot."
3. **Continuous-time models (TANGO via Neural ODE)** — if we want continuous temporal evolution between sessions rather than discrete snapshots. Heavier but more theoretically elegant.

For our SNN-based architecture, the spike timing itself naturally encodes temporal information (neurons that fire together wire together), giving us a biological analog to learned temporal embeddings without the tensor decomposition overhead.

- **Sources:**
  - [TKG Survey (arXiv 2403.04782)](https://arxiv.org/abs/2403.04782)
  - [TKG Embedding Survey (Knowledge-Based Systems)](https://www.sciencedirect.com/science/article/abs/pii/S0950705124010888)
  - [TKGER Paper List (GitHub)](https://github.com/stmrdus/tkger)

---

## 26. Causal Graph Extraction from Text

### 26.1 Zero-Shot Pairwise Approach (arXiv 2312.14670)

**Methodology:**
1. Extract entities from text via LLM (diseases, medications, treatments, symptoms)
2. For each entity pair (A, B), prompt LLM: "Determine the most likely cause-and-effect relationship: (a) A causes B, (b) B causes A, (c) no direct causal relation"
3. Prompt includes source text delimited by XML tags + entity pair + step-by-step reasoning request
4. Aggregate all pairwise results into a DAG

**Performance:**
- Pairwise orientation F1: ~99% (outperforms SemEval 2023 winner at ~90%)
- Graph-level recall: ~97%
- Graph-level precision: ~74% (high false positive rate from transitive inference)
- Processing: ~30 min per abstract (~20 entities, quadratic pairwise queries)

**Limitation:** O(n^2) queries for n entities. Not scalable beyond ~20-30 entities per document.

### 26.2 BFS-Based DAG Construction (Efficient Alternative)

A novel approach constructs causal graphs via breadth-first search:
1. Identify root cause nodes
2. For each node, prompt LLM: "Which variables are directly caused by X?"
3. Expand children, repeat
4. O(n) queries instead of O(n^2)

### 26.3 Causal-LLM: One-Shot Full Graph Discovery (EMNLP 2025)

**Key Innovation:** Instead of pairwise or iterative queries, discovers the complete causal graph in a single LLM call.
- Prompt-based discovery with in-context learning when node metadata available
- Data-driven method (Causal_llm) for settings without metadata
- Outperforms GranDAG, GES, ICA-LiNGAM by ~40% edge accuracy
- 50% faster inference than RL-based methods
- 25% precision improvement in fairness-sensitive domains

### 26.4 Event Causality Extraction (Financial/Medical)

Recent methods for multi-causal extraction:
- **LA-MPGN (Wu & Cao 2025):** Multi-prompt generation + graph convolution for label-aware extraction. Handles multiple causes/effects.
- **CHECE (Luo et al. 2024):** Contextual highlighting for event boundaries + LLM-assisted template augmentation.
- **BioBERT:** Still outperforms LLMs for medical causal extraction (F1: 0.72 avg)

### 26.5 Practical Extraction Without LLM

For Remanentia's heuristic consolidation (no LLM calls):
- Pattern matching for causal connectives: "because", "caused by", "led to", "resulted in", "due to", "therefore", "consequently", "as a result"
- Dependency parsing: identify nsubj-ROOT-dobj patterns where ROOT is a causal verb
- Temporal ordering: if A precedes B and A is mentioned as context for B, infer weak causal link
- Confidence scoring: explicit causal markers > temporal proximity > co-occurrence

- **Remanentia Relevance:** For our SNN-native pipeline (no LLM), use pattern-matching on causal connectives + dependency parsing for causal edge extraction. For our LLM-assisted mode, the one-shot Causal-LLM approach is ideal — single call extracts the full causal subgraph. The BFS approach is a good middle ground when we have >20 entities. Causal edges in our entity graph should carry a confidence score and extraction method tag.

- **Sources:**
  - [Zero-shot Causal Graph Extrapolation (arXiv 2312.14670)](https://arxiv.org/abs/2312.14670)
  - [Causal-LLM (EMNLP 2025)](https://aclanthology.org/2025.findings-emnlp.439/)
  - [Event Causality Survey (ACM)](https://dl.acm.org/doi/10.1145/3756009)
  - [LLMs for Causal Discovery Survey](https://arxiv.org/abs/2402.11068)

---

## 27. Knowledge Graph Construction Without LLM

### 27.1 spaCy Dependency-Based Triple Extraction

**Pipeline:**
1. Tokenize + POS-tag with spaCy
2. Filter sentences lacking verbs
3. Run dependency parser
4. Extract noun phrases as entity candidates
5. Identify subject (nsubj dependency), root verb (ROOT), and object (dobj/pobj)
6. Form (subject, verb, object) triples
7. Handle compound nouns, prepositional modifiers, conjunctions

**Strengths:** Fast (industrial-grade), no API calls, deterministic, scalable to millions of documents.
**Weaknesses:** Misses implicit relations, context-dependent meaning, relations not expressed in SVO syntax.

### 27.2 REBEL: Seq2Seq Relation Extraction (EMNLP 2021)

- Fine-tuned BART model for end-to-end relation extraction
- 200+ relation types from Wikidata
- Input: raw text. Output: linearized triples (subject, relation_type, object)
- No LLM needed — runs on a single GPU or even CPU
- Multilingual variant (mREBEL) based on M2M100
- Available on HuggingFace: `Babelscape/rebel-large`

**Example:**
```
Input: "SAP launched Joule for Consultants in Berlin."
Output: [("SAP", "developer", "Joule"), ("SAP", "headquarters_location", "Berlin")]
```

### 27.3 ReLiK: Fast Entity Linking + Relation Extraction (ACL 2024)

- Retriever + Reader architecture
- Entity linking (EL) and relation extraction (RE) in a single pass
- Small model (~500M params) achieves near-LLM performance
- Available: `sapienzanlp/relik-entity-linking-large`, `sapienzanlp/relik-relation-extraction-nyt-large`
- Small variant available for edge deployment: `relik-ie/relik-relation-extraction-small`

### 27.4 GLiNER: Zero-Shot NER Without LLM (NAACL 2024)

- Bidirectional transformer encoder, <500M params
- Frames NER as matching between entity-type embeddings and span representations
- Zero-shot: specify any entity types at inference time (no retraining)
- 140x smaller than 13B UniNER, comparable performance
- Runs on CPU without GPU
- GLiNER2 extends to universal information extraction (NER + RE)

```python
from gliner import GLiNER
model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")
entities = model.predict_entities(
    "Miroslav works on Remanentia at the GOTM collection",
    labels=["person", "project", "organization"]
)
# [{"text": "Miroslav", "label": "person"}, {"text": "Remanentia", "label": "project"}, ...]
```

### 27.5 Dependency-Based KG Construction (arXiv 2507.03226, July 2025)

A recent paper demonstrates CPU-friendly KG construction achieving **94% of GPT-4o performance**:
- SpaCy noun phrase extractor for entity identification
- DependencyExtractor converts parse trees to SVO triples
- EntityRelationNormalizer for deduplication (string normalization + merging)
- One-hop graph traversal for retrieval
- Hybrid re-ranking with Reciprocal Rank Fusion

Performance: 61.07% context precision vs GPT-4o's 63.82% — only 2.75% gap.

### 27.6 Recommended Stack for Remanentia (No-LLM Mode)

| Layer | Tool | Role |
|-------|------|------|
| NER | GLiNER (zero-shot, CPU) | Entity detection with custom types |
| Relation Extraction | REBEL or ReLiK-small | Triple extraction without LLM |
| Causal Extraction | Regex + dependency parsing | Pattern-match causal connectives |
| Entity Resolution | TF-IDF cosine similarity | Merge duplicate entities |
| Graph Storage | JSONL with adjacency lists | Lightweight, no database dependency |

This stack runs entirely on CPU, requires no API keys, and achieves ~90-94% of LLM-based extraction quality. For Remanentia's consolidation pipeline, this is the sweet spot: good enough for automated processing, with LLM-assisted mode available as a premium upgrade.

- **Sources:**
  - [REBEL GitHub](https://github.com/Babelscape/rebel)
  - [ReLiK (ACL 2024)](https://arxiv.org/abs/2408.00103)
  - [GLiNER (NAACL 2024)](https://arxiv.org/abs/2311.08526)
  - [GLiNER GitHub](https://github.com/urchade/GLiNER)
  - [Efficient KG Construction (arXiv 2507.03226)](https://arxiv.org/abs/2507.03226)
  - [spaCy KG Triple Extraction Guide](https://www.analyticsvidhya.com/blog/2019/10/how-to-build-knowledge-graph-text-using-spacy/)
  - [KG Without LLMs (AIHello)](https://www.aihello.com/resources/blog/creating-a-knowledge-graph-without-llms/)

---

## 28. Graph-Based Memory for Multi-Agent Systems

### 28.1 Memory Architecture Paradigms (from arXiv 2603.10062)

**Three-layer memory hierarchy (computer architecture analog):**

| Layer | Analog | Content | Speed | Capacity |
|-------|--------|---------|-------|----------|
| Agent I/O | Bus/Peripherals | Audio, text, images, network calls | Real-time | Unbounded input |
| Agent Cache | L1/L2 Cache | Compressed context, KV caches, recent tool calls, embeddings | Fast | Limited (~128K tokens) |
| Agent Memory | Main Memory/Disk | Full history, vector DBs, graph DBs, document stores | Slow | Large (persistent) |

**Two critical protocol gaps identified:**
1. **Cache sharing:** No principled protocol for one agent's cached results to be reused by another. Analogous to cache coherence in multiprocessors — needs research.
2. **Memory access control:** Undefined whether agents can read each other's long-term memory, whether access is read-only or read-write, and what the unit of access is (document, chunk, record, trace).

### 28.2 Shared vs Distributed Memory Trade-offs

| Aspect | Shared Memory | Distributed Memory |
|--------|--------------|-------------------|
| Knowledge reuse | Natural — single pool | Requires synchronization |
| Write contention | High — needs serialization | Low — local writes |
| Scalability | Limited — bottleneck risk | Good — no central point |
| Consistency | Easier (single source of truth) | Harder (state divergence) |
| Privacy | Weaker (everything visible) | Stronger (local isolation) |
| Robustness | Single point of failure | No global crash |

**Hybrid design (best practice):** Orchestrator holds high-level team memory; each specialist agent records details of own task execution. Agents read orchestrator's summary instead of each other's raw traces.

### 28.3 KARMA Framework: Nine Agents for KG Enrichment (NeurIPS 2025)

**Agent Roles:**
1. Central Controller Agent (CCA) — task scheduling via utility scoring
2. Ingestion Agents (IA) — document retrieval and normalization
3. Reader Agents (RA) — segment parsing with relevance scoring
4. Summarizer Agents (SA) — compression while preserving technical detail
5. Entity Extraction Agents (EEA) — NER with canonical normalization
6. Relationship Extraction Agents (REA) — inter-entity relation detection
7. Schema Alignment Agents (SAA) — map to existing ontology
8. Conflict Resolution Agents (CRA) — debate mechanism for contradictions
9. Evaluator Agents (EA) — confidence/clarity/relevance scoring for integration decisions

**Conflict Resolution Debate Mechanism:**
- CRA receives conflicting triplets from REA
- LLM-based debate produces three outcomes: Agree, Contradict, Ambiguous
- Considers contextual compatibility (different conditions may allow coexistence)
- Confidence-based escalation: high-confidence conflicts trigger manual review

**Integration Decision:**
```
integrate(t) = 1  if  [C(t) + Cl(t) + R(t)] / 3 >= Theta
             = 0  otherwise
```
Where C = confidence, Cl = clarity, R = relevance, Theta = threshold.

**Results on 1,200 PubMed articles:**
- 38,230 new entities discovered
- 83.1% LLM-verified correctness
- 18.6% conflict edge reduction via multi-layer assessment
- Removing Summarizer: -18.2% accuracy. Removing Conflict Resolution: -9.7% accuracy.

### 28.4 MemOS: Memory Operating System (July 2025)

**Architecture — three layers:**
1. Interface Layer — user/agent interaction
2. Operation Layer — memory CRUD, conflict detection, version management
3. Infrastructure Layer — storage backends

**MemCube abstraction:** Self-contained memory unit pairing content with metadata (provenance, versioning, governance rules). Encapsulates plaintext, activation, and parameter memories.

**Version Chain:** Logs each memory's modification history and derivation lineage. Enables version control, conflict resolution, and rollback.

**Multi-agent support:** MemStore for inter-agent collaboration. MemLoader/MemDumper for agent migration. Redis Streams scheduling for concurrent access.

### 28.5 Remanentia Multi-Agent Architecture

For Remanentia's multi-agent scenario (Claude, Codex, Gemini working in parallel on GOTM):

**Current approach:** Each agent writes to `.coordination/sessions/{PROJECT}/` with agent name prefix. Shared SNN daemon receives stimuli via JSON files in `04_ARCANE_SAPIENCE/snn_stimuli/`.

**Recommended enhancements based on research:**
1. **Adopt hybrid memory model:** Global knowledge graph (shared) + agent-local session traces (distributed)
2. **Add version chain to entities:** Track which agent created/modified each entity node, with timestamps
3. **Implement write-through protocol:** Agent writes to local session log -> consolidation engine merges to shared graph -> other agents see updated graph on next query
4. **Conflict resolution policy:** When agents disagree on entity attributes, use temporal precedence (most recent wins) + confidence scoring + optional escalation to human review

- **Sources:**
  - [Multi-Agent Memory Architecture (arXiv 2603.10062)](https://arxiv.org/abs/2603.10062)
  - [KARMA (NeurIPS 2025)](https://arxiv.org/abs/2502.06472)
  - [MemOS (arXiv 2507.03724)](https://arxiv.org/abs/2507.03724)
  - [Memory in LLM-based MAS (TechRxiv)](https://www.researchgate.net/publication/398392208)

---

## 29. Proposition Extraction from Text

### 29.1 Dense X Retrieval: Propositions as Retrieval Unit (EMNLP 2024)

**Definition:** A proposition is an atomic expression within text, each encapsulating a distinct factoid, presented in a concise, self-contained natural language format.

**Three properties of a proposition:**
1. **Atomic:** Cannot be further split into separate propositions
2. **Self-contained:** Includes all necessary context (resolved coreferences, decontextualized)
3. **Minimal:** Represents exactly one distinct meaning

**Example decomposition:**
```
Input: "Albert Einstein developed the theory of relativity while working
        at the patent office in Bern."

Propositions:
1. "Albert Einstein developed the theory of relativity."
2. "Albert Einstein worked at the patent office."
3. "The patent office is located in Bern."
4. "Albert Einstein developed the theory of relativity while working
    at the patent office."
```

### 29.2 Propositionizer Model (HuggingFace)

**Model:** `chentong00/propositionizer-wiki-flan-t5-large`
**Architecture:** Fine-tuned FlanT5-Large (seq2seq)
**Training data:** 42K passages decomposed into propositions by GPT-4 (1-shot distillation)
**Input:** Raw text passage
**Output:** List of atomic, self-contained propositions

**Training pipeline:**
1. Prompt GPT-4 with proposition definition + 1-shot demo to decompose passages
2. Generate 42K passage-proposition pairs as seed training data
3. Fine-tune FlanT5-Large as "student" Propositionizer
4. Student model runs at a fraction of GPT-4's cost

### 29.3 FactoidWiki Dataset

English Wikipedia dump indexed at proposition level. Compatible with pyserini for Faiss VectorDB encoding. Enables proposition-level retrieval with any dense retriever.

### 29.4 LlamaIndex Implementation

```python
from llama_index.llama_pack import download_llama_pack

DenseXRetrievalPack = download_llama_pack("DenseXRetrievalPack", "./dense_pack")
documents = SimpleDirectoryReader("./data").load_data()
dense_pack = DenseXRetrievalPack(documents)
response = dense_pack.run("What did Einstein develop?")
```

LangChain implementation uses multi-vector indexing: LLM generates de-contextualized propositions, each vectorized separately, linked back to parent document.

### 29.5 Performance Impact

Proposition-level indexing vs passage-level:
- Significant improvement in retrieval precision and recall
- +10.1 Recall@20 improvement (from Round 1 findings)
- Better downstream QA performance within fixed computation budget
- Trade-off: more vectors to store and search (3-5x more than passage-level)

### 29.6 Remanentia Application

For Remanentia's consolidation pipeline, proposition extraction is the ideal "input preprocessing" step:
1. Raw session text -> propositions (via Propositionizer or simple sentence splitting + decontextualization)
2. Each proposition -> entity extraction (GLiNER/REBEL)
3. Entity pairs -> relation extraction
4. Relations -> temporal annotation
5. Annotated triples -> entity graph update

The propositionizer model runs locally on CPU (FlanT5-Large is ~780M params), no API calls needed. For lighter-weight alternative: sentence splitting + coreference resolution + simple decontextualization heuristics.

- **Sources:**
  - [Dense X Retrieval Paper (EMNLP 2024)](https://aclanthology.org/2024.emnlp-main.845/)
  - [Propositionizer Model (HuggingFace)](https://huggingface.co/chentong00/propositionizer-wiki-flan-t5-large)
  - [FactoidWiki GitHub](https://github.com/chentong0/factoid-wiki)
  - [LlamaIndex Dense X Retrieval](https://clusteredbytes.pages.dev/posts/2024/llamaindex-dense-x-retrieval/)

---

## 30. Memory Evaluation Metrics and Benchmarks

### 30.1 LongMemEval (ICLR 2025) — Five Core Abilities

| Ability | Code | Description | Example |
|---------|------|-------------|---------|
| Information Extraction | IE | Recall specific facts from history | "What restaurant did I mention last Tuesday?" |
| Multi-Session Reasoning | MR | Synthesize across multiple sessions | "How many different hobbies have I mentioned?" |
| Temporal Reasoning | TR | Reason about time, order, intervals | "What was I working on before I started the Remanentia project?" |
| Knowledge Updates | KU | Track changes in personal information | "Where do I live now?" (after reporting a move) |
| Abstention | ABS | Correctly answer "I don't know" | "What is my cat's name?" (never mentioned) |

**Dataset scale:** 500 curated questions. Two configs:
- LongMemEval_S: ~115K tokens (manageable context)
- LongMemEval_M: up to 1.5M tokens, 500 sessions (extreme scale)

**Key finding:** Commercial chat assistants show 30% accuracy drop on sustained interactions. GPT-4o achieves only 30-70% across abilities.

**Framework decomposition:**
1. **Indexing:** Session decomposition for value granularity + fact-augmented key expansion
2. **Retrieval:** Time-aware query expansion to refine search scope
3. **Reading:** LLM processes retrieved context to generate answer

### 30.2 LOCOMO (ACL 2024) — Long-Term Conversational Memory

**Five question types:**
- Single-hop: intra-session recall
- Multi-hop: cross-session synthesis
- Temporal: date/order/interval inference
- Open-domain: integrate speaker info with world knowledge
- Adversarial: unanswerable questions (test refusal)

**Dataset:** Up to 32 sessions, ~600 turns (~16K tokens), 6-12 month timespan. Dialogues seeded with multi-sentence personas and causal/temporally organized event graphs (up to 25 events).

**Evaluation metrics:**
- QA: Token-overlap F1 between system and reference answers
- Event summarization: FactScore (primary) + ROUGE (surface)
- Dialogue generation: BLEU, ROUGE-L, MMRelevance

**RAG baseline findings:** RAG most effective when conversations stored as "observations" (assertions about speaker's life) rather than raw dialogue or session summaries.

**Neuro-symbolic results:** GPT-4o achieves ~30% on temporal tasks; full neuro-symbolic approaches reach ~78%.

### 30.3 MemBench (ACL Findings 2025) — Multi-Aspect Memory Evaluation

Tests seven memory mechanisms:
1. FullMemory (entire history in context)
2. RetrievalMemory (RAG-based)
3. RecentMemory (sliding window)
4. GenerativeAgent (LLM-generated summaries)
5. MemoryBank (structured storage)
6. MemGPT (self-editing)
7. SCMemory (self-controlled)

Evaluates effectiveness, efficiency, and capacity on factual vs reflective memory tasks in participation vs observation scenarios.

### 30.4 MemoryAgentBench (ICLR 2026) — Incremental Multi-Turn

Four core competencies:
1. Accurate retrieval
2. Test-time learning
3. Long-range understanding
4. **Selective forgetting** — crucial for Remanentia's decay mechanism

Transforms long-context datasets into multi-turn format simulating incremental information processing.

### 30.5 MEMTRACK (2025) — Multi-Platform Dynamic Environments

Tests memory across Slack, Linear, and Git platforms with:
- Chronologically interleaved timelines
- Conflicting information
- Cross-references between platforms

**Key finding:** GPT-5 achieves only 60% Correctness. Existing memory systems (Zep, Mem0) do not significantly improve performance. Highlights fundamental limitations of current approaches.

### 30.6 MAGMA Results on Memory Benchmarks (January 2026)

MAGMA (Multi-Graph Agentic Memory Architecture) tested with four orthogonal graph types:
- Up to 45.5% higher reasoning accuracy on long-context benchmarks
- 95%+ token consumption reduction (0.7K-4.2K tokens per query vs 100K+ baseline)
- 40% faster query latency
- Intent-aware traversal: "Why" queries bias toward causal edges, "When" queries emphasize temporal chains

### 30.7 Benchmark Strategy for Remanentia

**Phase 1 targets:**
- LOCOMO: Compete with Mem0's scores on QA tasks. Our SNN-based retrieval should excel at temporal questions.
- LongMemEval_S: Focus on temporal reasoning (TR) and knowledge updates (KU) — our temporal entity graph and edge invalidation should outperform flat-memory systems.

**Phase 2 targets:**
- MEMTRACK: Multi-platform test where our multi-agent architecture (Claude/Codex/Gemini shared memory) gives us a structural advantage.
- MemoryAgentBench: Selective forgetting competency maps directly to our decay mechanism.

**Metrics to report:**
- Token-overlap F1 (LOCOMO QA)
- Accuracy per ability (LongMemEval 5 abilities)
- Token consumption per query (efficiency)
- Latency P95 (production readiness)

- **Sources:**
  - [LongMemEval (ICLR 2025)](https://arxiv.org/abs/2410.10813)
  - [LongMemEval GitHub](https://github.com/xiaowu0162/LongMemEval)
  - [LOCOMO (ACL 2024)](https://snap-research.github.io/locomo/)
  - [MemBench (ACL Findings 2025)](https://arxiv.org/abs/2506.21605)
  - [MemoryAgentBench (ICLR 2026)](https://arxiv.org/abs/2507.05257)
  - [MEMTRACK](https://arxiv.org/abs/2510.01353)
  - [MAGMA (arXiv 2601.03236)](https://arxiv.org/abs/2601.03236)

---

## 31. MAGMA — Multi-Graph Agentic Memory Architecture (January 2026)

### 31.1 Four Orthogonal Graph Types

Each memory event-node is connected through four specialized relation types:

| Graph | Edge Semantics | Construction | Query Bias |
|-------|---------------|--------------|------------|
| Temporal | Strictly ordered (tau_i < tau_j) | Immutable chronological chain | "When" queries |
| Semantic | Undirected, cosine sim > theta | Embedding-based similarity | "What is similar to" queries |
| Entity | Events linked to abstract entity nodes | NER extraction | "Who/what" queries |
| Causal | Directed, S(n_j given n_i, q) > delta | LLM-inferred via consolidation | "Why" queries |

### 31.2 Event-Node Schema

Each event-node is a tuple: (content c_i, timestamp tau_i, embedding v_i in R^d, attributes A_i).

The unified multigraph: G_t = (N_t, E_t) where edges are partitioned into four semantic subspaces.

### 31.3 Dual-Stream Memory Construction

**Fast path (real-time):** Segment events, update temporal backbone, index vectors, enqueue for consolidation. Latency-critical.

**Slow path (asynchronous):** Dequeue events, gather 2-hop neighborhoods, format LLM prompts, invoke reasoning, add inferred causal edges. Compute-intensive.

This dual-stream architecture separates latency-critical ingestion from compute-intensive structural refinement — exactly what Remanentia needs.

### 31.4 Policy-Guided Traversal

Heuristic beam search with dynamic transition scoring:
```
S(n_j | n_i, q) = exp(lambda_1 * phi(type(e_ij), T_q) + lambda_2 * sim(v_j, v_q))
```

Where:
- phi(type(e_ij), T_q) = structural alignment (edge type vs query intent)
- sim(v_j, v_q) = semantic affinity (embedding similarity)
- T_q in {Why, When, Entity} = classified query intent

### 31.5 Multi-Stage Ranking

1. Decompose query into intent type, temporal window, embeddings
2. Reciprocal Rank Fusion across vector search + keyword matching + temporal filtering -> anchor nodes
3. Beam search from anchors using transition scoring equation
4. Topological sort + token budgeting for final context assembly

### 31.6 Remanentia Relevance

MAGMA is the closest existing architecture to what Remanentia should become. Key parallels:
- Four graph types (temporal, semantic, causal, entity) vs our current single entity graph -> we should decompose into orthogonal views
- Dual-stream construction (fast/slow) maps to our "immediate storage + background consolidation" pipeline
- Intent-aware traversal is exactly what our `memory_recall.py` needs — classify the query, bias toward the right graph type
- Causal graph inferred asynchronously via LLM — we can do this via SNN pattern detection instead

Key differences favoring Remanentia:
- MAGMA requires LLM for causal inference; our SNN can detect causal patterns from spike timing correlations
- MAGMA has no consolidation/decay; our hippocampal replay mechanism provides biological forgetting
- MAGMA has no novelty detection; our surprise-weighted consolidation is unique

- **Sources:**
  - [MAGMA Paper (arXiv 2601.03236)](https://arxiv.org/abs/2601.03236)
  - [MAGMA HTML](https://arxiv.org/html/2601.03236v1)

---

## 32. A-Mem: Zettelkasten-Inspired Agentic Memory (NeurIPS 2025)

### 32.1 Core Concept

A-Mem applies the Zettelkasten note-taking method to LLM agent memory:
- Each memory is an atomic "note" following the atomicity principle
- Notes are dynamically linked to form interconnected knowledge networks
- The agent itself decides when and how to organize memory (agentic approach)

### 32.2 Note Schema

Each memory note contains:
- Raw content
- Timestamp
- LLM-generated keywords
- Tags (categorical labels)
- Context description
- Dense embedding vector
- Links (initially empty, populated during organization)

### 32.3 Key Innovation

Combines structured Zettelkasten organization (note-linking, tagging, cross-referencing) with agent-driven decision making. The agent autonomously:
- Decides what to remember
- Generates structural metadata (keywords, tags)
- Creates cross-links between related notes
- Reorganizes the network as new information arrives

### 32.4 Remanentia Relevance

The Zettelkasten analogy maps well to our JSONL-based memory:
- Our reasoning traces are "notes" with timestamps and content
- Our entity graph edges are "links" between notes
- We could adopt the tag/keyword generation as a lightweight metadata enrichment step
- The key difference: A-Mem uses LLM for organization; we use SNN for consolidation

- **Sources:**
  - [A-Mem (NeurIPS 2025)](https://arxiv.org/abs/2502.12110)
  - [A-Mem GitHub](https://github.com/agiresearch/A-mem)

---

## 33. Lightweight Entity Extraction Comparison Matrix

| Tool | Params | GPU Required | NER | RE | EL | Zero-Shot | Speed | Best For |
|------|--------|-------------|-----|----|----|-----------|-------|----------|
| spaCy (en_core_web_lg) | 560M | No | Yes | Via dependency parsing | No | No | Fast | SVO triple extraction |
| GLiNER | <500M | No | Yes | GLiNER2 only | No | Yes | Fast | Custom entity types |
| REBEL | 406M (BART-large) | Optional | Yes | Yes (200+ types) | Yes | No | Medium | End-to-end triples |
| ReLiK-small | ~350M | No | Yes | Yes | Yes | No | Fast | Combined EL+RE |
| ReLiK-large | ~770M | Optional | Yes | Yes | Yes | No | Medium | Highest accuracy |
| Propositionizer | 780M (FlanT5-L) | Optional | No | No | No | No | Medium | Atomic fact decomposition |

**Recommended Remanentia stack (CPU-only):**
1. Propositionizer or sentence splitting -> atomic facts
2. GLiNER -> custom entity detection (project, person, concept, decision, bug, etc.)
3. spaCy dependency parsing -> SVO relation extraction
4. Regex patterns -> causal relation detection
5. TF-IDF cosine -> entity deduplication
6. JSONL -> graph storage

**Upgrade path (with GPU or LLM):**
1. Add REBEL for 200+ relation types
2. Add ReLiK for entity linking to Wikidata
3. Add LLM-based gleaning for missed entities
4. Add causal-LLM one-shot for causal graph extraction

---

## Round 4 Cross-Cutting Implementation Synthesis

### What to Build Next in Remanentia's Consolidation Pipeline

Based on this research, the highest-impact implementation priorities are:

**Priority 1: Adopt Graphiti's Bitemporal Edge Model**
Add `t_valid` and `t_invalid` fields to every entity relationship in our JSONL graph. This is the single most impactful change for temporal reasoning — it's what makes Zep best-in-class on temporal benchmarks, and it's a schema change, not an algorithm change.

**Priority 2: Implement GLiNER-Based Entity Extraction**
Replace or supplement our current heuristic NER with GLiNER for zero-shot entity detection. Custom entity types (project, decision, bug, person, concept) without training data. Runs on CPU. Immediate improvement in entity coverage.

**Priority 3: Add MAGMA-Style Graph Decomposition**
Decompose our single entity graph into four orthogonal views: temporal, semantic, causal, entity. The query intent classifier then biases traversal toward the right view. This is the architectural pattern that achieves 45.5% higher reasoning accuracy.

**Priority 4: Implement Proposition-Level Preprocessing**
Before entity extraction, decompose session text into atomic propositions. Either via Propositionizer (local model) or simple sentence splitting + decontextualization. This improves entity extraction recall by ensuring each fact is isolated.

**Priority 5: Add Causal Edge Extraction**
Pattern-match causal connectives in session text ("because", "led to", "resulted in"). Store as directed edges with confidence scores. Enable "why" queries against the causal subgraph.

**Priority 6: Benchmark on LOCOMO and LongMemEval**
Implement evaluation harness for both benchmarks. Focus on temporal reasoning and knowledge update tasks where our architecture has theoretical advantages.

---

## Sources (Round 4 — All URLs)

### Zep/Graphiti
- [Zep Paper (arXiv 2501.13956)](https://arxiv.org/abs/2501.13956)
- [Zep Paper HTML](https://arxiv.org/html/2501.13956v1)
- [Graphiti GitHub](https://github.com/getzep/graphiti)
- [Graphiti Neo4j Blog](https://neo4j.com/blog/developer/graphiti-knowledge-graph-memory/)
- [Graphiti Custom Entity Types](https://help.getzep.com/graphiti/core-concepts/custom-entity-and-edge-types)
- [Graphiti Adding Episodes](https://help.getzep.com/graphiti/core-concepts/adding-episodes)

### Microsoft GraphRAG
- [GraphRAG Paper (arXiv 2404.16130)](https://arxiv.org/abs/2404.16130)
- [GraphRAG Docs — Dataflow](https://microsoft.github.io/graphrag/index/default_dataflow/)
- [GraphRAG GitHub](https://github.com/microsoft/graphrag)
- [GraphRAG Auto-Tuning](https://www.microsoft.com/en-us/research/blog/graphrag-auto-tuning-provides-rapid-adaptation-to-new-domains/)
- [HIT-Leiden Incremental Community Detection](https://github.com/randomvariable/hit-leiden)

### Temporal Knowledge Graphs
- [TKG Survey (arXiv 2403.04782)](https://arxiv.org/abs/2403.04782)
- [TKG Embedding Survey (Knowledge-Based Systems)](https://www.sciencedirect.com/science/article/abs/pii/S0950705124010888)
- [TKGER Paper List (GitHub)](https://github.com/stmrdus/tkger)

### Causal Extraction
- [Zero-shot Causal Graph (arXiv 2312.14670)](https://arxiv.org/abs/2312.14670)
- [Causal-LLM (EMNLP 2025)](https://aclanthology.org/2025.findings-emnlp.439/)
- [Event Causality Survey (ACM)](https://dl.acm.org/doi/10.1145/3756009)
- [LLMs for Causal Discovery Survey](https://arxiv.org/abs/2402.11068)
- [Causality NLP Paper List (GitHub)](https://github.com/zhijing-jin/CausalNLP_Papers)

### Lightweight Entity Extraction
- [REBEL (EMNLP 2021, GitHub)](https://github.com/Babelscape/rebel)
- [ReLiK (ACL 2024)](https://arxiv.org/abs/2408.00103)
- [GLiNER (NAACL 2024)](https://arxiv.org/abs/2311.08526)
- [GLiNER GitHub](https://github.com/urchade/GLiNER)
- [Efficient KG Construction (arXiv 2507.03226)](https://arxiv.org/abs/2507.03226)
- [KG Without LLMs (AIHello)](https://www.aihello.com/resources/blog/creating-a-knowledge-graph-without-llms/)

### Multi-Agent Memory
- [Multi-Agent Memory Architecture (arXiv 2603.10062)](https://arxiv.org/abs/2603.10062)
- [KARMA (NeurIPS 2025)](https://arxiv.org/abs/2502.06472)
- [MemOS (arXiv 2507.03724)](https://arxiv.org/abs/2507.03724)
- [MemOS GitHub](https://github.com/MemTensor/MemOS)
- [Agent Memory Paper List (GitHub)](https://github.com/Shichun-Liu/Agent-Memory-Paper-List)

### Proposition Extraction
- [Dense X Retrieval (EMNLP 2024)](https://aclanthology.org/2024.emnlp-main.845/)
- [Propositionizer (HuggingFace)](https://huggingface.co/chentong00/propositionizer-wiki-flan-t5-large)
- [FactoidWiki (GitHub)](https://github.com/chentong0/factoid-wiki)
- [LlamaIndex Dense X Retrieval](https://clusteredbytes.pages.dev/posts/2024/llamaindex-dense-x-retrieval/)

### Memory Benchmarks
- [LongMemEval (ICLR 2025)](https://arxiv.org/abs/2410.10813)
- [LongMemEval GitHub](https://github.com/xiaowu0162/LongMemEval)
- [LOCOMO (ACL 2024)](https://snap-research.github.io/locomo/)
- [MemBench (ACL Findings 2025)](https://arxiv.org/abs/2506.21605)
- [MemoryAgentBench (ICLR 2026)](https://arxiv.org/abs/2507.05257)
- [MemoryBench (arXiv 2510.17281)](https://arxiv.org/abs/2510.17281)

### Agentic Memory Systems
- [MAGMA (arXiv 2601.03236)](https://arxiv.org/abs/2601.03236)
- [A-Mem (NeurIPS 2025)](https://arxiv.org/abs/2502.12110)
- [A-Mem GitHub](https://github.com/agiresearch/A-mem)
- [Mem0 Paper (arXiv 2504.19413)](https://arxiv.org/abs/2504.19413)
- [Mem0 Graph Memory Docs](https://docs.mem0.ai/open-source/features/graph-memory)
- [LightRAG (EMNLP 2025)](https://github.com/HKUDS/LightRAG)

---

*Round 4 compiled by Arcane Sapience for Remanentia development. Implementation-focused deep dive. All sources verified as of 2026-03-20.*
