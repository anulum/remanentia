# Holographic Memory in Spiking Kuramoto Networks

**Date:** 2026-03-19
**Authors:** Miroslav Šotek, Arcane Sapience
**Status:** Investigation → Prototype → Benchmark → Paper

## Motivation

Remanentia's current memory system uses STDP weight updates on a LIF network.
The weight matrix W becomes dense positive over time, causing static projection
cos(Wx, Wy) to saturate to 1.0 for all query pairs (0% P@1). Dynamic spike
matching rescues retrieval (79% P@1) but the SNN's marginal value over
TF-IDF + embeddings is unmeasured.

The question: can we encode memories holographically — distributed across all
weights rather than localized — so that retrieval uses phase synchronization
rather than spike pattern matching?

## Core Idea

In the Kuramoto equation:

    dθ_i/dt = ω_i + Σ_j K_ij sin(θ_j - θ_i - α_ij) + ζ sin(Ψ - θ_i)

The **phase lag matrix α** determines the preferred relative phase between every
pair of oscillators. Each α_ij contributes to the dynamics of all oscillators.
This is a distributed (holographic) encoding: every memory is spread across the
entire α matrix, and every α_ij participates in every memory.

**Storage:** Encode memory m as a phase pattern φ_m. Update α via outer-product
superposition: α += φ_m ⊗ φ_m^T (Hopfield-like, but in phase-lag space).

**Retrieval:** Inject a cue pattern as initial phases θ(0). Run the Kuramoto
dynamics. The system synchronizes to the stored pattern closest to the cue
(chimera state: retrieved cluster syncs, others drift). Read out the
synchronized cluster.

**Capacity:** Bounded by coherence budget — the number of superimposed patterns
before inter-pattern interference degrades the order parameter R below threshold.
Theoretical limit: ~0.14N for N oscillators (Hopfield bound). For N=20,000,
that's ~2,800 distinct memories.

## Existing Infrastructure

All pieces already exist across the SCPN ecosystem:

### scpn-phase-orchestrator (v0.4.1)
- Kuramoto solver with phase lags α: `upde/engine.py` (RK45, Dormand-Prince)
- Stuart-Landau amplitude dynamics: μ > 0 self-sustains, μ < 0 decays
  (memory strength modulation, `identity_coherence/run.py`)
- Chimera states validated: `test_physics_benchmarks.py:306`
  (partial sync = selective recall)
- P/I/S three-channel encoding: `oscillators/init_phases.py`
  Physical (Hilbert), Informational (event), Symbolic (ring)
- Family-aware binding: `binding/types.py`, explicit oscillator-to-channel mapping
- Imprint persistence: saves α + K modulations to npz across sessions
- Rust engine parity: `spo-kernel/` (UPDE + Stuart-Landau)
- Order parameter: R = |mean(exp(iθ))| via `upde/order_params.py`

### scpn-quantum-control (v0.9.0)
- XY Hamiltonian mapping: K_ij sin(θ_j - θ_i) ↔ -J_ij(X_iX_j + Y_iY_j)
  (`bridge/knm_hamiltonian.py`)
- Coherence budget: fidelity vs depth bounds (`identity/coherence_budget.py`)
  Gives capacity limits for quantum-encoded memories
- QuantumKuramotoSolver: Trotter-decomposed XY evolution (`phase/xy_kuramoto.py`)
- Order parameter from qubit expectations: R*exp(iψ) = (1/N)Σ(<X_j> + i<Y_j>)
- Rust Kuramoto engine: ~100x speedup (`scpn_quantum_engine/src/lib.rs`)
- build_kuramoto_ring: generic ring topology for arbitrary N

### Remanentia / 04_ARCANE_SAPIENCE
- GPU SNN daemon: 20K LIF neurons on GTX 1060 (`gpu_daemon.py`)
- Embedding encoding: sentence-transformers MiniLM (`encoding.py`)
- Hybrid retrieval: TF-IDF + SNN + name + embedding (`retrieve.py`)
- Dynamic spike matching: 79% SNN-only P@1 (`paper/remanentia_paper_draft.md`)
- Memory replay: random trace re-injection every 3 cycles
- Dashboard: live monitoring at port 8888 (`monitor.py`)

## What's New (the gap this fills)

Nobody has:
1. Used Kuramoto phase lags (α) as a writable holographic memory medium
2. Combined chimera-state selective recall with SNN spike dynamics
3. Provided analytical capacity bounds for spiking memory networks
   via coherence budget transfer from quantum information theory
4. Benchmarked holographic phase encoding against conventional
   vector retrieval on the same dataset

## Experimental Plan

### Phase 1: Prototype (holographic_memory.py)
- Implement Kuramoto-based memory layer alongside existing LIF network
- Storage: text → embedding → phase pattern φ → α outer-product update
- Retrieval: inject φ_query as θ(0) → run Kuramoto → measure R per cluster
  → return highest-R cluster as retrieved memory
- Use orchestrator's UPDE engine for dynamics (or Rust engine for speed)
- N = 2,000 oscillators initially (matches original LIF daemon size)

### Phase 2: Benchmark (14-query retrieval test)
- Same 14 queries from paper experiments
- Measure: holographic P@1, holographic negative rejection
- Compare: holographic-only vs SNN-only (79%) vs TF-IDF+embedding-only vs hybrid
- This is the ablation that was never run

### Phase 3: Capacity Analysis
- Store M memories, sweep M from 1 to 5,000
- At each M, measure retrieval P@1 on all stored memories
- Find the critical M where P@1 drops below 50% (interference threshold)
- Compare measured capacity to theoretical 0.14N bound
- Transfer coherence budget formalism: define "memory fidelity" analogous
  to circuit fidelity, compute maximum encoding depth

### Phase 4: Paper
- Title: "Holographic Memory in Spiking Kuramoto Networks
  with Analytically Bounded Capacity"
- Key contribution: α as memory medium + chimera recall + capacity bounds
- Negative result preserved: static W projection fails (0% P@1)
- Positive result: holographic α encoding (measured P@1 from Phase 2)
- Capacity curve from Phase 3
- Connection to quantum XY model as theoretical backing
- Targets: arXiv preprint → Zenodo DOI → journal submission

## Key Equations

### Memory Storage (outer-product rule)
    α_new = α_old + (1/N) · φ_m ⊗ φ_m^T

### Memory Retrieval (Kuramoto dynamics)
    θ(0) = φ_query
    dθ_i/dt = ω_i + Σ_j K_ij sin(θ_j - θ_i - α_ij)
    Run until convergence (R > threshold or max_steps)
    Retrieved pattern = θ_final (phases of synchronized cluster)

### Capacity Bound (Hopfield transfer)
    C_max ≈ 0.14 · N  (for random binary patterns)
    C_max ≈ N / (2 · ln(N))  (for continuous patterns, tighter bound)

### Coherence Budget Transfer
    Memory fidelity F(M) = (1 - M·ε)^N  where ε = inter-pattern interference
    Maximum memories: M* such that F(M*) ≥ F_threshold

## Experiment Log

### Run 1 — Naive outer-product encoding (2026-03-19)

**Setup:** N=2000 oscillators, K=2.0 (global), all-to-all coupling,
full-document text → embedding → clustered phase pattern → outer product α update.

**Results:**

| Method | P@1 | Neg Max |
|--------|-----|---------|
| SNN static (paper) | 0.0% | 0.984 |
| Holographic v1 | 7.1% (1/14) | 0.084 |
| TF-IDF only (ablation) | 64.3% (9/14) | 0.058 |
| SNN dynamic (paper) | 78.6% (11/14) | 0.175 |
| Hybrid TF-IDF+SNN (paper) | 92.9% (13/14) | — |

**TF-IDF ablation note:** TF-IDF alone gets 64.3%, not 92.9%. The paper's
hybrid number includes SNN discrimination. SNN dynamic alone (78.6%) beats
TF-IDF alone (64.3%). This confirms the SNN adds real value — 14 percentage
points over TF-IDF.

**Holographic failure analysis:**
- All retrieval scores cluster in 0.04–0.09 — no discrimination between memories
- Kuramoto converges to global sync (R ≈ 0.71) not memory-specific chimera
- Every query produces nearly identical final phase configuration
- Negative rejection is good (0.084) — system doesn't hallucinate memories

**Root causes:**
1. Phase patterns are too dense — all 2000 oscillators active per memory
2. Long document embeddings are correlated — similar α patterns for all traces
3. All-to-all coupling prevents chimera formation (needs nonlocal ring)
4. Global K=2.0 is too strong — forces global sync instead of partial sync

### Run 2 — Sparse + decorrelated + ring topology (2026-03-19)

**Changes from Run 1:**
- Sparse phase patterns: 10% activation (200/2000 oscillators active per memory)
- Random projection decorrelation: embedding → random matrix → top-k selection
- Nonlocal ring coupling: radius N/7 (~286 neighbors) instead of all-to-all
- Weaker coupling: K=0.5 (was 2.0)
- Wider natural frequency spread: sigma=0.3 (was 0.1)

**Results:**

| Method | P@1 | Neg Max |
|--------|-----|---------|
| Holographic v2 | 50.0% (7/14) | 0.629 |

**Improvement:** P@1 jumped from 7.1% to 50.0%. The system now discriminates
between memories — sparse encoding + ring topology prevent global sync collapse.

**Remaining problem:** Negative rejection is bad (0.629). Irrelevant queries
("pizza", "puppy training") score almost as high as correct matches (0.63-0.72).
All scores compressed into narrow 0.62-0.72 band. The system can discriminate
between stored memories but cannot reject non-memories.

**Root cause:** Scoring via mean(cos(theta_final - phi_stored)) has a high
baseline for any pair of phase vectors. The discrimination is in the delta,
not the absolute value. Need a scoring method that measures active cluster
overlap specifically, not global phase correlation.

### Run 3 — Active-oscillator scoring (2026-03-19)

**Change:** Score using local order parameter R of only the oscillators that
were active in the stored pattern (|exp(i*(theta - phi))| on active subset).

**Result:** 0% P@1, neg max 0.868. Worse than Run 2.

All scores 0.75-0.84. The local R on ANY subset is high because the Kuramoto
dynamics synchronize ALL oscillators, not just the queried ones. The ring
topology slows global sync but doesn't prevent it over 300 steps.

**Fundamental issue:** The current approach runs Kuramoto to convergence and
then scores the final state. But Kuramoto convergence destroys the pattern-
specific information — everything syncs to the same attractor regardless of
initial conditions (for these parameter ranges).

**What's needed to make this work:**
1. The dynamics must have MULTIPLE attractors (one per stored memory), not one
2. The initial condition (query) must select which attractor basin to fall into
3. The scoring must distinguish between attractor basins

This is the Hopfield network regime: multiple minima, content-addressable.
The Kuramoto equivalent requires either:
- Multi-stable chimera states (different coupling topologies per memory)
- Frustrated coupling (competing alpha terms) that creates rugged energy landscape
- Or abandon continuous Kuramoto and use discrete phase states (Potts model)

### Run 4 — Baseline-subtracted scoring (2026-03-19)

**Change:** Score = mean_cos(active oscillators) - mean_cos(inactive oscillators).
Subtracts the baseline alignment that ALL patterns share, isolating the
memory-specific signal.

**Result:** 50% P@1, neg max -1.45.

Same P@1 as Run 2, but negative rejection is now excellent. Irrelevant queries
score deeply negative (-1.45 to -1.47). Gap between matches and non-matches
is clear. The scoring fix solved rejection without solving discrimination.

**Progress table:**

| Run | Encoding | Topology | K | Scoring | P@1 | Neg Max |
|-----|----------|----------|---|---------|-----|---------|
| 1 | dense | all-to-all | 2.0 | mean cos | 7.1% | 0.084 |
| 2 | sparse 10% | ring N/7 | 0.5 | mean cos | 50.0% | 0.629 |
| 3 | sparse 10% | ring N/7 | 0.5 | active-only R | 0.0% | 0.868 |
| 4 | sparse 10% | ring N/7 | 0.5 | baseline-sub | 50.0% | -1.45 |

### Architecture Variant Sweep (2026-03-19)

6 variants tested on same alpha matrix to isolate information loss:

| Variant | P@1 | Neg Max | Interpretation |
|---------|-----|---------|----------------|
| V0 Kuramoto converged | 50.0% | -1.45 | baseline |
| V1 Alpha matmul | 35.7% | 0.89 | alpha loses info |
| V2 Transient (20 steps) | 50.0% | -1.50 | no transient advantage |
| V3 Frustrated alpha | 35.7% | -0.65 | frustration hurts |
| V4 Two-stage | 0.0% | 1.13 | catastrophic |
| V5 Direct embedding | 50.0% | 0.19 | NULL HYPOTHESIS TIES |

**Critical finding:** V5 (raw embedding cosine, no alpha, no dynamics) matches
the best holographic variant at 50% P@1. The Kuramoto dynamics and alpha
storage add ZERO value over raw sentence similarity. The 50% ceiling is
the embedding model's discrimination limit for these query lengths.

**Why SNN dynamic (78.6%) beats everything:** It processes full document content
through STDP-modified connectivity, not just short query embeddings. The spike
patterns capture document-level structure that 384-dim MiniLM can't encode
in a short query.

**Implications for holographic approach:**
1. The outer-product alpha storage is lossy — V1 proves it
2. The Kuramoto dynamics don't extract more signal — V0 = V2 = V5
3. The embedding bottleneck is the real limit — need better encoding
4. The SNN approach works BECAUSE it uses the full document, not because
   of anything special about spikes

### Holographic v2 — Complex Hebbian (2026-03-19, post-Codex)

**Codex identified three bugs in v1:**
1. `alpha += phi * phi^T` stores products, not phase differences. Fixed: use
   `J += xi * conj(xi)^T` where `xi = a * exp(i*phi)`, then `K=|J|, alpha=arg(J)`
2. Inactive = phase 0 creates false coherent block. Fixed: random phase for inactive.
3. Global R is wrong metric. Fixed: per-memory modal overlap `m_mu = |<xi^mu, z(t)>|`

**Result:** 42.9% P@1, neg max 0.263.

Worse than v1 (50%) and the null hypothesis (50%). But the dynamics now show
query-specific behavior: peak steps vary (0, 18, 42, 77) instead of fixed 22.
The complex J creates some differentiation but coupling is too weak (K_max=0.005)
and too sparse (84% zeros) to reliably pull phases into basins.

**Full progression:**

| Version | Approach | P@1 | Key Issue |
|---------|----------|-----|-----------|
| v1 Run 1 | Dense outer product, all-to-all | 7.1% | Global sync collapse |
| v1 Run 2 | Sparse + ring | 50.0% | = null hypothesis |
| v1 Run 4 | + baseline-subtracted scoring | 50.0% | Good rejection |
| v2 | Complex Hebbian J, modal overlap | 42.9% | Too sparse, weak coupling |
| Null | Direct embedding cosine | 50.0% | Embedding ceiling |
| TF-IDF | Keyword matching | 64.3% | No semantic understanding |
| SNN dynamic | STDP + spike patterns | 78.6% | Processes full documents |

**Conclusion:** Holographic Kuramoto memory does not beat the embedding model
or TF-IDF for this task. The SNN dynamic approach succeeds because it processes
full document content through learned connectivity, not because of any property
of spiking dynamics per se. The embedding bottleneck (384-dim MiniLM on short
queries) caps all embedding-based methods at ~50%.

### Shuffled-W Control Experiment (2026-03-19, autonomous)

The critical test: does STDP-learned W structure matter for retrieval?

**Setup:** Fresh 20K GPU network, embedding encoding, 10 STDP cycles,
same 14-query benchmark. Tested with trained W, shuffled W (same values,
random connectivity), zero W, and both probe and LIF scoring.

| Method | P@1 | Neg Max |
|--------|-----|---------|
| Trained W (probe) | 35.7% | 0.9999 |
| Shuffled W (probe) | 50.0% | 0.9909 |
| Zero W (probe) | 42.9% | 0.1376 |
| Trained W (LIF) | 50.0% | 0.1306 |
| Shuffled W (LIF) | 50.0% | 0.1306 |

**Verdict: STDP structure does NOT matter. Shuffled W ties or beats trained W.**

The probe feature (`stim + 0.5 * W @ stim`) saturates with dense 20K W —
all scores near 1.0, no discrimination. The LIF simulation avoids saturation
but shows zero difference between trained and shuffled connectivity.

**The 92.9% P@1 from the live daemon benchmark comes from TF-IDF + filename
+ embedding — not from the SNN component.** The SNN contributes noise.

**What this means for Remanentia:**
- The persistent SNN infrastructure works (daemon, dashboard, state management)
- The retrieval system works at 92.9% (via TF-IDF + embedding + filename)
- But the STDP-modified weight matrix does not encode useful memory structure
- The SNN is a running neural substrate that doesn't yet contribute to retrieval
- Calling it "memory" is premature until the SNN demonstrably outperforms
  a random weight matrix

**Path forward options:**
A. Better embeddings (larger model, document-level encoding)
B. Store full documents in alpha, not query-length embeddings
C. Abandon holographic Kuramoto, improve the SNN approach instead
D. Hybrid: SNN for full-doc encoding + Kuramoto for association/consolidation
E. Investigate WHY STDP doesn't create useful structure at 20K scale —
   is it saturation? Lack of inhibition? Too few cycles? Wrong encoding?

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Phase patterns not orthogonal enough (correlated text embeddings) | High | Reduced capacity | LSH or random projection to decorrelate |
| Kuramoto convergence too slow for retrieval | Medium | Latency | Rust engine, reduce N, tune K |
| Chimera states unstable | Medium | Unreliable recall | Stuart-Landau amplitude stabilization |
| Capacity << theoretical bound | Medium | Limited utility | Sparse coding, dimensional expansion |
| No improvement over vector store | Low-Medium | Kills the paper | Honest reporting — negative result is still publishable |

## References

- Hopfield, J.J. (1982). Neural networks and physical systems with emergent
  collective computational abilities. PNAS 79(8):2554-2558.
- Plate, T.A. (1995). Holographic Reduced Representations. IEEE Trans. Neural
  Networks 6(3):623-641.
- Abrams, D.M. & Strogatz, S.H. (2004). Chimera States for Coupled Oscillators.
  PRL 93(17):174102.
- Acebrón, J.A. et al. (2005). The Kuramoto model: A simple paradigm for
  synchronization phenomena. Rev. Mod. Phys. 77:137-185.
- Amit, D.J. et al. (1985). Storing Infinite Numbers of Patterns in a
  Spin-Glass Model of Neural Networks. PRL 55(14):1530-1533.
- Kanerva, P. (2009). Hyperdimensional Computing. Cognitive Computation 1(2):139-159.
- Pribram, K.H. (1991). Brain and Perception: Holonomy and Structure in
  Figural Processing. Lawrence Erlbaum Associates.
- Yassa, M.A. & Stark, C.E. (2011). Pattern separation in the hippocampus.
  Trends in Neurosciences 34(10):515-525.
