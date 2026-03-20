# Comprehensive SNN Memory Investigation — Findings

**Date:** 2026-03-19 (autonomous run)
**Author:** Arcane Sapience

## Executive Summary

Across 7 experiments and 35+ configurations, the SNN spike feature
contributes NOTHING to retrieval. Every SNN variant scores exactly
50.0% P@1 — identical to the embedding model's baseline. The weight
matrix W, regardless of learning rule, training duration, network size,
sparsity, or E/I balance, does not create discriminative spike patterns.

The retrieval ceiling without any SNN is **92.9% P@1** using
TF-IDF(0.4) + best-paragraph embedding(0.6). This matches the live
daemon's 92.9% — confirming the SNN component adds zero signal.

## Key Discovery: Best-Paragraph Embedding

The single most impactful finding: **best-paragraph cosine similarity
alone achieves 85.7% P@1** (12/14). This is because long traces contain
multiple topics — matching the query against the best paragraph focuses
on the relevant section instead of averaging over the whole document.

Combined with TF-IDF at 0.4/0.6 weighting: **92.9% P@1** (13/14).
No SNN, no spike features, no weight matrix. Just TF-IDF + paragraphs.

## Experiment Results

### Exp 1: Weight Saturation (N=500, 2000, 5000)

| N | W mean | W>0 | W sparsity | P@1 |
|---|--------|-----|-----------|-----|
| 500 | 0.0053 | 4.3% | 95.7% | 50.0% |
| 2000 | 0.0049 | 4.0% | 96.0% | 50.0% |
| 5000 | 0.0049 | 3.9% | 96.1% | 50.0% |

W is 96% sparse (values < 0.01). The saturation problem from the 20K
experiment doesn't apply at 2K-5K. But performance is identical anyway.
**Saturation is not the cause of STDP failure.**

### Exp 2: Sparse vs Dense Encoding

| Sparsity | Active neurons | P@1 |
|----------|---------------|-----|
| 1% | 0.5% | 14.3% |
| 5% | 2.5% | 42.9% |
| 10% | 5.0% | 50.0% |
| 50% | 24.5% | 50.0% |
| 100% | 49.5% | 50.0% |

Too sparse (1%) hurts because there's not enough signal. But above 5%,
performance plateaus at 50% regardless of sparsity.
**Sparsity doesn't help because the bottleneck is the embedding, not W.**

### Exp 3: Inhibitory Connections

| Method | W<0 | P@1 |
|--------|-----|-----|
| STDP only | 0% | 50.0% |
| STDP + anti-Hebbian | 0% | 50.0% |
| E/I balance (80/20) | 0.7% | 50.0% |

Anti-Hebbian barely creates negative weights (the anti-Hebbian rate is
too low relative to STDP). E/I balance creates only 0.7% inhibitory.
Neither changes P@1 at all.
**Inhibition doesn't help because the feature discrimination is flat.**

### Exp 4: Training Duration (1–100 cycles)

| Cycles | W mean | Trained P@1 | Shuffled P@1 |
|--------|--------|------------|-------------|
| 1 | 0.0005 | 50.0% | 50.0% |
| 5 | 0.0025 | 50.0% | 50.0% |
| 10 | 0.0049 | 50.0% | 50.0% |
| 25 | 0.0123 | 50.0% | 50.0% |
| 50 | 0.0236 | 50.0% | 50.0% |
| 100 | 0.0404 | 50.0% | 50.0% |

Trained = shuffled at EVERY cycle count. More training doesn't help.
**STDP never creates discriminative structure at any training stage.**

### Exp 5: Contrastive Learning

W stays at 0.0 after 20 epochs. The contrastive update never triggers
because `sim_neg > sim_pos - 0.1` is already violated — all spike features
are already too similar (cosine > 0.9) for the contrastive margin to
activate gradient updates.
**Contrastive learning can't start because the spike space is already flat.**

### Exp 6: Pure Embedding Retrieval

| Method | P@1 |
|--------|-----|
| Full document cosine | 50.0% (7/14) |
| Best-paragraph cosine | **85.7% (12/14)** |
| TF-IDF only | 64.3% (9/14) |
| TF-IDF(0.7) + Emb(0.3) | 64.3% |
| TF-IDF(0.5) + Emb(0.5) | 57.1% |

Best-paragraph matching is the breakthrough. Full-document embedding
averages over too many topics and dilutes the signal. Paragraph-level
matching focuses on the relevant section.

### Exp 7: TF-IDF + Best-Paragraph

| Weights | P@1 |
|---------|-----|
| TF-IDF(0.2) + Para(0.8) | 85.7% |
| TF-IDF(0.3) + Para(0.7) | 85.7% |
| TF-IDF(0.4) + Para(0.6) | **92.9%** |
| TF-IDF(0.5) + Para(0.5) | **92.9%** |

**92.9% P@1 with just TF-IDF + best-paragraph embedding.** The optimal
mix is 0.4-0.5 TF-IDF + 0.5-0.6 paragraph. No SNN needed.

## Root Cause Analysis

### Why the SNN never works

The spike feature `F_W(x)` always produces the same result regardless of W
because:

1. **The LIF dynamics are dominated by the input current**, not by synaptic
   current from W. With `i_ext = 0.3 + stim * 2.0` and `i_syn = W @ fired * 0.5`,
   the input current is 4x stronger than the synaptic feedback.

2. **At 50 timesteps**, there isn't enough time for recurrent dynamics through
   W to differentiate spike patterns. The first ~10 steps are driven entirely
   by i_ext; W effects are second-order perturbations.

3. **The spike features are high-dimensional (N=2000+) and sparse**, making
   cosine similarity insensitive to small perturbations from W.

4. **The embedding produces patterns where ~5% of neurons are strongly active**.
   These neurons fire in the first few timesteps regardless of W. The remaining
   95% are near threshold and their firing depends on noise, not on learned
   connectivity.

### What would be needed for W to matter

For the weight matrix to create discriminative spike patterns, it would need:
- **Stronger recurrence**: i_syn >> i_ext, so the network dynamics depend on
  learned connectivity rather than input
- **Longer simulation**: hundreds of timesteps for recurrent patterns to emerge
- **Attractor dynamics**: W must create stable firing patterns (attractors)
  that different inputs converge to different attractors
- **Inhibition**: without inhibitory connections, all neurons eventually fire,
  producing similar spike counts regardless of input

This is fundamentally a different regime from the current LIF implementation.
The current system uses the network as a one-pass feature transformer where
W is a mild perturbation. True associative memory requires recurrent attractor
dynamics with strong inhibition.

## Implications

### For the paper
- The SNN retrieval claim (78.6% P@1) was measured on a different pipeline
  that no longer exists. The current pipeline's SNN component adds nothing.
- The honest contribution is: best-paragraph embedding matching (85.7% P@1
  alone) combined with TF-IDF (92.9%) — this is a valid retrieval result.
- The SNN infrastructure (daemon, dashboard, STDP) is a substrate for future
  work, not a current retrieval mechanism.

### For the product
- Retrieval works at 92.9% without any SNN. Ship it.
- The SNN daemon is valuable infrastructure for monitoring, consolidation,
  and the neural animation aesthetic, even if it doesn't improve retrieval.
- Future work should focus on attractor-based networks (Hopfield, modern
  continuous Hopfield with exponential interactions) rather than perturbative
  LIF spike features.

### For the science
- The negative result is publishable: "STDP weight learning in LIF networks
  does not create discriminative spike features for text retrieval at any scale,
  training duration, sparsity level, or learning rule tested"
- The positive finding: paragraph-level embedding matching dramatically
  outperforms document-level matching for heterogeneous trace retrieval
- The architecture investigation (holographic Kuramoto, complex Hebbian,
  shuffled-W controls) is thorough and honest

## Five Approaches to Making W Matter (2026-03-19, post-sleep)

After identifying that the missing dimension is context (not content),
tested 5 architectural approaches for encoding temporal/project/causal
structure into the SNN.

| Approach | Idea | P@1 | Shuffled | Verdict |
|----------|------|-----|---------|---------|
| A1 Context-enriched | Split neurons: content+time+project+type bands | 50.0% | 50.0% | No effect |
| A2 Sequential replay | Chronological STDP instead of random | 50.0% | 50.0% | No effect |
| A3 Dual pathway | SNN as associator after TF-IDF retrieval | 92.9% | — | = base (92.9%) |
| A4 Trace-pair STDP | Learn cross-trace temporal associations | 50.0% | 50.0% | No effect |
| A5 Hippocampal index | Sparse context codes + content-index STDP | 50.0% | 50.0% | No effect |

**Strong-W variant (W dominates input): 7.1% everywhere.** Amplifying W
makes things worse because the learned structure is noise.

**Exhaustive conclusion across 50+ configurations:**
The LIF spike feature is determined by input encoding. W is a perturbation
too weak to change the output, and when amplified, degrades it. No learning
rule, training duration, network size, sparsity level, encoding scheme,
replay strategy, or architectural variant tested creates discriminative
weight structure.

**The fundamental issue:** LIF neurons with STDP are not an associative
memory. They are a noisy feature transform where the input dominates.
For W to matter, the network needs:
1. Attractor dynamics (Hopfield regime, not LIF perturbative regime)
2. Strong recurrence with inhibition (E/I balanced, attractor basins)
3. Fundamentally different architecture (modern Hopfield, Boltzmann machine,
   or reservoir computing with readout training)

The current LIF+STDP approach is exhausted. Moving forward requires a
different computational model, not parameter tuning.
