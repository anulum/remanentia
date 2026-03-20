# Architecture Routes for SNN Memory — Analysis

**Date:** 2026-03-19
**Status:** Deep investigation, pre-experiment

## What We've Proven Doesn't Work

50+ experiments confirmed: LIF + STDP with input-dominated dynamics creates
no discriminative weight structure. The root cause is structural, not parametric:
- Input current >> synaptic current (4:1 ratio)
- 50 timesteps too short for recurrent dynamics to emerge
- No competition between memories (no inhibition)
- STDP strengthens everything uniformly (no selectivity)

## Candidate Architectures

### Route A: Modern Continuous Hopfield (Ramsauer et al., 2021)

**Mechanism:** E = -log(sum exp(beta * xi^mu . x)). Exponential energy creates
sharp basins. Retrieval: x_new = softmax(beta * M * x) * M^T.

**Key insight:** This IS the transformer attention mechanism. Hopfield retrieval =
attention over memory matrix M with temperature 1/beta.

**For Remanentia:** Store traces as rows of M (embeddings). Query = attention
over M. Learnable beta controls sharpness.

**Problem:** This is mathematically equivalent to embedding similarity with
softmax normalization. Our best-paragraph approach already does this manually.
The "learning" is adjusting M and beta — same as fine-tuning embeddings.

**Verdict:** Sound theory but adds nothing over what we have. REJECT as
retrieval mechanism. KEEP as theoretical framing for the paper.

### Route B: E/I Attractor Network

**Mechanism:** 80% excitatory + 20% inhibitory neurons. Strong recurrence
(i_syn >> i_ext). Winner-take-all dynamics: only ONE memory pattern active
at a time. Competition through inhibition creates multiple stable attractors.

**Key insight:** Our current network has no competition. ALL neurons fire
together because there's no inhibition to suppress irrelevant activations.
Adding strong inhibition creates the selectivity STDP alone can't.

**For Remanentia:** Each trace = one attractor basin. Query = partial cue
that selects the correct basin. Inhibition prevents co-activation of
non-matching memories.

**Testable parameters:**
- i_ext * 0.1, i_syn * 3.0 (flip the ratio)
- 80/20 E/I split with Dale's law enforced
- 500 timesteps (not 50)
- Anti-Hebbian on I neurons (competitive inhibition)

**Risk:** May need more neurons per attractor than we have (20 traces in 2000
neurons = 100 neurons per attractor). Might work at N=5000+.

**Verdict:** Most aligned with Remanentia's SNN identity. TEST.

### Route C: Reservoir + Trained Readout

**Mechanism:** Fixed random recurrent network (reservoir) generates
high-dimensional nonlinear features. Trained linear readout extracts
discriminative signal.

**Key insight:** Our spike features ARE reservoir features — they're just
not discriminative under cosine. A TRAINED readout learns which dimensions
matter. cosine() treats all dimensions equally; a trained readout weights them.

**For Remanentia:** Keep SNN as reservoir. Add ridge regression readout
trained on (query_spike_features -> trace_id).

**Training data problem:** 14 queries isn't enough. But we can augment:
- Paraphrase each query 10 ways (using the embedding model)
- Use trace paragraphs as additional queries (each paragraph "belongs to"
  its parent trace)
- Cross-validation: leave-one-out over 14 queries

**Verdict:** Practical, testable, theoretically grounded. TEST.

### Route D: Predictive Coding

**Mechanism:** Network stores predictions, not patterns. Each memory =
a generative model that predicts what follows. Retrieval = find the model
that best explains the query.

**Key insight:** Memory is reconstructive (Bartlett 1932). The brain doesn't
store exact copies — it stores compressed models that reconstruct on demand.

**For Remanentia:** Each trace trains a local predictive model. At retrieval,
the query is compared against what each model would predict.

**Problem:** Requires hierarchical architecture (predictive coding has
multiple levels). Complex to implement and train.

**Verdict:** Elegant but too complex for immediate testing. DEFER.

### Route E: Paragraph-Level SNN with Attention

**Mechanism:** Split traces into paragraphs. Each paragraph encoded
separately. W learns which paragraph-pairs are related across traces.
Query activates relevant paragraphs.

**Key insight:** Best-paragraph matching (85.7%) works because it avoids
dilution. The SNN could learn WHICH paragraphs matter, making the
paragraph selection itself learned rather than brute-force.

**For Remanentia:** Encode each paragraph as a group of neurons. STDP
between paragraph-groups creates cross-trace paragraph associations.
Query activates best-matching paragraph groups.

**Testable:** Split 20 traces into ~100 paragraphs. N=2000 = 20 neurons
per paragraph. W learns which paragraphs co-occur.

**Verdict:** Combines our best finding (paragraph matching) with SNN learning. TEST.

### Route F: Kanerva Sparse Distributed Memory

**Mechanism:** Binary addresses, hard locations, radius-based read/write.
Content-addressable via address similarity.

**Problem:** With embedding addresses, this reduces to embedding similarity.
Same as Route A.

**Verdict:** Equivalent to what we have. REJECT.

### Route G: Graph Neural Network on Trace Relations

**Mechanism:** Build a graph: nodes = traces, edges = temporal proximity,
same project, same type, lexical overlap. Use GNN to propagate context.

**Key insight:** The missing dimension (context) IS a graph. Temporal
sequence, project membership, causal chains — all graph structure.

**Problem:** Not an SNN. Would require PyTorch Geometric or similar.
Departure from Remanentia's identity.

**Verdict:** Interesting but not SNN-native. DEFER unless SNN routes fail.

### Route H: Contrastive Hebbian Learning

**Mechanism:** Two-phase learning: free phase (query alone) and clamped
phase (query + correct trace). W learns the difference.

**Key insight:** This is biologically plausible supervised learning.
STDP is unsupervised — it can't learn discrimination. CHL adds the
supervision signal needed for retrieval.

**Problem:** Same training data limitation as Route C. But augmentation
helps.

**Verdict:** Biologically plausible alternative to STDP. TEST alongside C.

## Routes to Test (ranked)

| Route | What | Why Test | Risk |
|-------|------|----------|------|
| B | E/I attractor | Most SNN-native, addresses root cause | May need N>5000 |
| C | Reservoir + readout | Reuses existing infrastructure | Needs training data |
| E | Paragraph-level SNN | Combines best finding with SNN | Encoding complexity |
| H | Contrastive Hebbian | Plausible supervised learning | Same data limitation as C |

## Routes Rejected

| Route | What | Why Reject |
|-------|------|-----------|
| A | Modern Hopfield | Equivalent to embedding similarity |
| D | Predictive coding | Too complex for immediate testing |
| F | Kanerva SDM | Equivalent to embedding similarity |
| G | Graph NN | Not SNN-native |

## What Each Route Tells Us If It Works

- **B works:** Inhibition was the missing ingredient all along. The SNN
  can be an associative memory with architectural fix.
- **C works:** The SNN produces useful features, they just need trained
  extraction. The "memory" is in the reservoir, not in cosine similarity.
- **E works:** Granularity was the issue. Document-level is too coarse.
  Paragraph-level SNN encoding is the right scale.
- **H works:** Unsupervised STDP is fundamentally insufficient. Memory
  requires supervised or semi-supervised learning signals.

## What Each Route Tells Us If It Fails

- **B fails:** Even with inhibition, LIF networks can't create content-
  addressable memory at this scale. Move to different computational model.
- **C fails:** The reservoir features are truly uninformative. The spike
  dynamics add nothing to the input encoding.
- **E fails:** Paragraph segmentation doesn't help the SNN either. The
  dilution isn't the problem — the dynamics are.
- **H fails:** Not enough training signal in 14 queries even with
  augmentation. Need larger eval set.
