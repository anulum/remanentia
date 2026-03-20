# Reasoning Trace — SNN Retrieval Failure Analysis (2026-03-18)

## The experiment
Retrained a fresh 2000-neuron small-world SNN with embedding-encoded traces (sentence-transformers all-MiniLM-L6-v2). 50 STDP cycles, each injecting all 18 traces.

## The result
SNN-only retrieval: 0% Precision@1 (trained) vs 7% (random). Negative queries score 0.95+. Weight mean saturated from 0.35 to 0.65. The SNN retrieval mechanism is fundamentally broken.

## Root cause
`cos(W @ x, W @ y)` approaches 1.0 for ANY x, y when W is large, dense, and positive. STDP pushes weights up uniformly because every trace activates ~15% of neurons via sparsified encoding, and those neurons spike, strengthening ALL their outgoing connections. After 50 cycles the matrix is nearly uniform — all projections land in the same output region.

## What this means
The weight-projection retrieval approach (project stimulus through W, compare cosines) cannot discriminate when W is dense. The infrastructure (daemon, encoding, dashboard, replay, CLS) is sound. The retrieval scoring is the flaw.

## Possible fixes
1. **Sparse W retrieval** — use only the top-k connections per neuron, zero the rest before projection. Forces discrimination.
2. **Dynamic retrieval** — instead of static projection, inject the query as stimulus, run the network, and compare the resulting spike pattern against stored trace spike patterns. Uses network dynamics, not just weights.
3. **Hebbian fingerprinting** — for each trace, record which neuron pairs had the strongest STDP modification. Retrieval checks which trace's fingerprint best matches the query's activation pattern.
4. **Normalize W per-row** — divide each row by its L2 norm before projection. Prevents the uniform-projection collapse.
5. **Competitive STDP** — add lateral inhibition so only the most-activated neurons learn, not all spiking neurons. Produces winner-take-all dynamics and sparser representations.

## What stays valid
- The encoding upgrade (hash → LSH → embeddings) is correct and confirmed
- The daemon infrastructure (replay, CLS, homeostatic, small-world) is sound
- The dashboard, cognitive snapshots, skill extraction, active retrieval all work
- TF-IDF retrieval is solid and carries the system
- The product concept is valid — the retrieval mechanism just needs a different approach
