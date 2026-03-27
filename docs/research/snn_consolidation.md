# SNN Consolidation: Architecture and Negative Results

## Summary

Remanentia uses a 1,000-neuron LIF (Leaky Integrate-and-Fire) spiking
network with STDP (Spike-Timing-Dependent Plasticity) for memory
consolidation orchestration. The SNN's role is novelty detection and
consolidation timing — not retrieval.

## The Negative Result

Over 60 experiments tested whether SNN-based features could improve
text retrieval quality. The conclusion: **STDP in LIF networks does
not create discriminative features for text retrieval.**

- Best-paragraph embedding: 85.7% P@1 vs 50.0% document-level
- TF-IDF(0.4) + best-paragraph(0.6) hybrid: 92.9% P@1 **without SNN**
- No SNN configuration improved retrieval over BM25+embedding baseline

This is a validated negative result. The SNN was not abandoned — its
role was redefined from retrieval to consolidation orchestration.

## What the SNN Does

1. **Novelty detection**: measures how much new input diverges from
   stored weight patterns. Novel information triggers consolidation.

2. **Consolidation timing**: the daemon checks for new traces every
   60 seconds. When novelty exceeds threshold, full consolidation runs.

3. **Complementary learning**: fast weights (STDP, current session)
   and slow weights (consolidated, cross-session) provide dual timescale
   memory. McClelland et al. (1995).

4. **Hippocampal replay**: during idle cycles, stored traces are
   re-injected at low amplitude, strengthening existing associations.

## Architecture

- **Network**: 1,000 LIF neurons, Watts-Strogatz small-world topology
- **Synapses**: STDP with A+=A-=0.005, tau=20ms, W_max=2.0
- **Backends**: Rust (arcane_stdp, 2-3x speedup), GPU (PyTorch), CPU (NumPy)
- **Encoding**: hash-based unigram+bigram encoding to spike patterns
- **Persistence**: membrane potentials + weights saved between sessions

## Rust Acceleration

| Neurons | Rust | Python | Speedup |
|---------|------|--------|---------|
| 1,000 | 156ms | 453ms | 2.9x |
| 2,000 | 907ms | 1,422ms | 1.6x |
| 5,000 | 3,531ms | 7,000ms | 2.0x |

Spike counts identical across all sizes. Rust path is bit-exact with Python.

## Neuromorphic Hardware Path

The SNN backend maps naturally to neuromorphic hardware:
- Intel Loihi (Lava framework)
- SpiNNaker
- BrainScaleS

A proof-of-concept on Intel's Lava framework (CPU simulation) would
demonstrate Remanentia's consolidation on a neuromorphic stack.
