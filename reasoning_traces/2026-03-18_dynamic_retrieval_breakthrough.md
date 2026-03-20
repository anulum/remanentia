# Reasoning Trace — Dynamic Retrieval Breakthrough (2026-03-18)

The static projection approach (cos(Wx, Wy)) was fundamentally broken. All four normalization attempts (row-norm, Pearson, delta-W, original) produced 0% SNN-only precision and 0.95+ negative scores.

Dynamic spike pattern matching — inject stimulus, run 50ms, compare which neurons fire — achieved:
- SNN-only: 79% P@1 (from 0%), MRR 0.834 (from 0.1)
- Negative rejection: max 0.175 (from 0.984), mean 0.133 (from 0.967)

The key insight: the SNN must be used as a dynamical system, not as a matrix multiply. The nonlinear threshold + recurrent dynamics produce discriminating spike patterns that linear projection cannot. This is how biological associative memory works — pattern completion through dynamics, not static inner products.

With embeddings + dynamic retrieval + TF-IDF hybrid:
- Precision: 93% (13/14)
- Negative max: 0.170
- The SNN now genuinely contributes to discrimination

Remaining issue: interference test still fails (confounders outscore real trace) — this is a keyword-level problem, not SNN. The confounders share identical keywords. Content fingerprinting or trace-ID disambiguation needed.
