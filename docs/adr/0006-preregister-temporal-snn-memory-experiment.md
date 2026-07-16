# ADR-0006: Preregister the temporal SNN memory experiment

- **Status:** Accepted
- **Date:** 2026-07-14
- **Decider:** Miroslav Šotek
- **Supersedes:** —
- **Superseded by:** —

## Context

The historical SNN path did not isolate recurrent temporal memory from encoder
similarity, did not apply an agreed online STDP convention at every timestep,
and did not compare frozen recall against matched shuffled, random, zero-weight,
and encoder-only controls. A negative result from that path therefore cannot
decide whether a correctly implemented temporal attractor can store and recover
text-derived memories.

The replacement is a research-only experiment. It must remain separate from
``MemoryIndex``, ``ArcaneRetriever``, the legacy daemon, and product retrieval
weights until the preregistered mechanism and product gates pass. Recall must
receive a frozen checkpoint and a cue only; it must not receive the corpus or a
candidate-document collection.

## Decision

Implement a separately packaged recurrent E/I LIF memory experiment whose
equations, controls, seeds, persisted contracts, and decision gates are frozen
before the locked evaluation.

- Store recurrent weights as ``W[pre, post]``. Synaptic current is
  ``I_syn[post] = sum_pre spikes[pre] * W[pre, post]``; implementations therefore
  use ``spikes @ W`` or the equivalent ``W.T @ spikes``.
- Apply pair-based STDP online at every 1 ms timestep. A postsynaptic spike
  applies ``W[i,j] += A_plus * pre_trace[i]`` and a presynaptic spike applies
  ``W[i,j] -= A_minus * post_trace[j]``. At each step, existing traces decay,
  the weight deltas use those decayed traces, and only then are current spikes
  added to the traces. Initial constants are ``A_plus=0.005``,
  ``A_minus=0.006``, and ``tau_plus=tau_minus=20 ms``.
- Use 2,048 LIF neurons with an 80/20 E/I split. Excitatory presynaptic rows are
  non-negative, inhibitory presynaptic rows are non-positive, the diagonal is
  zero, and bounds are enforced after every plastic update. Only recurrent
  E→E weights are plastic in the primary experiment; inhibitory weights are
  fixed.
- Train on complete ordered temporal sequences, freeze plasticity, and probe in
  a fresh process with partial or corrupted cues. Autonomous completion is
  measured after removal of external input.
- Compare every cue and seed with trained, topology/value-matched shuffled,
  matched-random, zero-recurrent, and encoder-only conditions. Initial state,
  cue packets, encoder output, and RNG seed are held constant across conditions.
- Persist experiment, checkpoint, and result records against the versioned JSON
  Schemas in ``docs/schema``. Checkpoints are whole-directory staged bundles with
  authenticated NPZ arrays and complete JSONL training history; unsafe pickle is
  not permitted. The metadata binds the exact corpus manifest, encoder tree,
  encoder configuration, training seed and input current.

The primary null hypothesis is
``E[P@1_trained - P@1_shuffled] <= 0``. The alternative is that correct online
STDP creates recurrent temporal completion that persists after cue removal and
produces a reproducible trained-over-shuffled gain. Product value is a separate
hypothesis and cannot rescue a failed mechanism test.

## Decision gates

- **G0 — equation correctness:** hand-computed causal LTP and reverse-order LTD,
  declared simultaneous-spike behaviour, timestep Python/Rust parity, Dale
  signs, zero diagonal, refractory state, and weight bounds must all pass.
- **G1 — controlled completion:** trained exceeds shuffled by at least 0.25
  P@1, the paired 95% interval lower bound is above zero, completion occurs
  after cue removal, and zero recurrent weights do not show the same effect.
- **G2 — locked text-derived recall:** use seeds ``11, 29, 47, 71, 101, 131,
  167, 211, 257, 307``. Mean ``P@1_trained - P@1_shuffled`` must be at least
  0.15 with paired 95% lower bound above 0.05; trained must beat random and
  zero, retain a positive correct-attractor margin at 25% corruption, keep
  no-input false recall below 0.05, and include every seed.
- **G3 — mechanism audit:** record recurrence/input current over time, observe
  the correct memory in the temporal state before rate reduction, show that
  shuffling materially reduces the attractor, prove fresh-process checkpoint
  persistence, and stay within the declared forgetting bound.
- **G4 — product integration:** may be proposed only after G2 and G3 pass. It
  requires at least 0.05 absolute P@1 gain over encoder-only on a separate task,
  an approved latency/memory budget, and no production retrieval regression.

Any G0 failure stops the experiment. A G2 failure rejects support for the tested
configuration; parameters must not be returned on the locked evaluation set.

## Options considered

- **Preregister a separate SNN research package and contracts.** Chosen; it
  isolates the mechanism while making a negative result interpretable.
- **Patch the legacy SNN daemon in place.** Rejected; it would preserve ambiguous
  timing and couple the experiment to deprecated global state.
- **Restore an SNN product weight before mechanism validation.** Rejected; a
  retrieval score could hide encoder or reranking leakage and would not test
  recurrent memory.
- **Abandon the SNN path without retesting.** Rejected; the historical path did
  not exercise the stated online temporal-memory hypothesis.

## Consequences

- Positive: results are paired, multi-seed, control-matched, checkpointed, and
  reproducible through explicit machine-readable contracts.
- Positive: the cue-only fresh-process boundary prevents candidate-document
  re-encoding from masquerading as memory.
- Negative: Python/Rust parity, locked manifests, and matched controls add
  implementation and compute cost before any product experiment.
- Negative: the first supported scale is 2,048 neurons; no claim about a
  20,000-neuron system follows from it.
- Follow-up: implement the accepted contracts and G0–G3 experiment. Draft a new
  ADR for product integration only if G2 and G3 pass.
- Implementation evidence is deliberately narrower than a universal proof: one
  installed PyO3 full-episode fixture exactly matches the Python public oracle at
  every recorded timestep, including simultaneous E/I recurrence and connected
  zero-weight topology. Broader configuration parity remains a future expansion.
