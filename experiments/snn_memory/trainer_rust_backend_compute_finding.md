# SNN trainer throughput — the numpy reference is the bottleneck, the Rust backend is the lever (2026-07-18)

Grounded compute assessment for the amended-config G2 development probe (ADR-0007). Determines what
hardware/path makes the run feasible. All numbers measured on the project-workspace workstation (12 cores, load ~50,
5 GB available) under `.venv`.

## Measured — the numpy reference trainer scales ~N^3 and is punishing at 2048

`trainer.train_memories` calls `snn_memory.reference.run_episode` (a pure-numpy per-step Python loop that
copies six state arrays per timestep). Single-thread, on the loaded box:

| N | ms/step | scaling |
|---|---------|---------|
| 512 | 99 | — |
| 1024 | 768 | 7.7x for 2x N (~N^3) |
| 2048 | ~5,900 (extrapolated) | ~7.7x again |

Full development run = 8 documents x ~1,200 steps x 3 epochs x 3 seeds ≈ **86,400 `run_episode` steps**.
At ~5.9 s/step that is **~140 CPU-hours** — infeasible.

## The bottleneck is the numpy reference, NOT the algorithm — a bit-identical Rust path already exists

`rust_snn_memory.run_episode(state, weights, spike_packets, plasticity_enabled, config)` (lib.rs:102) is a
compiled row-pre/column-post LIF + online STDP episode with `plasticity_enabled=True`, exposed through
`snn_memory.stream_backend.StreamBackend.run(..., plasticity_enabled, ...)`. It is **bit-identical** to the
numpy reference — the G0 gate proves one installed-PyO3 full-episode fixture matches spikes, every voltage
timestep, recurrent current, traces, refractory counters, and final weights (`docs/research/temporal_snn_memory.md`).

Grounded Rust throughput at 2048: the `stream_stage1` verifier's `large_replay` ran a **2,048-neuron episode
twice (2 x 16 steps) in 0.737 s ≈ 23 ms/step** (sparse, plasticity-off). Even allowing for dense connectivity
(0.3) and plasticity, a Rust training step is **tens of ms, ~100-250x the numpy reference**. The full
development run drops from ~140 CPU-hours to **roughly ~1 hour on THIS box** — feasible, with identical results.

## GPU is not applicable here (three independent reasons)

1. **The GPU is torch-incompatible.** The GTX 1060 6 GB has compute capability 6.1; the installed PyTorch is
   built only for CC >= 7.5, so `cuda.is_available()` is True but no kernel runs on it.
2. **The SNN sim has no GPU path.** `reference.py` / `state.py` / `stream_backend.py` / `rust_snn_memory` carry
   no cuda/cupy/torch/numba — it is pure CPU (numpy + a Rust crate).
3. **The encoder is CPU-pinned** (`sentence_encoder.py:54` `device="cpu"`) and is a one-off, not the bottleneck.

## The ML350 (weak CPU) is the wrong direction

This is a CPU-bound, per-core-limited, ~N^3 workload that is effectively single-threaded in the numpy path.
Moving it to a **weaker per-core CPU** makes it slower, not faster; extra server cores do not help while the
trainer is single-thread numpy. Do not start the ML350 for this — wire the trainer to Rust instead.

## Decision + the Rust-wiring spec (next execution step)

Wire `train_memories` to a Rust-backed episode path, keeping the numpy reference as the verification ORACLE
(the multi-language-compute pattern: fastest path in production, Python floor as the bit-exact oracle):

1. Add a Rust episode step to the trainer that constructs the Rust `NetworkState` / `WeightMatrix` / `ModelConfig`
   from the numpy state/weights/config (the exact G0-fixture conversion), calls `run_episode(plasticity_enabled=True)`,
   and reads back `final_weights` to chain the next episode — using the same conversion the G0 parity already proves.
2. **Parity test (G0 discipline):** train a small network (e.g. N=12/20, the reference-test scale) both ways and
   assert bit-identical `final_weights` and `signatures`. If any bit differs, the Rust path is not admissible.
3. Keep the numpy path selectable (oracle / fallback) so parity stays checkable.
4. 100 % coverage on the new path, ruff clean; then run the amended-config (2048 / conn 0.3 / wmax 2.0) dev probe
   on the Rust path (~1 h on this box) for the real signal on whether STDP potentiates E->E to autonomous completion.

Records: measured with `snn_memory.trainer.train_memories` micro-bench (scratchpad); Rust reference from the
`stream_stage1` large_replay. Grounded in `snn_memory/reference.py`, `snn_memory/gain_regime.py`,
`rust_snn_memory/src/lib.rs`, `snn_memory/stream_backend.py`, ADR-0006/0007.
