# ADR-0009: Amend ADR-0006 — `weight_max` headroom so the reachable depolarisation crosses the gap

- **Status:** Proposed
- **Date:** 2026-07-18
- **Decider:** Miroslav Šotek (pending acceptance — no locked run until Accepted)
- **Supersedes:** —
- **Superseded by:** —
- **Amends:** [ADR-0006](0006-preregister-temporal-snn-memory-experiment.md); complements
  [ADR-0007](0007-amend-g2-supra-threshold-operating-point.md) +
  [ADR-0008](0008-amend-stdp-reach-ltp-dominant-plasticity.md)

## Context

The amendment chain has converged the failure to a single, marginal, quantified gap:

- **G1** (`held_out_g1_result.md`): sub-threshold even at the saturated ceiling → P@1 0.
- **ADR-0007** (`amended_config_g1_dev_probe_result.md`): the `weight_max`-saturated ceiling crosses
  (39.9 mV) but LTD-dominant STDP capped E→E at ~1/3 of it → P@1 0.
- **ADR-0008** (`stdp_reach_dev_probe_result.md`, LTP-dominant): STDP now saturates the strongest E→E
  synapses (`max W_EE` = 2.0 = `weight_max`), yet the trained network's **reachable** depolarisation is
  8.25 mV vs the 10 mV gap — **NO-GO at 82 %** (`gain_regime.reachable_gain_report`) → P@1 still 0,
  trained − shuffled −0.0417.

The residual is a **marginal sparsity × ceiling** gap of 1.75 mV. The potentiated E→E is sparse and
trace-specific; the peak synapses that carry it are `weight_max`-capped at 2.0. So the reachable
depolarisation is now ceiling-limited on the very synapses STDP already saturated — raising `weight_max`
lets those sparse, saturated synapses climb higher and lifts the reachable depolarisation proportionally.

## Decision

Amend `weight_max` upward so the sparse LTP-dominant-potentiated E→E can clear the firing gap, preregistered
and fixed **before** the run:

- **`weight_max` 2.0 → 3.0.** Analytic basis: the reachable depol scales with the potentiated-synapse
  magnitude (the trained peak tracks the ceiling), so 8.25 mV × (3.0 / 2.0) ≈ 12.4 mV — a comfortable ~1.24×
  over the 10 mV gap, so even partial re-potentiation to the new ceiling crosses. (`weight_max` 2.5 is the
  marginal crossing at ≈ 10.3 mV; 3.0 gives margin.)
- Hold ADR-0007 (2048 / conn 0.3 / 10 mV gap) and ADR-0008 (LTP-dominant `a_plus 0.008 > a_minus 0.004`,
  6 epochs). This amendment changes only the E→E weight ceiling.

**Go/no-go instrument:** `gain_regime.reachable_gain_report` on the trained weights (landed 3937f4a) — the
sufficient-side predictor — MUST report `crosses_threshold_reachable = True` at the encoder active fraction
before the locked G2 is proposed. Do not fund G2 unless a development probe shows a supra-threshold **trained**
network AND non-zero held-out completion.

## Held fixed from ADR-0006 (preregistration integrity)

The primary null `E[P@1_trained − P@1_shuffled] ≤ 0`; the G2 decision thresholds (mean ≥ 0.15, paired 95 %
lower bound > 0.05, beats random/zero, 25 % corruption margin, no-input false recall < 0.05, all ten seeds
`11,29,47,71,101,131,167,211,257,307`); the trained/shuffled/random/zero/encoder-only controls; the pinned
`.snn_models` encoder + its `active_fraction`; the cue-only fresh-process boundary; 2048 neurons; 80/20 E/I;
E→E-only plasticity with Dale signs and post-update bounds; the 10 mV biophysical gap.

**Anti-p-hacking guard:** exactly ONE amended `weight_max`, chosen from the analytic reachable-depol scaling
of the ADR-0008 measurement **before** observing any locked recall. Not tuned on the locked set; a G2 failure
rejects the tested configuration and returns no parameters.

## Consequences

- Positive: it targets the exact 1.75 mV residual the ADR-0008 probe isolated — the smallest, most-precise
  move in the chain (a single ceiling parameter, analytically predicted to cross).
- Negative: a higher `weight_max` widens the runaway/instability envelope — the network could tip from a
  memory attractor into `wandering_active` saturation-noise. GUARDS: the post-update `weight_max` bound holds
  the ceiling; the `settled_*` vs `wandering_active` trajectory classification + the no-input false-recall
  audit must confirm the trained network settles into a stable attractor, not runaway activity; and
  `reachable_gain_report` must show the reachable (not just ceiling) depolarisation crossing, not overshooting
  into instability. A dev probe that produces `wandering_active` or a high false-recall rate is itself a
  negative (the ceiling is too high), not a pass.
- Negative: it does not follow that any `weight_max` succeeds; the amended run binds only its configuration.
- Follow-up: on acceptance, run the LTP-dominant `weight_max` 3.0 development probe (memory-scheduled per the
  CEO deep-swap policy — REMANENTIA heavy runs serialise against the fleet suites); check `reachable_gain_report`
  GO + `settled_*` trajectory + non-zero held-out completion; only then is the locked G2 worth proposing. G1
  and the ADR-0007/0008 negatives are retained as recorded results.
