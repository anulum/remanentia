# ADR-0007: Amend ADR-0006 — a supra-threshold operating point for the G2 fair test

- **Status:** Proposed
- **Date:** 2026-07-18
- **Decider:** Miroslav Šotek (pending acceptance — no locked run until Accepted)
- **Supersedes:** —
- **Superseded by:** —
- **Amends:** [ADR-0006](0006-preregister-temporal-snn-memory-experiment.md)

## Context

G1 returned an honest negative at the 512-neuron development scale: trained/shuffled P@1 = 0.000
in every condition, H0 not rejected (`experiments/snn_memory/held_out_g1_result.md`). The
mechanistic diagnostic `snn_memory/gain_regime.py` localised it to a **sub-threshold recurrent-gain
regime** — with the cue removed the only drive is `I = spikes @ W`, and the excitatory recurrent
depolarisation stays far below the `v_threshold − v_rest` = 10 mV firing gap, so the network is
silent and there is nothing to score.

ADR-0006 fixes the first supported scale at **2,048 neurons**. An analytic gain-regime feasibility
sweep over the not-ADR-pinned levers (using `gain_regime.py` verbatim; the operative criterion is
`crosses_threshold_at_saturation` — does the `weight_max`-saturated E→E ceiling reach the gap at a
given autonomous active fraction) shows that the **ADR-0006-locked 2,048 config is also sub-threshold**:
at the encoder's 5 % cue drive the saturated ceiling reaches only 6.76 mV of the 10 mV gap (needs
7.4 % active). Running the locked G2 as written is therefore a **foregone negative** that would burn
the locked 10-seed evaluation without giving the STDP-attractor hypothesis a fair test.

The sweep also shows the shortfall is marginal (~1.5×) and that modest, biophysically-plausible
single-lever amendments cross the self-sustain ceiling at the 5 % drive, keeping the 10 mV gap
untouched: **connectivity 0.1 → ≥ 0.15** (10.1 mV) or **weight_max 1.0 → ≥ 1.5** (10.0 mV). Denser
combinations (e.g. connectivity 0.3 + weight_max 2.0) need only ~1.3 % autonomous E-active at the
ceiling — well below the 5 % the cue already drives — so even *partial* STDP potentiation (not full
saturation) can suffice, which the sub-threshold locked config could never allow. Full analysis:
`experiments/snn_memory/gain_regime_feasibility_result.md`; regression-locked in
`tests/test_snn_memory_gain_regime.py`.

`crosses_threshold_at_saturation` is a **necessary, not sufficient** condition — it says the ceiling
is above threshold, not that online STDP reaches it. Only the run decides whether STDP potentiates
E→E enough. The amendment's purpose is to move the fair test from a provable foregone negative to a
regime where a positive is dynamically possible.

## Decision

Amend the ADR-0006 G2 operating point to a **supra-threshold recurrent regime**, preregistered and
fixed **before** the locked run. **Recommended point** (owner fixes the final values on acceptance):

- **connectivity: 0.1 → 0.3**, **weight_max: 1.0 → 2.0** (the two recurrent-gain levers), giving a
  comfortable margin (~1.3 % autonomous E-active needed at the ceiling vs the 5 % cue drive).

**Held fixed from ADR-0006 (the preregistration integrity — unchanged):** 2,048 neurons; 80/20 E/I
Dale split; STDP `A_plus=0.005, A_minus=0.006, tau_plus=tau_minus=20 ms`, E→E-only plasticity; the
`v_threshold − v_rest = 10 mV` biophysical gap; the primary null `E[P@1_trained − P@1_shuffled] ≤ 0`;
the **G2 decision thresholds** (mean ≥ 0.15, paired 95 % lower bound > 0.05, beats random/zero,
25 % corruption margin, no-input false recall < 0.05, all ten seeds `11,29,47,71,101,131,167,211,257,307`);
the trained/shuffled/random/zero/encoder-only controls; the pinned `.snn_models` encoder; the
cue-only fresh-process boundary.

**Go/no-go pre-check:** before the run, `gain_regime.gain_regime_report` at the amended config MUST
report `crosses_threshold_at_saturation = True` at the encoder active fraction. If it does not, do
not run.

**Anti-p-hacking guard:** exactly ONE amended operating point, chosen analytically from the
gain-regime criterion **before** observing any recall, is preregistered here. Parameters are NOT
tuned on the locked evaluation set; ADR-0006's rule stands — a G2 failure at the amended config
rejects support for *that* configuration and no parameters are returned on the locked set.

## Consequences

- Positive: the STDP-attractor claim gets its first *fair* test (a regime where autonomous
  completion is dynamically possible), instead of a foregone sub-threshold negative.
- Positive: publishable either way — a fair-test positive is first support for the mechanism; a
  fair-test negative (ceiling crossed, STDP still under-potentiates) is a sharper mechanism-level
  negative than the sub-threshold one.
- Positive: it resolves the D3 dynamical-repertoire open question (ADR/CEO caveat on `f2f439a`) — the
  `settled_periodic`/residual-tail regimes the preregistered 72-episode witness search never produced
  are absent because every tested config is sub-threshold; a supra-threshold regime makes them
  dynamically accessible, so those four D3 pragma-excluded branches are reachable-in-principle, not
  proven-dead.
- Negative: it does not follow from a 512-neuron negative that 2,048 self-sustains; the amended run is
  a new experiment and its result binds only its configuration.
- Negative: compute cost — the locked 10-seed 2,048 run needs the workstation to free (CPU-saturated
  per `plan_2026-07-07`) or the GTX 1060 bench; the analytic feasibility is cheap but the run is not.
- Follow-up: on acceptance, fix the final operating point, re-run the `gain_regime` go/no-go, then run
  the locked G2. G1's sub-threshold negative is retained as a recorded result, not overwritten.
