# ADR-0008: Amend ADR-0006 — target the STDP-potentiation-reach with LTP-dominant plasticity

- **Status:** Proposed
- **Date:** 2026-07-18
- **Decider:** Miroslav Šotek (pending acceptance — no locked run until Accepted)
- **Supersedes:** —
- **Superseded by:** —
- **Amends:** [ADR-0006](0006-preregister-temporal-snn-memory-experiment.md); complements
  [ADR-0007](0007-amend-g2-supra-threshold-operating-point.md)

## Context

Two negatives now localise the failure precisely:

- **G1** (`experiments/snn_memory/held_out_g1_result.md`): 512-neuron, sub-threshold even at the
  `weight_max`-saturated E→E ceiling → silent → P@1 0.000.
- **ADR-0007 amended-config dev probe** (`experiments/snn_memory/amended_config_g1_dev_probe_result.md`,
  commit 656492f): at the supra-threshold-ceiling operating point (2048 / conn 0.3 / `weight_max` 2.0;
  `gain_regime` GO, saturated depol@5% 39.9 mV vs the 10 mV gap), P@1 is **still 0.000 in every
  condition**. The measured reason: online STDP potentiated the E→E block substantially
  (`max|ΔW_EE| = 0.5339`) but only to `max W_EE = 0.6513` against `weight_max = 2.0` — **~1/3 of the
  saturation ceiling**. So the trained network's real recurrent depolarisation is a fraction of 39.9 mV,
  still below the 10 mV gap, and the network is silent after cue removal.

The binding constraint is therefore **not the gain-ceiling** (it crosses) **but the STDP-potentiation-reach**
— how far online plasticity actually drives E→E toward saturation. ADR-0006 froze the STDP constants at
`a_plus = 0.005`, `a_minus = 0.006` — i.e. **`a_minus > a_plus`, an LTD-dominant rule**: for coincident
pre→post pairs the depression amount slightly exceeds the potentiation amount, so on the roughly balanced
completion-phase activity the net drift caps E→E well below saturation. That cap, not the ceiling, is what
holds recall at zero.

## Decision

Amend the STDP-reach parameters, preregistered and fixed **before** the locked run, to let online
plasticity reach the depolarisation self-sustain needs:

- **Flip to LTP-dominant plasticity: `a_plus > a_minus`.** Recommended `a_plus = 0.008`, `a_minus = 0.004`
  (owner fixes the final pair on acceptance), keeping `tau_plus = tau_minus = 20 ms`. This makes coincident
  E→E pairs net-potentiating so STDP can climb toward saturation instead of being pinned by LTD.
- **Epoch headroom:** allow up to **6 epochs** (from 3) so the LTP-dominant rule has room to accumulate
  toward the reachable fixed point (not merely more of an LTD-capped drift).
- Hold the ADR-0007 gain-ceiling amendment (2048 / conn 0.3 / `weight_max` 2.0, 10 mV gap) so the ceiling
  stays above threshold; this amendment addresses only the reach to it.

**New go/no-go instrument (required before the run):** extend `gain_regime` to score the **reachable-weight**
depolarisation — run the LTP-dominant training at development scale, read the trained `W_EE`, and check its
depolarisation@5% against the 10 mV gap — instead of only the `weight_max`-saturated ceiling. The saturated
ceiling was necessary-not-sufficient (ADR-0007); this reachable-depol check is the sufficient-side predictor.
Do not fund the locked G2 unless the reachable depolarisation crosses the gap at the amended operating point.

## Held fixed from ADR-0006 (preregistration integrity)

The primary null `E[P@1_trained − P@1_shuffled] ≤ 0`; the **G2 decision thresholds** (mean ≥ 0.15, paired
95 % lower bound > 0.05, beats random/zero, 25 % corruption margin, no-input false recall < 0.05, all ten
seeds `11,29,47,71,101,131,167,211,257,307`); the trained/shuffled/random/zero/encoder-only controls; the
pinned `.snn_models` encoder; the cue-only fresh-process boundary; 2048 neurons; the 80/20 E/I Dale split;
E→E-only plasticity with Dale signs and post-update bounds; the 10 mV biophysical gap.

**Anti-p-hacking guard:** exactly ONE amended `(a_plus, a_minus, epochs)` point, chosen from the physical
LTD-vs-LTP argument **before** observing any locked recall, is preregistered here. Parameters are NOT tuned
on the locked evaluation set; ADR-0006's rule stands — a G2 failure at the amended config rejects support
for that configuration and no parameters are returned on the locked set.

## Consequences

- Positive: it gives the STDP-attractor claim the first test where online plasticity can actually reach the
  self-sustain threshold — the reach axis the ADR-0007 dev probe isolated as the true blocker.
- Positive: the reachable-depol `gain_regime` extension tightens the go/no-go to the sufficient condition,
  so a future locked G2 is only funded when a development probe already shows a supra-threshold *trained*
  network — not merely a supra-threshold ceiling.
- Negative: LTP-dominant plasticity risks runaway potentiation / instability; the post-update `weight_max`
  bound and the `wandering_active` / `settled_*` trajectory classification are the guards to watch, and a
  development probe must confirm the network settles rather than saturates into noise.
- Negative: it does not follow that any bounded STDP rule succeeds; the amended run binds only its config.
- Follow-up: on acceptance, extend `gain_regime` for the reachable-depol check + a test; run the LTP-dominant
  development probe; only if it produces a supra-threshold trained network AND non-zero held-out completion
  is the locked G2 worth proposing. G1 and the ADR-0007 negative are retained as recorded results.
