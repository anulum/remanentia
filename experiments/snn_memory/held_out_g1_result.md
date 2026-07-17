<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
-->

# Held-out G1 development result — honest negative (2026-07-17)

The temporal SNN memory hypothesis (recurrent E/I LIF + online pair-based STDP forms a temporal
attractor that completes a text-derived memory after the cue is removed) was tested at development
scale with the **disjoint held-out probe** (`snn_memory/held_out_probe.py`), which calibrates each
label's target signature from the calibration **suffix** disjoint from the scored cue prefix — so
no target is derived from bytes the cue exposes. This replaces the earlier circular self-matching
(`experiment.evaluate_condition`), which the code itself held fail-closed.

## Configuration (development)

`experiments/snn_memory/development_config.json`: 512 neurons, 80/20 E/I, connectivity 0.1, 3
epochs, `input_current` 18.0, encoder `active_fraction` 0.05. Corpus `development_corpus.json`
(8 documents, sequence lengths 700–2450 steps). Per-seed independent training on seeds
[11, 29, 47]; disjoint held-out probe + 25% cue-corruption and no-input false-recall audits.

## Result — H0 not rejected

| seed | trained P@1 | shuffled P@1 | completion signal (Σ) |
|------|-------------|--------------|-----------------------|
| 11   | 0.000       | 0.000        | 0.0                   |
| 29   | 0.000       | 0.000        | 0.0                   |
| 47   | 0.000       | 0.000        | 0.0                   |

P@1 is 0.000 in **every** condition (trained / shuffled / random / zero / encoder-only); the
paired trained−shuffled effect is 0.0 (95% CI [0.0, 0.0], 24 pairs); false-recall rate 0.0. Every
ADR gate returns False (G1 and G2). The null hypothesis E[P@1_trained − P@1_shuffled] ≤ 0 is **not
rejected**. This is a valid, recorded scientific result, not a failure to hide.

## Why — sub-threshold recurrent-gain regime (not a bug)

The completion signal is 0 because the network is **silent once the external cue drive is
removed**: there is no autonomous activity to score. `snn_memory/gain_regime.py` quantifies why. A
resting neuron must climb the `v_threshold − v_rest` = **10 mV** gap to fire; with the cue gone the
only drive is the recurrent input `I = spikes @ W`.

| N    | init recurrent depol. @5% active | E→E-saturated @5% active | E active fraction needed (saturated) |
|------|----------------------------------|--------------------------|--------------------------------------|
| 512  | 0.25 mV                          | 1.66 mV                  | ~30 %                                |
| 2048 | 1.02 mV                          | 6.76 mV                  | ~7.4 %                               |

With **initial** weights the recurrent depolarisation is 10–40× below the 10 mV gap at both scales —
no self-sustain is possible from initialisation. Only STDP potentiation of the E→E block could
bridge it, but even the `weight_max`-**saturated** upper bound needs ~30% of the excitatory
population co-active at 512 (~7.4% at 2048), whereas the encoder drives only `active_fraction` = 5%
and the trained network's completion signal is 0 — i.e. **online STDP under this configuration does
not potentiate the E→E block anywhere near saturation.**

(The raw spectral radius of W exceeds 1 — 3.7 @512, 15.4 @2048 — but that is the Dale 80/20
net-excitatory mean outlier, not the threshold-gated dynamical gain; the depolarisation-vs-gap
comparison above is the physical criterion.)

## G2-spend decision

The recurrent input scales with `n_excitatory · connectivity` (∝ N), so 2048 neurons multiply it
~4× (required co-active fraction 30% → 7.4%). That is materially closer to criticality but still
sub-threshold at the operative 5% active fraction given the under-potentiation the 512 run
demonstrates. **A brute-force 2048 one-shot at this configuration is therefore expected to reproduce
the negative**, so it was not run (keeper autonomy; declining a spend that buys a known-negative).

Crossing the threshold requires a **regime change** — stronger potentiation, higher
`active_fraction`, higher `weight_max`, or more recurrent excitation — which tests a materially
different hypothesis and is a separate, owner-gated decision, not this run.

## Reproduce

- Held-out probe + gates: `snn_memory/held_out_probe.py` (100% covered).
- Gain-regime diagnostic: `snn_memory/gain_regime.py` → `gain_regime_report(config, seed,
  active_fraction)` (100% covered).
- End-to-end development run: `experiments/snn_memory/held_out_g1.py` (needs the pinned encoder;
  per-seed train + held-out probe + decision).
