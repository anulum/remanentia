<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
-->

# Gain-regime feasibility for G2 — the negative is a marginal sub-threshold operating point (2026-07-18)

The held-out G1 result (`held_out_g1_result.md`) is an honest negative: the 512-neuron development
network is silent once the external cue is removed, so recall is 0 in every condition. `gain_regime.py`
localised the cause — the excitatory recurrent depolarisation stays below the `v_threshold − v_rest`
= 10 mV firing gap. This analysis answers the decision that determines whether the owner-gated locked
G2 (2,048-neuron, 10-seed) is worth funding, and at what operating point, **without paying for a
training run**.

## Method

Analytic sweep using `snn_memory.gain_regime` verbatim (no reinvention; `spectral_radius` is skipped —
it is documented as *not* the operative criterion and is an expensive O(N³) eig). The operative test is
`gain_regime_report(...)["crosses_threshold_at_saturation"]`: whether the `weight_max`-**saturated**
E→E ceiling — the most online STDP could ever build — depolarises a resting neuron to the firing gap
at a given autonomous active fraction. Levers swept are the ones ADR-0006 does **not** pin (recurrent
connectivity, `weight_max`, the threshold gap, the encoder active fraction); N = 2,048 and the 80/20
E/I split are held at the ADR-0006-locked values. Cost: one weight initialisation plus a column-sum per
config — it runs on the current workstation despite CPU pressure. Seed 11 (the finding is seed-robust;
the fleet sweep over seeds 11/29/47 gives the same 7.4 % for the locked config).

## Result — the ADR-0006-locked config is ~1.5× short of self-sustain

At the ADR-0006-locked point (N = 2,048, connectivity 0.1, `weight_max` 1.0, 10 mV gap) with the
encoder driving 5 % of the population, the saturated E→E ceiling depolarises a neuron by only **6.76 mV
of the 10 mV gap** — it does **not** cross; self-sustain would need **7.4 %** autonomous excitatory
activity, above the 5 % the cue drives. This reproduces the G1 negative's mechanism exactly: the
network cannot self-sustain, hence it is silent after cue removal.

Each single-lever amendment (everything else held at the locked values) that crosses at the 5 % drive:

| lever | locked | crosses at | saturated depolarisation @5% | autonomous E-active needed |
|-------|--------|-----------|------------------------------|----------------------------|
| connectivity | 0.10 | **≥ 0.15** | 10.13 mV | 4.9 % → 0.7 % at 1.0 (67.5 mV) |
| weight_max | 1.0 | **≥ 1.5** | 10.03 mV | 5.0 % → 1.3 % at 6.0 (39.5 mV) |
| encoder active fraction | 0.05 | **≥ ~0.08** | 10.00 mV @ 0.074 | (drive lever, not a model change) |

Denser bounded combinations cross with **< 1.5 % autonomous E-active needed** (e.g. connectivity 0.5 +
`weight_max` 1.5 → 1.0 %; connectivity 0.3 + `weight_max` 2.0 → 1.3 %) — **below the 5 % the cue already
drives** — so in those regimes even *partial* STDP potentiation, not full saturation, can suffice.

## Interpretation

`crosses_threshold_at_saturation` is **necessary, not sufficient**: it says the ceiling is above
threshold, not that STDP reaches the ceiling. G1 showed STDP did not potentiate E→E near saturation
at the sub-threshold locked config. But in a supra-threshold regime the needed autonomous fraction
(0.7–2.5 %) sits below the 5 % cue drive, so a fair G2 test there does not require full saturation.
The locked config, by contrast, is a **provable foregone negative**: even a fully-saturated ceiling is
below threshold at the cue drive.

This also resolves the D3 dynamical-repertoire open question (the ADR/CEO caveat on `f2f439a`): the D3
preregistered 72-episode witness search found no `settled_periodic` / residual-tail episode because
every tested config is sub-threshold — the network dies (`silent_decay`) or wanders and never
self-sustains an attractor. A supra-threshold amended regime makes `settled_fixed`/`settled_periodic`
dynamically accessible, so the four D3 pragma-excluded branches (`experiment_lock` 555/598 +
`gb_preflight` 223/274) are reachable-in-principle, **not proven-dead**.

## Decision + reproduction

Do not fund the locked G2 as written (a foregone negative). Preregister a supra-threshold operating
point via **[ADR-0007](../../docs/adr/0007-amend-g2-supra-threshold-operating-point.md)** (recommended:
connectivity 0.1 → 0.3, `weight_max` 1.0 → 2.0, 10 mV gap untouched), owner-accepted before any run,
with a `gain_regime` go/no-go pre-check. Publishable either way.

Reproduce: `gain_regime_report(ModelConfig(n_neurons=2048, excitatory_fraction=0.8, connectivity=C,
weight_max=W), 11, 0.05)` over the grid above. The locked-vs-amended crossing numbers are
regression-locked in `tests/test_snn_memory_gain_regime.py`
(`test_g2_locked_config_is_marginally_subthreshold_at_the_encoder_drive`,
`test_g2_single_lever_amendment_crosses_the_ceiling_at_the_encoder_drive`).
