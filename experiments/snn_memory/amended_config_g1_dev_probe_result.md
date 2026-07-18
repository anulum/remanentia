<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
-->

# Amended-config G1-style development result — honest negative, relocated to the STDP-reach axis (2026-07-18)

The held-out G1 result (`held_out_g1_result.md`) was an honest negative rooted in a sub-threshold
recurrent-gain regime (`gain_regime.py`). The gain-regime feasibility analysis
(`gain_regime_feasibility_result.md`, ADR-0007) then showed that the G1/G2-locked config is only
*marginally* sub-threshold and that a modest supra-threshold amendment (connectivity 0.1 → 0.3,
`weight_max` 1.0 → 2.0, 10 mV gap untouched) crosses the self-sustain **ceiling**. This development
probe tests that amended operating point — the fair test the sub-threshold config could not give.

## Configuration (development)

Amended operating point: **2048 neurons, 80/20 E/I, connectivity 0.3, `weight_max` 2.0** (the ADR-0007
recommendation; `gain_regime` go/no-go = GO, saturated depol@5% = 39.9 mV vs the 10 mV gap). Development
corpus (8 documents, 700–2450 steps), encoder `active_fraction` 0.05, `input_current` 18.0, train seed
11, 3 epochs. Trained on the **bit-identical Rust streamed backend** (13a93f0): 34,125 steps in 228 s
(**6.7 ms/step**) — numpy would have needed ~140 CPU-hours. The held-out benchmark ran on the same Rust
path (the `validate_stream_result` silent-first-row fix, b77fa20, was required for the completion probe).
Disjoint held-out probe (calibration suffix disjoint from the scored cue prefix), probe seeds [11, 29, 47],
25 % cue-corruption and no-input false-recall audits.

## Result — H0 not rejected (again), P@1 = 0.000 in every condition

| condition | trained | shuffled | random | zero | encoder-only |
|-----------|---------|----------|--------|------|--------------|
| P@1       | 0.000   | 0.000    | 0.000  | 0.000| 0.000        |

Paired `E[P@1_trained − P@1_shuffled]` = 0.000 (95 % CI [0.000, 0.000], 24 pairs); 25 % corruption margin
0.000; no-input false-recall rate 0.000. Every ADR-0006 gate is False (`g1_pass`, `g2_effect_threshold`,
`g2_beats_random`, `g2_beats_zero`, `g2_corruption_margin_positive`) except `g2_false_recall_ok`. The null
`E[P@1_trained − P@1_shuffled] ≤ 0` is **not rejected**. This is a valid, recorded scientific result.

## Why — the ceiling crosses, but online STDP does not reach it (sharper than G1)

The gain-regime GO for this config was computed at the `weight_max`-**saturated** E→E block (the upper
bound STDP could ever build). This run measures what STDP actually built: after training,
`max|W_EE − W_EE_init| = 0.5339` and `max W_EE = 0.6513` against `weight_max = 2.0` — online pair-based
STDP potentiated the recurrent excitation **substantially (not zero) but only to ~1/3 of the saturation
ceiling**. So the trained network's real recurrent depolarisation is a fraction of the 39.9 mV ceiling
and is still below the 10 mV firing gap → the network is silent after cue removal → P@1 = 0.

This **relocates the binding constraint.** G1's negative was "sub-threshold even at saturation"; this
negative is "the saturation *ceiling* crosses (necessary condition met) but STDP-potentiation *reach* does
not get there" — the `crosses_threshold_at_saturation` necessary-not-sufficient gap, with the *sufficient*
side failing on the STDP-reach axis, not the gain-ceiling axis. Raising the neuron count or the
gain-ceiling further will not help; the lever is whatever governs how far STDP drives E→E toward
saturation.

## Next (owner-gated — a materially different hypothesis)

The next preregistered amendment should target the **STDP-potentiation-reach**, not the gain-ceiling:
learning-rate `a_plus`/`a_minus` (ADR-0006 froze 0.005/0.006), epoch count, and `weight_max` headroom vs
the depolarisation the reachable (not saturated) E→E delivers. `gain_regime` should be extended to score
the reachable-weight depolarisation, not only the saturated ceiling, so the go/no-go predicts the fair
test more tightly. This is a materially different hypothesis and needs its own ADR-0006 amendment before a
locked run. The LOCKED 10-seed G2 stays owner-gated (ADR-0007 not accepted; and this dev result argues the
recommended amended point is not yet worth the locked run — fix STDP-reach first).

## Provenance
Runner `scratchpad/amended_dev_probe.py` (Rust training via train_memories(backend=…); Rust probe via a
bit-identical run_episode monkeypatch — the durable equivalent is threading `backend` through
`held_out_benchmark`). Grounded in ADR-0006/0007, `gain_regime.py`, `held_out_g1_result.md`,
`gain_regime_feasibility_result.md`. Commits this line: 13a93f0 (Rust trainer), b77fa20 (validate fix).
Development probe (dev seeds) — NOT the locked G2. REMANENTIA HEAD at write = b77fa20.
