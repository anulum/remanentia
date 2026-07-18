<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
-->

# STDP-reach (LTP-dominant) development result — the reach axis is fixed, the residual is a marginal sparsity gap (2026-07-18)

The ADR-0007 amended-config dev probe (`amended_config_g1_dev_probe_result.md`) isolated the blocker: at
the supra-threshold-ceiling operating point, online STDP potentiated E→E only to ~1/3 of the `weight_max`
saturation, so the trained network stayed sub-threshold (P@1 0). Root cause: ADR-0006 froze
`a_plus = 0.005 < a_minus = 0.006` — an **LTD-dominant** rule that caps potentiation. ADR-0008 amends to
**LTP-dominant** plasticity to test whether online STDP can then reach the self-sustain threshold.

## Configuration (development)

ADR-0008 point: 2048 neurons, 80/20 E/I, connectivity 0.3, `weight_max` 2.0 (ADR-0007 gain ceiling) **plus
`a_plus = 0.008 > a_minus = 0.004` (LTP-dominant), 6 epochs**. Development corpus (8 documents, 700–2450
steps), encoder `active_fraction` 0.05, `input_current` 18.0, train seed 11. Trained + probed on the
bit-identical Rust streamed backend: 68,250 steps in 455 s (6.7 ms/step). Disjoint held-out probe, seeds
[11, 29, 47], 25 % corruption and no-input false-recall audits.

## The reach axis IS fixed — saturation reached

| | LTD-dominant (ADR-0007) | LTP-dominant (ADR-0008) |
|---|---|---|
| `max\|ΔW_EE\|` | 0.5339 | **1.9464** |
| `max W_EE` (vs `weight_max` 2.0) | 0.6513 (~1/3) | **2.0000 (saturation)** |

Flipping to LTP-dominant did exactly what the LTD-vs-LTP argument predicted: online STDP now drives the
recurrent excitation all the way to the `weight_max` ceiling. The reach axis is no longer the blocker.

## But the trained network is still (marginally) sub-threshold — P@1 = 0

Reachable-depol go/no-go (`gain_regime.reachable_gain_report`, landed 3937f4a): trained `W_EE` depol@5 % =
**8.25 mV vs the 10 mV gap → NO-GO** (82 % of the gap). Held-out result:

| condition | trained | shuffled | random | zero | encoder-only |
|-----------|---------|----------|--------|------|--------------|
| P@1       | 0.000   | 0.0417   | 0.0417 | 0.000| 0.000        |

Paired `E[P@1_trained − P@1_shuffled]` = **−0.0417** (95 % CI [−0.125, 0.000], 24 pairs); 25 % corruption
margin −0.0417; false-recall 0.000. Every ADR-0006 gate is False except `g2_false_recall_ok`. H0 **not
rejected** — the trained network shows no completion, and the shuffled/random controls even pick up an
occasional spurious hit (1/24) the silent trained network does not.

## Why — the residual is SPARSITY × ceiling, and it is MARGINAL

The gain GO (39.9 mV) was the `weight_max`-saturated ceiling. LTP-dominant STDP now reaches that ceiling on
the strongest synapses (`max W_EE` = 2.0), yet the network-average reachable depolarisation is only 8.25 mV,
because the potentiated E→E is **sparse and trace-specific** (only the coincident synapses saturate, not the
whole block) AND those peak synapses are themselves `weight_max`-capped at 2.0. The residual is a **marginal**
miss: 8.25/10 = 82 %; the self-sustain fixed point needs 6.06 % autonomous E-active (the encoder drives 5.0 %),
so the miss is +1.75 mV / +21 % active — a far smaller move than G1's deep sub-threshold.

## Converging trajectory

| stage | state | P@1 |
|---|---|---|
| G1 (512, LTD-dominant) | sub-threshold even at saturation | 0 |
| ADR-0007 (2048/conn0.3/wmax2.0, LTD-dominant) | ceiling crosses (39.9 mV) but STDP capped at 0.65 | 0 |
| **ADR-0008 (LTP-dominant, 6 epochs)** | **STDP saturates (2.0); reachable 8.25 mV = 82 % of gap** | **0 (marginal)** |

Each amendment removed one binding constraint (scale → gain-ceiling → STDP-reach). The residual is now a
**sparsity/ceiling** gap of 1.75 mV, not a mechanism failure.

## Next (owner-gated — a small, targeted amendment)

The remaining levers are small and quantified: (a) encoder `active_fraction` 0.05 → ~0.065 so the trace seeds
≥ 6.1 % co-active (above the self-sustain fixed point); and/or (b) `weight_max` headroom (the peak synapses
are ceiling-capped at 2.0 → raising it lets the sparse potentiated E→E climb higher). A next ADR-0006
amendment should preregister one of these (recommend (a): it is the physiological cue-density lever, and the
`reachable_gain_report` go/no-go already predicts the crossing fraction). Only if a dev probe then shows a
supra-threshold *trained* network AND non-zero held-out completion is the locked G2 worth proposing.
The locked 10-seed G2 stays owner-gated; ADR-0007/0008 are not accepted.

## Provenance
Runner `scratchpad/amended_dev_probe.py` (Rust training via `train_memories(backend=…)`; Rust probe via a
bit-identical `run_episode` monkeypatch). Grounded in ADR-0006/0007/0008, `gain_regime.py`
(`reachable_gain_report`), `held_out_g1_result.md`, `amended_config_g1_dev_probe_result.md`. Commits this
line: 13a93f0 (Rust trainer), b77fa20 (validate fix), 3937f4a (reachable-depol go/no-go). Development probe
(dev seeds) — NOT the locked G2. REMANENTIA HEAD at write = 916ecf1.
