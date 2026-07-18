<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
-->

# `weight_max` headroom (ADR-0009) development result — the gap narrows to 90.5% but the trained network stays sub-threshold, and the linear-scaling prediction is falsified (2026-07-18)

The ADR-0008 LTP-dominant probe (`stdp_reach_dev_probe_result.md`) isolated the residual blocker to a
**marginal sparsity × ceiling** gap: online STDP saturated the strongest E→E synapses at `weight_max` = 2.0,
yet the trained network's **reachable** depolarisation was 8.25 mV vs the 10 mV firing gap (82 %) → P@1 0.
ADR-0009 preregistered a single lever against exactly that residual — raise `weight_max` 2.0 → 3.0 so the
sparse, ceiling-capped synapses can climb higher — with the analytic prediction that the reachable
depolarisation would scale to 8.25 × (3.0 / 2.0) ≈ 12.4 mV and clear the gap.

**Outcome: an honest negative that also falsifies the ADR-0009 analytic model.** Raising the ceiling did lift
the reachable depolarisation, but only to **9.05 mV (90.5 % of the gap) — NOT the predicted 12.4 mV**. The
trained network remains sub-threshold, P@1 is 0, and — importantly — the network stayed **stable** (no
`wandering_active` runaway, false-recall 0.000), so this is a clean sub-threshold miss, not the instability
ADR-0009 flagged as its downside risk.

## Configuration (development)

ADR-0007 gain-ceiling operating point (2048 neurons, 80/20 E/I, connectivity 0.3, 10 mV gap) + ADR-0008
LTP-dominant plasticity (`a_plus` = 0.008 > `a_minus` = 0.004, 6 epochs) + **ADR-0009 `weight_max` 2.0 → 3.0**
— this amendment changes only the E→E weight ceiling. Development corpus (8 documents, 700–2450 steps),
encoder `active_fraction` 0.05, `input_current` 18.0, train seed 11. Trained + probed on the bit-identical
Rust streamed backend: 68,250 steps in 609 s (8.9 ms/step). Disjoint held-out probe, seeds [11, 29, 47],
25 % corruption and no-input false-recall audits.

## The ceiling was reached — saturation, as at `weight_max` = 2.0

| | `weight_max` 2.0 (ADR-0008) | `weight_max` 3.0 (ADR-0009) |
|---|---|---|
| `max\|ΔW_EE\|` | 1.9464 | **2.8258** |
| `max W_EE` (vs `weight_max`) | 2.0000 (saturation) | **3.0000 (saturation)** |

LTP-dominant STDP again drives the strongest recurrent synapses all the way to the new ceiling — the reach
axis is not the blocker, exactly as at 2.0.

## But the reachable depolarisation is SUBLINEAR — the linear-scaling prediction is falsified

Reachable-depol go/no-go (`gain_regime.reachable_gain_report` / `recurrent_depolarisation_mv`, landed 3937f4a),
the sufficient-side predictor on the **trained** weights:

| `weight_max` | reachable depol @ 5 % | % of 10 mV gap | verdict |
|---|---|---|---|
| 2.0 (ADR-0008) | 8.25 mV | 82.5 % | NO-GO |
| **3.0 (ADR-0009)** | **9.05 mV** | **90.5 %** | **NO-GO** |
| — predicted (linear 2.0→3.0) | 12.4 mV | 124 % | (would GO) |

A 50 % ceiling increase lifted the reachable depolarisation by only **+0.80 mV (+9.7 %)**, not the +50 %
(→ 12.4 mV) the ADR-0009 linear model assumed. **The scaling is strongly sublinear**, so the analytic
prediction is falsified at source.

## Held-out result — P@1 0, H0 not rejected

| condition | trained | shuffled | random | zero | encoder-only |
|-----------|---------|----------|--------|------|--------------|
| P@1       | 0.000   | 0.0417   | 0.000  | 0.000| 0.000        |

Paired `E[P@1_trained − P@1_shuffled]` = **−0.0417** (95 % CI [−0.125, 0.000], 24 pairs); 25 % corruption
margin −0.0417; **false-recall 0.000**. Every ADR-0006 gate is False except `g2_false_recall_ok`. H0 **not
rejected** — the trained network shows no completion; the shuffled control even picks up 1/24 spurious hit the
silent trained network does not. Identical in shape to the `weight_max` = 2.0 result: closer to threshold, but
still sub-threshold, so still zero recall.

## Why — sparsity × ceiling, and why the ceiling lever is sublinear

The reachable depolarisation is `active_fraction × mean_excitatory_column_sum` — it tracks the **mean** E→E
column sum, which is dominated by the large **unsaturated bulk** of the recurrent block. LTP-dominant STDP
saturates only the **sparse, trace-specific** synapses at the ceiling; raising `weight_max` lifts just that
sparse saturated fraction, so the mean — and hence the reachable depolarisation — barely moves. That is the
sublinearity: the ceiling governs the peak synapse, but the peak is a small share of the column sum. The
instability guard passed — `false_recall_rate` 0.000 confirms the higher ceiling did **not** tip the network
into `wandering_active` saturation-noise; it is a stable, sub-threshold attractor, a clean negative.

## Converging trajectory — the ceiling lever is nearly exhausted

| stage | binding constraint | reachable depol | P@1 |
|---|---|---|---|
| G1 (512, LTD-dominant) | scale / sub-threshold at saturation | (sat ceiling < gap) | 0 |
| ADR-0007 (2048/conn0.3/wmax2.0, LTD-dominant) | gain-ceiling vs STDP-reach (capped 0.65) | (deep) | 0 |
| ADR-0008 (LTP-dominant, 6 epochs) | STDP-reach fixed; sparsity × ceiling | 8.25 mV (82.5 %) | 0 |
| **ADR-0009 (`weight_max` 3.0)** | **sparsity — ceiling lever sublinear** | **9.05 mV (90.5 %)** | **0** |

Each amendment removed one binding constraint (scale → gain-ceiling → STDP-reach → ceiling headroom). The
residual is now the **sparsity of the potentiated E→E**, and the `weight_max` lever meets diminishing returns:
extrapolating the observed marginal rate (+0.80 mV per +1.0 `weight_max`, with the sublinearity worsening as
the saturated fraction shrinks) puts the crossing near `weight_max` ≈ 4.2, into the runaway-instability
envelope ADR-0009 warned against — an unattractive lever.

## Next (owner-gated) — `active_fraction` is the linear lever

The reachable depolarisation is **linear** in `active_fraction` (more co-active presynaptic E → proportionally
more terms in the effective column sum, and more synapses potentiated). At `weight_max` 3.0 the crossing
active fraction is **5.52 %** (reachable 9.05 mV at the 5.0 % encoder drive → `0.05 × 10 / 9.05`), down from
6.06 % at `weight_max` 2.0 — the smallest residual in the chain (+0.52 pp of co-active E). So the two levers
compose: the ceiling headroom lowered the crossing fraction to 5.52 %, and `active_fraction` (encoder cue
density) is the direct linear lever to close the last +0.52 pp. This is exactly the lever the ADR-0008 result
doc recommended first (option (a), the physiological cue-density lever), which ADR-0009 bypassed for the
ceiling. A next ADR-0006 amendment should preregister `active_fraction` ~0.05 → ~0.056–0.065; only if a
development probe then shows a supra-threshold **trained** network (`reachable_gain_report` GO + `settled_*`
trajectory + false-recall low) AND non-zero held-out completion is the locked G2 worth proposing. The locked
10-seed G2 stays owner-gated; ADR-0007/0008/0009 are recorded negatives, not accepted.

## Provenance

Runner `scratchpad/amended_dev_probe.py` (Rust training via `train_memories(backend=…)`; Rust held-out probe
via a bit-identical `run_episode` monkeypatch). Grounded in ADR-0006/0007/0008/0009, `gain_regime.py`
(`reachable_gain_report`, `recurrent_depolarisation_mv`), `held_out_g1_result.md`,
`amended_config_g1_dev_probe_result.md`, `stdp_reach_dev_probe_result.md`. Commits this line: 13a93f0 (Rust
trainer, bit-identical), b77fa20 (validate silent-first-row fix), 3937f4a (reachable-depol go/no-go),
1731a8e (ADR-0009). Development probe (dev seeds) — NOT the locked G2.
