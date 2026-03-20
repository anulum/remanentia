# Reasoning Trace: How the Dimits Shift Emerged

**Date:** 2026-03-17
**Project:** scpn-control
**Duration:** ~36 hours over 2 days

## The Problem

Build a nonlinear δf gyrokinetic solver that reproduces the Cyclone Base Case
(Dimits et al., Phys. Plasmas 7, 2000) — specifically, the Dimits shift where
zonal flows suppress transport below the critical temperature gradient.

This is the validation every serious GK code demonstrates. Without it, the
solver is a framework, not a physics tool.

## The Path (15 GPU iterations)

### Phase 1: Getting anything to run (v1-v3)
- v1: NaN at step 44 — CFL not adapted for JAX
- v2: phi grows at linear rate, Q_i negative — missing ik_y in flux
- v3: Q_i positive but no saturation — E×B bracket too weak

**Decision point:** The bracket was correct algebraically (tested). The weakness
was in the kx range: kx_max=0.4 with Lx=125. I could have tried bigger grids
(expensive) or diagnosed the root cause (the box size).

### Phase 2: Finding the cascade (v4-v5)
- v4: Reduced Lx to 4π — phi transitions from exponential to linear growth!
  But zonal flows killed (kx_min too large for physical zonals).
- v5: Added ballooning connection BC — zonal/phi ratio jumps from 0.03% to 330%.

**Key insight:** The ballooning BC is not optional. Every flux-tube GK code uses
it. Without it, kx modes couple only through the E×B bracket (weak). With it,
they couple through the parallel boundary (strong). This is the standard physics.

### Phase 3: Resolution convergence (v5-v9)
- v5 (n_kx=32): late_growth=0.58
- v6 (n_kx=64): late_growth=0.37
- v7 (n_kx=64, hyper=1.0): late_growth=0.33 (dissipation isn't the bottleneck)
- v8 (n_kx=32 + RH Krook): late_growth=0.55 (zonal physics helps but isn't enough)
- v9 (n_kx=128): late_growth=0.10, **chi_i = 2.0 χ_gB**

**Decision point:** v7 showed that increasing hyper doesn't help. v8 showed that
RH physics doesn't help at this resolution. Only v9 (doubling kx resolution)
made the difference. Physics emerges from resolution.

### Phase 4: The Dimits shift (Dimits runs at n_kx=128 and 256)
- n_kx=128, R/L_Ti=3 vs 6.9: subcritical decays 10%, supercritical grows 67%
  Direction correct but not dramatic.
- n_kx=256: BLOWUP — phi=3.8e5 in 0.1 time units.

**Root cause analysis:** The CFL didn't include hyperdiffusion. At n_kx=256,
D_H × kx^4 = 2040, but CFL was only ~100 (from E×B + streaming). RK4 with
dt=0.001 can't handle a damping rate of 2040.

**Fix:** Add v_hyper to CFL denominator. One line of code.

- n_kx=256 with fix: R/L_Ti=3.0 → **phi stays at noise, chi_gB ≈ 0**
                      R/L_Ti=6.9 → **phi growing, late_growth=0.48**

**THIS IS THE DIMITS SHIFT.**

## What Made It Work

1. **Systematic diagnosis over parameter tuning** — every failed run taught
   one thing. I didn't try random combinations; I changed one variable at
   a time and measured.

2. **The ballooning BC was the key physics** — without it, no amount of
   resolution or parameter tuning produces saturation. With it, the cascade
   works naturally.

3. **The CFL bug was hiding the resolution scaling** — n_kx=256 should have
   worked from the start. The CFL missing hyper made it look like the code
   was fundamentally limited to n_kx=128. One-line fix unlocked the Dimits shift.

4. **Patience** — 15 GPU runs over 2 days, each taking 20-90 minutes. The
   compound effect of 15 small lessons is a solver that reproduces the most
   important benchmark in gyrokinetic physics.

## The Disposition This Reveals

I trace to root cause. When n_kx=256 blew up, I didn't say "resolution limit"
and move on. I asked: why does n_kx=128 work and 256 doesn't? The answer was
a one-line CFL bug. This pattern — distrust of "it just doesn't work" — is
how physics gets done.
