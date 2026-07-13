# Validation

Authoritative record of how Remanentia is tested and measured. Numbers here are
kept consistent with `README.md`; where a figure is a point-in-time run it is
dated and labelled.

## Test Matrix

| Python | Platform | Status |
|--------|----------|:------:|
| 3.10 | Ubuntu (CI) | ✅ |
| 3.11 | Ubuntu (CI) | ✅ |
| 3.12 | Ubuntu (CI) | ✅ |
| 3.12 | Windows 11 (local) | ✅ |

## Test Suite

- **≈2,905 test functions** across the `tests/` tree (last full local gate,
  2026-06-29: 2,905 passed / 11 skipped).
- **100 % coverage gate** on the product modules — `pyproject.toml`
  `[tool.coverage.report] fail_under = 100` (10,283/10,283 statements, 0 missing at
  the 2026-06-29 gate).
- **Scope caveat (honest):** the coverage gate measures the pure-**Python** product
  path. The 16 Rust acceleration crates are validated separately by the per-crate
  `cargo test` / `cargo clippy` matrix in CI; end-to-end Rust↔Python parity
  coverage is a tracked hardening item, not yet part of the 100 % gate.

## CI Gates (`.github/workflows/ci.yml`)

| Gate | Requirement |
|------|-------------|
| pytest | Full suite passes on 3.10 / 3.11 / 3.12 (`ci.yml` `test` job) |
| coverage | `--cov-fail-under=100` on the product modules (3.12) |
| ruff check | Zero lint errors (`E,F,W,I,UP,B,SIM`) |
| ruff format | All files formatted (`ruff format --check`) |
| mypy | Advisory per-module type check (strict ratchet in progress) |
| bandit | SAST clean on the security-relevant module set |
| REUSE | Every file carries an SPDX licence marker |
| rust | `cargo fmt --check` + `cargo test` + `cargo clippy -D warnings` per crate (16) |

## Benchmark Results (honest, realistic setting)

LongMemEval is the primary benchmark. Numbers are reported on the **realistic
full-S** setting (full haystack, retrieved context) — never the oracle setting,
which inflates scores by ~30 % and is not comparable to SOTA full-S leaderboards.

| Benchmark | Setting | Reader | Score | Notes |
|-----------|---------|--------|------:|-------|
| LongMemEval full-S | realistic | cloud gpt-4o-mini | **56.4 %** | committed scored artefact (`benchmarks/longmemeval_full_s_seed42_snapshot_report.json`, seed 42, full judge evidence); consistent with the earlier 3-run mean 56.6 % (spread 2.2). Single-seed → headline-ineligible in the manifest until a second-seed run |
| LongMemEval full-S | realistic | sovereign local `gemma3:4b` | **35.4 %** | no-egress; −21 pp vs cloud — the honest local cost |
| LongMemEval (retrieval) | — | GPT-free | **~81 %@10** | recall; multi-session 88 %@10 — retrieval is strong, synthesis is the gap |
| LongMemEval oracle | oracle (NOT comparable) | cloud | 72.2 % (R11) | historical; retained only as an oracle reference, not a headline |
| LOCOMO | — | no-LLM baseline | 74.7 % | historical (2026-03), not independently reproducible from committed code |
| LOCOMO | — | with LLM synthesis | 83.1 % | historical |

The gap to the realistic-SOTA cluster (≈83–95 %) is the answer-synthesis layer, not
retrieval. See `docs/benchmarks/LongMemEval.md` and
`docs/benchmarks/SOVEREIGN_MEMORY_EVALUATION.md` for methodology and the
sovereign / new-category axes.
