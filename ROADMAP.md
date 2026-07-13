# Roadmap

Public roadmap for Remanentia. Benchmark numbers follow the honest **realistic
full-S** convention (never the oracle setting); see `VALIDATION.md` and
`docs/benchmarks/` for methodology.

## Shipped (v0.5.0)

- [x] Honest full-S LongMemEval baseline: **56.6 %** (cloud gpt-4o-mini, 3-run mean)
- [x] First sovereign no-egress full-S run: **35.4 %** (local `gemma3:4b`, −21 pp) —
  the honest cost of staying local
- [x] GPT-free retrieval recall ~81 %@10 (multi-session 88 %@10)
- [x] ArcaneRetriever 4-channel parallel retrieval (BM25 + entity + session +
  temporal) with RRF fusion + cross-encoder rerank
- [x] Atomic fact decomposer with bi-temporal validity windows; entity + temporal
  graphs; signed finding envelopes with supersession-closure verification
- [x] Write-discipline enforcement gate (canonical stimulus schema, per-producer
  quarantine ledger)
- [x] World-class evaluation harness (calibrated abstention, no-egress audit,
  lineage completeness, scorecard + `remanentia-scorecard` CLI)
- [x] MCP server (4 tools) with stdio rate limiting; REST API with bearer auth,
  `--require-auth`, rate limiting, PII redaction
- [x] Docker deployment (non-root, digest-pinned, healthcheck, hash-pinned install)
  + OpenAPI export
- [x] 16 Rust acceleration crates (PyO3 + maturin, Python fallback)
- [x] 100 % product-module coverage gate; 73 API reference pages; ADR log; model
  cards; CycloneDX SBOM + sigstore + SLSA provenance release pipeline

## In progress — World-Class Hardening (2026-07)

Two objectives: (A) match measurable SOTA on the realistic full-S setting, and (B)
define and own a new evaluation category no leaderboard measures.

- [ ] **Reader synthesis is the gap, not retrieval** — lean bi-temporal
  observe→reflect context (validated +8.8 pp on affected categories with a strong
  cloud reader); target cloud full-S ≥ 80 %
- [ ] **Sovereign no-egress accuracy** with a stronger local reader
  (`gemma3:12b` / Qwen); target pure-local 70–75 % and publish the honest cost
- [ ] **New-category axes:** write-discipline→accuracy curve, calibrated abstention
  (coverage-accuracy), fleet-fed multi-producer recall, lineage-of-belief
  auditability (every answer traces to a queryable provenance record)
- [ ] **Make every quality claim real:** exercise the Rust fast-path in CI
  (build + parity coverage), close the lineage producer loop, hash-pin dependencies
  with a supply-chain gate, strict-mypy + numpy-docstring enforcement

## v1.0.0 — General Availability

- [ ] Realistic full-S LongMemEval ≥ 80 % cloud / 70–75 % sovereign, with committed
  multi-seed variance artefacts
- [ ] Category-defining scorecard lit end-to-end on a committed run
- [ ] Multi-tenant deployment guide; web UI / knowledge-graph viewer
- [ ] npm package for JS/TS clients; academic paper submission

## Research Track

- [ ] Temporal-reasoning and multi-session synthesis gaps on LongMemEval
- [ ] LongMemEval-V2 / BEAM agentic evals (unsaturated, credibility-winnable)
- [ ] Principled forgetting / decay + admission control
- [ ] Sleep-time / background consolidation as a built-in
- [ ] Cross-encoder fine-tuning on conversational-memory domain
- [ ] Neuromorphic hardware proof-of-concept (Intel Lava)
