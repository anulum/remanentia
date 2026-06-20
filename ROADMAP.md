# Roadmap

## Completed (v0.3.1)

- [x] LongMemEval benchmark: 72.2% (361/500) at R11, results committed
- [x] LOCOMO benchmark: 83.1% (1,651/1,986) with LLM synthesis; 74.7% no-LLM baseline
- [x] PyPI publish workflow (OIDC trusted publisher, `publish.yml`)
- [x] 100% coverage gate on product modules (2,166 test functions across 62 files)
- [x] MCP server with 4 tools (recall, remember, status, graph)
- [x] ArcaneRetriever 4-channel parallel retrieval with RRF fusion
- [x] Atomic fact decomposer with temporal validity windows
- [x] Enterprise hardening (CI, CodeQL, Scorecard, REUSE, CITATION.cff)
- [x] Documentation site (MkDocs, 18 API refs, 5 guides, 2 benchmarks)
- [x] 16 Rust acceleration crates (PyO3 + maturin, Python fallback, up to 76× speedup)
- [x] Pluggable LLM backend (Auto, Local, hosted, Null)
- [x] 8-model local LLM benchmark (Qwen 2.5 3B recommended, ROCm)
- [x] Pipeline performance documented (0.6ms regex pipeline, 27 budget tests)
- [x] SPDX header standardisation (103 Python files)

## v0.4.0 — Temporal Breakthrough

The primary target: temporal-reasoning accuracy from 45.9% toward 70%+.

- [x] Temporal pipeline (rule-based date normaliser + reference-date fix): temporal-reasoning 45.9% (R8) → 65.4% (R11, +19.5pp)
- [x] Tier 1–3 + recall Rust rustification complete (PyO3 + maturin)
- [ ] Date arithmetic in answer extraction (TReMu code execution for ordering/counting)
- [ ] Multi-session entity tracking (supersession chains across sessions)
- [ ] LongMemEval R9+ targeting 75%+ overall
- [ ] Commit LOCOMO results for reproducibility
- [ ] Multi-user namespace support (user_id isolation)

## v0.5.0 — Production Hardening

- [ ] PyPI publish (v0.4.0 or v0.5.0 — first release with committed benchmark results)
- [ ] Cloud-hosted demo at remanentia.com/demo
- [ ] Docker deployment with health checks
- [ ] REST API authentication
- [ ] Rate limiting on MCP server

## v1.0.0 — General Availability

- [ ] LongMemEval score >= 85%
- [ ] Multi-tenant deployment guide
- [ ] Web UI / knowledge graph viewer
- [ ] npm package for JS/TS clients
- [ ] Academic paper submission

## Research Track

- [ ] Temporal-reasoning gap (60.2% → 70%+ on LongMemEval)
- [ ] Multi-session gap (61.7% → 80%+ on LongMemEval)
- [ ] SNN contribution to consolidation quality (currently novelty detection only)
- [ ] Neuromorphic hardware proof-of-concept (Intel Lava)
- [ ] Cross-encoder fine-tuning on conversational memory domain
