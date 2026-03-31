# Roadmap

## Completed (v0.3.1)

- [x] LongMemEval benchmark: 69.0% (345/500), results committed
- [x] LOCOMO benchmark: 74.7% (1,986 questions, no LLM)
- [x] PyPI publish workflow (OIDC trusted publisher, `publish.yml`)
- [x] 100% test coverage (1,343 tests, 19 modules, zero lines missing)
- [x] MCP server with 4 tools (recall, remember, status, graph)
- [x] ArcaneRetriever 4-channel parallel retrieval with RRF fusion
- [x] Atomic fact decomposer with temporal validity windows
- [x] Enterprise hardening (CI, CodeQL, Scorecard, REUSE, CITATION.cff)
- [x] Documentation site (MkDocs, 18 API refs, 5 guides, 2 benchmarks)
- [x] 7 Rust acceleration crates (PyO3 + maturin, up to 14× speedup)
- [x] Pluggable LLM backend (Auto, Local, Anthropic, Null)
- [x] 8-model local LLM benchmark (Qwen 2.5 3B recommended, ROCm)
- [x] Pipeline performance documented (0.6ms regex pipeline, 27 budget tests)
- [x] SPDX header standardisation (103 Python files)

## v0.4.0 — Temporal Breakthrough

The primary target: temporal-reasoning accuracy from 45.9% toward 70%+.

- [ ] Temporal fact retrieval via FactIndex validity windows (infrastructure built, needs benchmark integration)
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

- [ ] Temporal-reasoning gap (45.9% → 70%+ on LongMemEval)
- [ ] Multi-session gap (61.7% → 80%+ on LongMemEval)
- [ ] SNN contribution to consolidation quality (currently novelty detection only)
- [ ] Neuromorphic hardware proof-of-concept (Intel Lava)
- [ ] Cross-encoder fine-tuning on conversational memory domain
