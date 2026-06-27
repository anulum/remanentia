# Validation

## Test Matrix

| Python | Platform | Status |
|--------|----------|:------:|
| 3.10 | Ubuntu (CI) | ✅ |
| 3.11 | Ubuntu (CI) | ✅ |
| 3.12 | Ubuntu (CI) | ✅ |
| 3.12 | Windows 11 (local) | ✅ |

## Test Suite

- **2,005 tests** across 53 test files
- **100% coverage** — product modules, zero lines missing
- **Config:** `pyproject.toml` `[tool.coverage.report] fail_under = 100`

## Coverage by Module

| Module | Stmts | Cover |
|--------|------:|:-----:|
| memory_index.py | 928 | 100% |
| consolidation_engine.py | 499 | 100% |
| knowledge_store.py | 395 | 100% |
| fact_decomposer.py | 266 | 100% |
| answer_extractor.py | 245 | 100% |
| temporal_graph.py | 341 | 100% |
| memory_recall.py | 207 | 100% |
| cli.py | 196 | 100% |
| mcp_server.py | 192 | 100% |
| arcane_retriever.py | 179 | 100% |
| date_normalizer.py | 141 | 100% |
| llm_backend.py | 141 | 100% |
| reflector.py | 127 | 100% |
| api_server.py | 190 | 100% |
| observer.py | 104 | 100% |
| api.py | 224 | 100% |
| temporal_relation.py | 88 | 100% |
| answer_normalizer.py | 78 | 100% |
| entity_extractor.py | 77 | 100% |
| fact_validity_model.py | 69 | 100% |
| llm_setup.py | 33 | 100% |
| **Total** | **~4,563** | **100%** |

## Excluded from Coverage

Legacy/experimental modules (not part of the product surface):

- `snn_daemon.py`, `snn_backend.py` — SNN experimental, excluded from product gate
- `monitor.py`, `retrieve.py`, `encoding.py` — dead/legacy code
  (`gpu_daemon.py` + `build_memory_standalone.py` archived to 00_SAFETY_BACKUPS 2026-06-24)
- `git_stimulus.py`, `heartbeat_register.py`, `hooks.py` — infrastructure scripts
- `pattern_separation.py`, `cognitive_snapshot.py`, `active_retrieval.py`, `skill_extractor.py` — experimental
- `bench_locomo.py`, `bench_longmemeval.py`, `bench_experiments.py`, `run_exp*.py` — benchmark runners
- `extractors/*`, `experimental/*` — experimental directories
- `training/*` — training infrastructure (not product code)

## CI Gates

| Gate | Requirement |
|------|-------------|
| pytest | 1,447 tests pass (3.10, 3.11, 3.12) — CI excludes Rust crate tests |
| coverage | 100% fail-under on product modules |
| ruff check | Zero lint errors across 16 modules + tests |
| ruff format | All files formatted |
| bandit | Zero high/medium findings (skips: B101, B110, B112, B301, B310, B311, B324, B403, B404, B603, B607) |

## Benchmark Results

| Benchmark | Score | Questions | Committed | Date |
|-----------|------:|----------:|:---------:|------|
| LongMemEval R8 | 69.0% | 500 | Yes (`data/longmemeval_hypotheses.results.jsonl`) | 2026-03-27 |
| LOCOMO (no LLM) | 74.7% | 1,986 | No (experiment runs only) | 2026-03-25 |
| Internal P@1 | 76.0% | 50 | Yes (`paper/internal_benchmark.json`) | 2026-03-26 |

LongMemEval is the primary benchmark. LOCOMO results are historical and not independently reproducible from committed code.
