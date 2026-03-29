# Validation

## Test Matrix

| Python | Platform | Status |
|--------|----------|:------:|
| 3.10 | Ubuntu (CI) | ✅ |
| 3.11 | Ubuntu (CI) | ✅ |
| 3.12 | Ubuntu (CI) | ✅ |
| 3.12 | Windows 11 (local) | ✅ |

## Test Suite

- **1,049 tests** across 28 test files
- **100% coverage** — 19 product modules, zero lines missing
- **Config:** `pyproject.toml` `[tool.coverage.report] fail_under = 100`

## Coverage by Module

| Module | Stmts | Cover |
|--------|------:|:-----:|
| memory_index.py | 888 | 100% |
| knowledge_store.py | 357 | 100% |
| consolidation_engine.py | 294 | 100% |
| answer_extractor.py | 217 | 100% |
| fact_decomposer.py | 216 | 100% |
| memory_recall.py | 207 | 100% |
| temporal_graph.py | 197 | 100% |
| mcp_server.py | 185 | 100% |
| cli.py | 173 | 100% |
| arcane_retriever.py | 138 | 100% |
| api_server.py | 117 | 100% |
| reflector.py | 108 | 100% |
| api.py | 95 | 100% |
| observer.py | 86 | 100% |
| entity_extractor.py | 73 | 100% |
| answer_normalizer.py | 72 | 100% |
| date_normalizer.py | 139 | 100% |
| temporal_relation.py | 88 | 100% |
| fact_validity_model.py | 69 | 100% |
| **Total** | **~3,720** | **100%** |

## Excluded from Coverage

Legacy/experimental modules (not part of the product surface):

- `snn_daemon.py`, `snn_backend.py` — SNN experimental, excluded from product gate
- `monitor.py`, `gpu_daemon.py`, `retrieve.py`, `encoding.py` — dead/legacy code
- `git_stimulus.py`, `heartbeat_register.py`, `hooks.py` — infrastructure scripts
- `pattern_separation.py`, `cognitive_snapshot.py`, `active_retrieval.py`, `skill_extractor.py` — experimental
- `bench_locomo.py`, `bench_longmemeval.py`, `bench_experiments.py`, `run_exp*.py` — benchmark runners
- `extractors/*`, `experimental/*` — experimental directories
- `training/*` — training infrastructure (not product code)

## CI Gates

| Gate | Requirement |
|------|-------------|
| pytest | 844 tests pass (3.10, 3.11, 3.12) |
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
