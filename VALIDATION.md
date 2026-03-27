# Validation

## Test Matrix

| Python | Platform | Status |
|--------|----------|:------:|
| 3.12 | Ubuntu (CI) | ✅ |
| 3.12 | Windows 11 (local) | ✅ |

## Test Counts

- **669 tests** across 17 test files
- **672 tests** with coverage (3 additional coverage-specific tests)

## Coverage Gate

- **Target:** 100% on product modules
- **Current:** 96.36% (memory_index.py at 89% due to model-dependent paths)
- **Config:** `pyproject.toml` `[tool.coverage.report] fail_under = 100`

### Coverage by Module

| Module | Coverage |
|--------|:--------:|
| api.py | 100% |
| cli.py | 100% |
| memory_recall.py | 100% |
| reflector.py | 100% |
| entity_extractor.py | 100% |
| answer_normalizer.py | 100% |
| answer_extractor.py | 99% |
| consolidation_engine.py | 99% |
| knowledge_store.py | 99% |
| mcp_server.py | 99% |
| observer.py | 98% |
| temporal_graph.py | 98% |
| memory_index.py | 89% |

### Excluded from Coverage

- Legacy/dead modules: snn_daemon, snn_backend, monitor, gpu_daemon, retrieve, encoding
- Benchmark runners: bench_locomo, bench_experiments, run_exp*.py, analyze_failures
- Test files and conftest

## CI Gates

| Gate | Requirement |
|------|-------------|
| pytest | 669 tests pass |
| coverage | 70% (CI gate, local gate 100%) |

## Benchmark Results

| Benchmark | Score | Date |
|-----------|-------|------|
| LOCOMO (exp8b) | 88.5% | 2026-03-25 |
| Internal P@1 | 92.9% | 14 self-authored queries |
| LongMemEval | Not yet run | — |
