.PHONY: test test-all test-cov test-cov-combined build-rust lint fmt preflight preflight-fast docs docs-build install-hooks bench clean build

PYTHON ?= python

# Maturin crates whose compiled extensions back the fast dispatch paths.
CRATES = remanentia_topology rust_active_retrieval rust_aggregate_precompute \
	rust_answer_extractor rust_answer_normalizer rust_consolidation \
	rust_entity_extractor rust_fact_decomposer rust_knowledge_store \
	rust_pii_redactor rust_recall rust_retrieve rust_search \
	rust_skill_extractor rust_snn_memory rust_stdp rust_temporal

test:
	$(PYTHON) -m pytest tests/ -q --tb=short

test-cov:
	$(PYTHON) -m pytest tests/ --cov=. --cov-config=pyproject.toml --cov-report=term-missing -q --tb=short

# De-hollow coverage gate: Python-fallback pass UNION native-Rust-dispatch pass.
# Requires the Rust extensions to be importable — run `make build-rust` first.
test-cov-combined:
	PYTHON=$(PYTHON) bash tools/run_combined_coverage.sh

build-rust:
	@for crate in $(CRATES); do \
		echo "== maturin develop $$crate ==" ; \
		$(PYTHON) -m maturin develop --release --manifest-path $$crate/Cargo.toml || exit 1 ; \
	done

test-all: test

lint:
	$(PYTHON) -m ruff check . --fix
	$(PYTHON) -m ruff format --check .

fmt:
	$(PYTHON) -m ruff format .
	$(PYTHON) -m ruff check . --fix

bandit:
	$(PYTHON) -m bandit -r memory_index.py mcp_server.py consolidation_engine.py cli.py api.py -q

sast: bandit

preflight: lint bandit test-cov

preflight-fast: lint bandit

install-hooks:
	git config core.hooksPath .githooks

docs:
	mkdocs serve

docs-build:
	mkdocs build --strict

bench:
	$(PYTHON) bench_locomo.py

build:
	$(PYTHON) -m build

clean:
	rm -rf build/ dist/ *.egg-info/ __pycache__/ .coverage
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
