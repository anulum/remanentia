.PHONY: test test-all test-cov test-cov-combined build-rust verify-snn-floors lint fmt preflight preflight-fast docs docs-build install-hooks bench clean build

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
	@wheel_root=$$(mktemp -d); \
	trap 'rm -rf "$$wheel_root"' EXIT; \
	for crate in $(CRATES); do \
		wheel_dir="$$wheel_root/$$(basename "$$crate")"; \
		mkdir -p "$$wheel_dir"; \
		echo "== maturin build $$crate =="; \
		$(PYTHON) -m maturin build --release --manifest-path "$$crate/Cargo.toml" \
			--interpreter "$(PYTHON)" --out "$$wheel_dir" || exit 1; \
		set -- "$$wheel_dir"/*.whl; \
		[ "$$#" -eq 1 ] && [ -f "$$1" ] || { echo "expected one wheel for $$crate" >&2; exit 1; }; \
		$(PYTHON) -m pip install --force-reinstall --no-deps "$$1" || exit 1; \
	done

# Prove the installed-wheel SNN-memory coverage floors that live OUTSIDE the
# default CI core (omitted in pyproject): each verifier builds + installs a
# fresh wheel and enforces its floor (D1/D2/D3 = 100 %, D4-A >= 95 % with
# preregistered A/B/C residuals, stream/model-gates = 100 %). Heavy; D2 and the
# model gates need the pinned .snn_models encoder present. Run before tagging a
# release — this is the automated proof that the omitted modules are at floor.
SNN_FLOOR_VERIFIERS = source_universe_d1 cue_materializer_d2 d3 d4a stream_stage1 model_gates

verify-snn-floors:
	@for v in $(SNN_FLOOR_VERIFIERS); do \
		echo "== verify_snn_memory_$$v ==" ; \
		$(PYTHON) tools/verify_snn_memory_$$v.py || exit 1 ; \
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
