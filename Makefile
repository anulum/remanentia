.PHONY: test test-all lint fmt preflight preflight-fast docs docs-build install-hooks bench clean build

PYTHON ?= python

test:
	$(PYTHON) -m pytest tests/ -q --tb=short

test-cov:
	$(PYTHON) -m pytest tests/ --cov=. --cov-config=pyproject.toml --cov-report=term-missing -q --tb=short

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
