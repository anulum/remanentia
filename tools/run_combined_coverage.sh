#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
#
# Combined coverage gate — the de-hollow measurement.
#
# Pass A runs the suite with the compiled Rust extensions made unimportable
# (REMANENTIA_COVERAGE_NO_RUST=1, honoured by tests/conftest.py) so the dispatch
# modules take their pure-Python fallbacks — identical to the setuptools-only CI
# test job. Pass B runs it again with the extensions present so the native
# dispatch paths execute. Combining the two data files measures BOTH code paths,
# so every dispatch module reaches 100% without a single
# `# pragma: no cover - native dispatch`.
#
# Pass B needs the Rust extensions built and importable — run `make build-rust`
# first (or use a virtualenv that already has them installed).
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"

rm -f .coverage .coverage.*

echo "== combined coverage — pass A: Python fallback (Rust extensions blocked) =="
REMANENTIA_COVERAGE_NO_RUST=1 "$PY" -m coverage run --parallel-mode \
	--rcfile=pyproject.toml -m pytest tests/ -q --tb=short

echo "== combined coverage — pass B: native Rust dispatch (extensions present) =="
"$PY" -m coverage run --parallel-mode \
	--rcfile=pyproject.toml -m pytest tests/ -q --tb=short

"$PY" -m coverage combine
"$PY" -m coverage report --rcfile=pyproject.toml --show-missing --fail-under=100
