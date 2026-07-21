# Contributing to Remanentia

SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.

## Getting Started

```bash
# Fork and clone
git clone https://github.com/YOUR_USERNAME/remanentia.git
cd remanentia

# Create a feature branch
git checkout -b feature/your-feature

# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -q --tb=short

# Run with coverage (quick, single pass)
pytest tests/ --cov=. --cov-config=pyproject.toml -q

# The authoritative 100% gate — combined coverage (Python fallback UNION native
# Rust dispatch), so both paths are measured with no dispatch pragmas.
make build-rust        # build the Rust extensions once (needed for pass B)
make test-cov-combined
```

## Code Style

- Python 3.10+ with type hints on public API boundaries
- No `# noqa` comments — fix the code instead
- Use the repository's current seven-line SPDX/commercial header on source files;
  copy it from a neighbouring file of the same type.
- Anti-slop policy: no narration comments, no defensive boilerplate walls,
  no buzzword naming, no trivial wrappers

## Testing

- 100% coverage gate on product modules via `make test-cov-combined` (Python
  fallback UNION native Rust dispatch — both measured, no dispatch pragmas)
- Tests in `tests/` with `pytest`
- New code = new tests. No exceptions.
- Run with CI's Python version (3.12)
- The installed-wheel SNN-memory modules are omitted from the default coverage
  denominator (they need a built + installed wheel) and are proven at their
  floors by their own verifiers. Run `make verify-snn-floors` before tagging a
  release — it builds each wheel and enforces the floor (D1/D2/D3 = 100 %,
  D4-A >= 95 % with preregistered residuals, stream/model-gates = 100 %). D2 and
  the model gates need the pinned `.snn_models` encoder present.

## Rust Modules

PyO3 acceleration modules live in the tracked Rust directories. Build and
install the complete wheel set through the repository's canonical target:

```bash
make build-rust PYTHON=python
make test-cov-combined PYTHON=python
```

The Makefile builds each module as a uniquely named wheel before installation;
do not replace this with a root-level `maturin develop` command.

## Pull Requests

- One logical change per PR
- Tests must pass
- Coverage must not decrease
- SPDX headers on new files
- Commit messages describe the change, not who found the issue

## Reporting Issues

- [GitHub Issues](https://github.com/anulum/remanentia/issues)
- Security vulnerabilities: see SECURITY.md

## License

By contributing, you agree that your contributions will be licensed under
AGPL-3.0-or-later, consistent with the project license.
