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

# Run with coverage
pytest tests/ --cov=. --cov-config=pyproject.toml -q
```

## Code Style

- Python 3.10+ with type hints on public API boundaries
- No `# noqa` comments — fix the code instead
- SPDX 6-line header on every source file (see root CLAUDE.md)
- Anti-slop policy: no narration comments, no defensive boilerplate walls,
  no buzzword naming, no trivial wrappers

## Testing

- 2,005 tests across 53 files, 100% coverage gate on product modules
- Tests in `tests/` with `pytest`
- New code = new tests. No exceptions.
- Run with CI's Python version (3.12)

## Rust Modules

13 PyO3 crates (54 functions). Sources live in `workspace-internal/rust_*/` (external)
and `rust_recall/`, `rust_search/` (in-repo). Build with maturin:

```bash
cd rust_recall  # or any rust_* directory
VIRTUAL_ENV=.venv maturin develop --release

# Verify
python -c "from remanentia_recall import tokenize_words; print('OK')"
```

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
