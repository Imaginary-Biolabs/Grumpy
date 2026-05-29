# Contributing to Grumpy

Thank you for helping improve Grumpy. This project is developed by [Imaginary Biolabs](https://www.imaginary.bio).

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip maturin
maturin develop --release
python -m pip install -e ".[dev]"
pytest
```

Rust changes require a recent stable toolchain. `Cargo.lock` is committed for reproducible CI builds.

## Pull requests

1. Fork or branch from `main`.
2. Add tests for behavior changes; Python package coverage for `python/grumpy/` should stay at 100%.
3. Run `pytest` and, when touching hot paths, relevant benchmarks under `benchmarks/`.
4. Update `CHANGELOG.md` under **Unreleased** (or the next version section).
5. Keep diffs focused; match existing naming and module layout.

## Code style

- Rust: performance-first kernels in `src/`, thin bindings in `py_api.rs`.
- Python: thin wrappers in `python/grumpy/__init__.py`; streaming/compiler logic in dedicated modules.
- Docstrings: NumPy-style for public Python APIs.

## Reporting issues

Use GitHub issues with a minimal reproducer (Python version, OS, and `grumpy.__version__`). For security reports, see [SECURITY.md](SECURITY.md).
