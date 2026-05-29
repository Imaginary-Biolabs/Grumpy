# Changelog

All notable changes to Grumpy are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - 2026-05-29

### Added

- Initial public alpha release engineering: CI, docs site skeleton, benchmarks runner.
- `__version__` synced with `pyproject.toml` (0.1.0).
- `gr.compile` exported from the top-level package.
- Streaming uses `stored_len` / `load_slice` for per-batch loading.
- CONTRIBUTING, SECURITY, and expanded README.

### Changed

- `.gitignore` no longer ignores `.github/`; compiled `_core*.so` artifacts are excluded.
- `Cargo.lock` committed for reproducible builds.

[0.1.0]: https://github.com/imaginary-bio/grumpy/releases/tag/v0.1.0
