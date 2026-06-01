# Changelog

All notable changes to Grumpy are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.2] - Unreleased

### Added

- Union `quantile` / `median` for `dim=0` and innermost axis on `UnionScalarList` layouts (per-row leaf quantile).
- Union coordinate assignment (`x[i, j] = value`) via layout-native `leaf_mut_at_coord`.
- Benchmark smoke test (`benchmarks/ci_smoke.py`) in CI to catch large performance regressions.
- Structured error reporting across einsum, linalg, neighbors, histogram, compare, unary, setops, and whereops.

### Changed

- CI coverage job runs all tests (including `@pytest.mark.coverage`) so the 95% gate is meaningful.
- `gr.cat(..., dim>0)` with unions still uses Python list merge; layout-native path remains future work.
- Deep list-chain reductions share the no-GIL engine (`reduce_list_chain_to_layout_nogil`).
- Clippy `correctness` and `suspicious` lints enabled at warn level.

### Fixed

- Coverage CI would fail at ~74% because compiler coverage tests were excluded by default pytest markers.

## [0.1.1] - 2026-05-29

### Added

- ML dataloader streaming with partial leaf I/O, `batch_on`, batch-order shuffle, within-batch shuffle, DDP sharding, and I/O prefetch workers.
- `Stream.__getitem__` for subset batch iteration (int, slice, or sequence of batch indices).
- `gr.save(generator, ...)` incremental writes via `append_batch` (load + concat + rewrite per batch).
- `chunk_dim` on save: target a schema level or numeric depth for Zarr chunk sizing.
- `compiled_stream_apply` uses partial batch loads (parity with `Stream.__iter__`).
- **`UnionScalarList`**: compact axis-0 slice, streaming load/save, partial I/O, `batch_on`, scalar in-place ops, reductions/stats/sort/search, broadcast, neighbors (`dim=0`), einsum (1D), unique, shuffle, axis-0 concat/append (with list-chain lifting). Documented as first-class alongside list-chains.

### Changed

- `load_slice` / stream batch loads read only the leaf ranges needed for each batch.
- Documentation updated for streaming, partial I/O, and new save options.

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

[0.1.1]: https://github.com/imaginary-bio/grumpy/releases/tag/v0.1.1
[0.1.0]: https://github.com/imaginary-bio/grumpy/releases/tag/v0.1.0
