# Changelog

All notable changes to Grumpy are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.1] - 2026-05-29

### Added

- ML dataloader streaming with partial leaf I/O, `batch_on`, batch-order shuffle, within-batch shuffle, DDP sharding, and I/O prefetch workers.
- `Stream.__getitem__` for subset batch iteration (int, slice, or sequence of batch indices).
- `gr.save(generator, ...)` incremental writes via `append_batch` (load + concat + rewrite per batch).
- `chunk_dim` on save: target a schema level or numeric depth for Zarr chunk sizing.
- `compiled_stream_apply` uses partial batch loads (parity with `Stream.__iter__`).
- **`UnionScalarList`**: compact axis-0 slice, streaming load/save, scalar in-place ops, sum/mean, unique, shuffle, axis-0 concat/append (with list-chain lifting).

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
