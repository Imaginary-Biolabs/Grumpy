# Developer

This page is for contributors and advanced users who need to understand how Grumpy is structured, how layouts map to Rust kernels, and how errors are reported. End-user tutorials start on the [Home](index.md) page.

## Repository structure

```
Grumpy/
├── python/grumpy/       # Python package: public API, stream, compiler, doc injection
│   ├── __init__.py      # Thin wrappers around _core
│   ├── stream.py        # Stream / StreamApply, batch scheduling
│   ├── compiler.py      # @gr.compile static analysis → CompiledPlan
│   ├── errors.py        # Python-side grumpy.* errors
│   └── _docinject.py    # Injected docstrings for GrumpyArray methods
├── src/                 # Rust library (PyO3 extension grumpy._core)
│   ├── layout/          # GrumpyArray layout tree, builders, gather/scatter
│   ├── ops.rs           # Elementwise / broadcast kernels
│   ├── neighbors.rs     # kNN and radius search
│   ├── geometry/        # CPU pairwise_distances, grid_pool
│   ├── gpu/             # Optional Metal/CUDA kNN
│   ├── stream/          # Zarr batch iterator, partial I/O
│   ├── py_api/          # PyO3 bindings (keep thin; call src/ kernels)
│   └── error.rs         # ErrorCode, cause/fix formatting
├── docs/                # MkDocs site (this documentation)
├── benchmarks/          # Public API and engine benchmarks
└── tests/               # pytest suite
```

Build the extension with [maturin](https://www.maturin.rs/):

```bash
maturin develop --release
pytest
```

See [CONTRIBUTING.md](https://github.com/Imaginary-Biolabs/Grumpy/blob/main/CONTRIBUTING.md) for PR expectations (≥95% coverage on `python/grumpy/`, changelog entries).

## Core concepts

### Layout tree

Every `GrumpyArray` is a tree ending in typed **leaf** buffers:

- **`ListOffset`** — variable-length lists; children share one leaf pool addressed by offset arrays.
- **`UnionScalarList`** — one axis where each row is either a scalar or a list (tag + index pools).
- **`Leaf`** — homogeneous `int32`, `float64`, `string`, etc., plus an optional validity bitmap for nulls.

Python nested lists are **materialized once** at construction (or load); hot paths never re-walk Python objects.

### List-chain vs union

| Layout | Python shape intuition | Typical use |
|--------|------------------------|-------------|
| List-chain | `[[…], […]]` fixed depth | Coordinates, atom tables |
| Union | `[1, [2, 3], 4]` mixed rows | GO terms, isoforms, nullable mixes |

Kernels implement both paths where supported; the [arrays](arrays.md) tutorial introduces user-facing behavior.

### Dataframes

A `GrumpyDataFrame` stores columns as arrays plus optional **schema** metadata. Shared list offsets at a schema level are stored once. Dot notation maps to `drop_layout_axes` in Rust.

### Streaming and I/O

`gr.save` writes Zarr groups with a `grumpy.json` manifest. `StreamBatchesIter` computes leaf byte ranges per batch so union and list-chain stores read only needed chunks.

### Compilation IR

`compiler.py` parses AST → JSON opcodes → `CompiledPlan` in Rust. `compiled_stream_apply` runs opcodes with optional Rayon over batches.

## Implementation notes

- **Hot paths release the GIL** via `py.allow_threads` in PyO3 wrappers.
- **Broadcasting** walks layout trees in `ops.rs`; union outer lengths must match or broadcast from length 1.
- **Neighbors** return edge indices `(src, dst)` suitable for graph construction; optional GPU brute-force kNN (`gpu=True`, `False`, or `"auto"` on `gr.neighbors`).
- **Docstrings** — top-level functions use `_docinit.py`; `GrumpyArray` methods use `_docinject.py` so mkdocstrings generates the [API Reference](api.md).

When adding a kernel:

1. Implement in `src/<module>.rs` (no GIL on the compute path).
2. Expose via `src/py_api/`.
3. Support list-chain and union if the op is user-facing (or document the gap).
4. Add pytest + error tests with `cause:` / `fix:` assertions.

## Dtypes and casting

Arrays carry one dtype on all leaves. Inference: Python `int` → `int64`, `float` → `float64`.

`GrumpyArray.astype(dtype, casting='safe')` preserves layout:

| Mode | Behavior |
|------|----------|
| `safe` | Widen numerics; bool → numeric; char → string |
| `same_kind` | Safe plus float narrowing; integer narrowing with overflow errors |
| `unsafe` | All numeric casts; narrowing wraps |

Binary ops promote with NumPy `promote_types` rules. Null slots stay null across casts.

## Error handling

Grumpy errors are designed to be **actionable**:

```text
grumpy.<Code>: <one-line summary>
  cause: <root constraint that was violated>
  fix: <concrete remediation>
```

Example:

```text
grumpy.BroadcastFailed: incompatible union outer lengths 3 and 4
  cause: UnionScalarList broadcasting requires equal outer length, or one side with outer length 1.
  fix: align outer lengths, insert a length-1 axis, or reshape so one array broadcasts.
```

### Error codes

| Code | Typical use |
|------|-------------|
| `ArgumentInvalid` | Bad function arguments (`batch_size`, `compile`, …) |
| `BroadcastFailed` | Incompatible shapes for elementwise / broadcast |
| `CastNotAllowed` | `astype(..., casting='safe')` rejected a conversion |
| `ConcatIncompatible` | `gr.cat` cannot merge layouts or dtypes |
| `DtypeMismatch` | Operands require matching dtypes |
| `IndexOutOfBounds` | Indexing, slicing, or batch index out of range |
| `IoFailed` | Zarr / filesystem read or write problems |
| `LayoutUnsupported` | Op does not support this layout |
| `ReduceDimInvalid` | Invalid `dim` for reduction |
| `ReduceEmpty` | Reduction over empty or all-null data |
| `SchemaViolation` | DataFrame schema / column shape constraints |
| `ShapeMismatch` | Reshape, unflatten, or axis length mismatch |
| `Unsupported` | Valid call but not implemented for this dtype/layout |
| `InternalError` | Unexpected invariant violation (please report) |

Python exceptions subclass `ValueError` or `IndexError` where appropriate so existing tests keep working.

### Contributor checklist

1. Use helpers — Rust: `crate::error::{…}`; Python: `grumpy.errors.raise_grumpy_error`.
2. Pick the closest `ErrorCode`.
3. State **cause** (what invariant failed) and **fix** (what the user should do).
4. Add context (axis, dtype, column name, path) when helpful.
5. Test messages in `tests/test_errors.py` for user-facing failures.

Implementation: [`src/error.rs`](https://github.com/Imaginary-Biolabs/Grumpy/blob/main/src/error.rs), [`python/grumpy/errors.py`](https://github.com/Imaginary-Biolabs/Grumpy/blob/main/python/grumpy/errors.py).

## Documentation site

```bash
pip install -e ".[dev]"
mkdocs serve -f mkdocs.yml
```

Homepage benchmark charts regenerate on build via `docs/hooks.py` → `benchmarks/generate_perf_charts.py`.

---

**Next:** [License FAQ](license-faq.md) — Business Source License 1.1 terms and commercial use.
