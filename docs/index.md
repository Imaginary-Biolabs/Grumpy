## Highlights

- Layout-first kernels for elementwise ops, reductions, neighbors, and more — **list-chains** and **`UnionScalarList`** are both first-class
- `GrumpyDataFrame` with optional schema and dot-notation column access (list-chain and union columns)
- Zarr save/load and axis-0 streaming with optional `gr.compile` fusion
- BSL 1.1 license (Imaginary Biolabs GmbH) — [License FAQ](license-faq.md)

## Quick links

- [Getting started](getting-started.md)
- [Arrays](arrays.md)
- [Dtypes and casting](dtypes.md)
- [Error reporting](errors.md)
- [Dataframes](dataframes.md)
- [Saving and loading](saving-loading.md)
- [Compilation](compilation.md)
- [API Reference](api.md)

## Performance

Representative **public API** timings on slightly ragged data (Grumpy, Awkward) vs rectangular NumPy with the same leaf count. Bar groups are **Grumpy · NumPy · Awkward**; lower is better. Charts are regenerated on each docs build.

<iframe class="perf-chart-frame perf-chart-frame--home" src="generated/performance/summary.html" title="Representative benchmarks"></iframe>

### Compilation

**`gr.compile`** shines in **Zarr streaming** pipelines: it fuses batch transforms into one Rust plan and enables Rayon batch scheduling (`scheduler="auto"`). The chart below times a full mini-epoch over a protein-like dataset (**256 structures × 96 residues**, `batch_size=32`, `cpu=4`) — Python vs compiled, single- vs multi-core. The suite completes in under one minute.

<iframe class="perf-chart-frame perf-chart-frame--home" src="generated/performance/compile_summary.html" title="Compiler benchmarks"></iframe>

Full suites: [`benchmark_ragged_api.py`](https://github.com/Imaginary-Biolabs/Grumpy/blob/main/benchmarks/benchmark_ragged_api.py), [`benchmark_compile_suite.py`](https://github.com/Imaginary-Biolabs/Grumpy/blob/main/benchmarks/benchmark_compile_suite.py) — see [benchmarks/README.md](https://github.com/Imaginary-Biolabs/Grumpy/blob/main/benchmarks/README.md) for setup.
