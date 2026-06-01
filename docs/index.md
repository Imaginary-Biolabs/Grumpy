## Highlights

- Layout-first kernels for elementwise ops, reductions, neighbors, and more — **list-chains** and **`UnionScalarList`** are both first-class
- `GrumpyDataFrame` with optional schema and dot-notation column access (list-chain and union columns)
- Zarr save/load and axis-0 streaming with optional `gr.compile` fusion
- BSL 1.1 license (Imaginary Biolabs GmbH) — [License FAQ](license-faq.md)

## Quick links

- [Getting started](getting-started.md)
- [Arrays](arrays.md)
- [Dataframes](dataframes.md)
- [Saving and loading](saving-loading.md)
- [Compilation](compilation.md)
- [GPU Acceleration](gpu-acceleration.md)
- [API Reference](api.md)

## Performance

Representative **public API** timings on slightly ragged data (Grumpy, Awkward) vs rectangular NumPy with the same leaf count. Bar groups are **Grumpy · NumPy · Awkward**; lower is better. Charts are regenerated on each docs build.

<iframe class="perf-chart-frame perf-chart-frame--home" src="generated/performance/summary.html" title="Representative benchmarks"></iframe>

Full suite: [`benchmarks/benchmark_ragged_api.py`](https://github.com/Imaginary-Biolabs/Grumpy/blob/main/benchmarks/benchmark_ragged_api.py) — see [benchmarks/README.md](https://github.com/Imaginary-Biolabs/Grumpy/blob/main/benchmarks/README.md) for setup and other benchmarks.
