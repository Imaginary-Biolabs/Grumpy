# Benchmarks

Kernel-oriented microbenchmarks comparing **Grumpy**, **NumPy**, and (where installed) **Awkward Array**.

Construction from Python lists or `ak.from_numpy` is timed **separately** from the timed kernel region so comparisons focus on compute, not input materialization.

## Prerequisites

```bash
maturin develop --release
pip install -e ".[benchmark]"   # adds awkward
```

## Run all benchmarks

```bash
make bench-all
```

Or individual scripts:

```bash
python benchmarks/benchmark_elementwise.py --nrows 4096 --ncols 256
python benchmarks/benchmark_awkward_elementwise.py --nrows 2048 --ncols 128
python benchmarks/benchmark_indexing.py --nrows 4096 --ncols 64 --nfancy 4096
python benchmarks/benchmark_reductions.py
python benchmarks/benchmark_streaming.py
python benchmarks/benchmark_neighbors.py
python benchmarks/benchmark_compile.py
python benchmarks/benchmark_protein_pipeline.py
```

Quick smoke:

```bash
make bench
```

## Interpreting results

- **numpy_kernel** — rectangular typed NumPy baseline.
- **grumpy_kernel** — Rust layout kernel (often `_mul2d_*` helpers) on ragged list-chain data.
- **awkward_kernel** — requires `pip install awkward`; uses `ak` arrays built once outside the timed region.

Lower kernel time is better. Ratios above 1.0 mean Grumpy/Awkward is slower than NumPy for that microbenchmark.
