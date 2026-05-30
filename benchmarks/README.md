# Benchmarks

Comparisons of **Grumpy**, **NumPy**, and (where installed) **Awkward Array**.

## Prerequisites

```bash
maturin develop --release
pip install -e ".[benchmark]"   # adds awkward
```

## Primary benchmark (docs / charts)

**Use this for user-facing performance claims.**

```bash
python benchmarks/benchmark_ragged_api.py --nrows 4096 --ncols 256
python benchmarks/benchmark_ragged_api.py --json benchmarks/results/ragged_api.json
```

Times **public API** calls only: `(a * b).sum(...)`, `gr.array(...)`, `gr.isin`, `x[i, j]`, etc. Intermediate arrays are included in the timed region so numbers reflect what users feel in interactive code.

- NumPy: rectangular `(nrows, ncols)` with the same total leaf count
- Grumpy / Awkward: slightly ragged rows (`ncols±1`), same flat value order
- Construction from nested lists is timed separately

## Secondary benchmark (engineers)

**Not for docs.** Internal fused Rust kernels (`_mul2d_*`) and preallocated NumPy `out=` buffers.

```bash
python benchmarks/benchmark_ragged_kernels.py --nrows 4096 --ncols 256
```

## Run both ragged suites

```bash
python benchmarks/benchmark_ragged_suite.py --nrows 4096 --ncols 256
```

## Run all benchmarks

```bash
make bench-all
```

Quick smoke (includes legacy elementwise + full ragged suite):

```bash
make bench
```

Other scripts:

```bash
python benchmarks/benchmark_elementwise.py --nrows 4096 --ncols 256
python benchmarks/benchmark_indexing.py --nrows 4096 --ncols 64 --nfancy 4096
python benchmarks/benchmark_reductions.py
# ...
```

## Interpreting results

| Suite | Audience | What it measures |
|---|---|---|
| `benchmark_ragged_api.py` | Users, docs, marketing | Idiomatic library calls end-to-end |
| `benchmark_ragged_kernels.py` | Engineers | Kernel throughput without temporaries |
| `benchmark_elementwise.py` | Engineers | Rectangular vs ragged; kernel vs via-op split |

Lower time is better. Ratio columns show Grumpy or Awkward time divided by NumPy (or each other).

## JSON output for charts

The API benchmark accepts `--json PATH` and writes a structured report:

```json
{
  "suite": "ragged_public_api",
  "cases": [
    {"name": "(a * b).sum()", "category": "Elementwise", "numpy_ms": 0.3, "grumpy_ms": 1.2, ...}
  ]
}
```

Use this file to drive docs site charts or CI trend tracking.
