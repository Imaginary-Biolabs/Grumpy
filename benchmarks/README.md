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

Use this file to drive docs homepage charts or CI trend tracking.

## Docs homepage chart

The representative chart on the [docs landing page](https://imaginary-biolabs.github.io/Grumpy/) is rebuilt on every `mkdocs build` via `docs/hooks.py`:

```bash
maturin develop --release
pip install -e ".[dev]" plotly awkward
mkdocs build -f mkdocs.yml
```

Or regenerate the chart only:

```bash
python benchmarks/generate_perf_charts.py
```

## Compiler benchmark (docs / charts)

Compares **streaming** pipelines where ``gr.compile`` is meant to be used — full mini-epoch over a Zarr-backed protein dataset:

```bash
python benchmarks/benchmark_compile_suite.py
python benchmarks/benchmark_compile_suite.py --quick          # <60 s elementwise smoke
python benchmarks/benchmark_compile_suite.py --json docs/generated/performance/compile_suite.json
```

Default dataset: **256 proteins × 256 residues × 3 coords** (CA trace). Streaming with `batch_size=32`, `cpu=4`. Full suite budget ~300 s; per-mode timeout 90 s. Use `--quick` for 96-residue elementwise-only runs under 60 s.

Pipelines: fused elementwise + pool, staged elementwise (4 functions → one plan), normalize + kNN + pool (`k=16`), kNN (`k=16`) + pool.

Each case times five modes: Python stream (cpu=1), Python parallel (cpu=4), compiled (cpu=1), compiled + ThreadPool (cpu=4), compiled + Rust scheduler (cpu=4). **Compile pays off primarily via the Rust batch scheduler (cpu=4)**; cpu=1 compiled and eager paths both dispatch to Rust kernels.

## In-memory vs streaming

Compares batched transforms over **in-memory** data vs **Zarr streaming** (plus load-only baselines):

```bash
python benchmarks/benchmark_memory_vs_stream.py
python benchmarks/benchmark_memory_vs_stream.py --json docs/generated/performance/memory_vs_stream.json
```

Default: 256 proteins × 96 residues, `batch_size=32`, **< 60 s** wall budget. Shows how much of a stream epoch is Zarr I/O vs compute.

The compile chart on the [docs landing page](https://imaginary-biolabs.github.io/Grumpy/) is rebuilt together with the API chart by `generate_perf_charts.py`.
