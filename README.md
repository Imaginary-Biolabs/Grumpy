<p align="center">
  <img src="docs/assets/grumpy_icon.svg" alt="Grumpy" width="72" />
</p>

<h1 align="center">Grumpy</h1>

<p align="center">
  <strong>High-performance numerical computing on ragged and nested data</strong><br/>
  Rust core · Python bindings · Zarr I/O · optional compile-time fusion
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-BSL--1.1-2A2725?style=for-the-badge&logo=opensourceinitiative&logoColor=E3E1DE" alt="license BSL 1.1" /></a>
  <a href="https://github.com/Imaginary-Biolabs/Grumpy/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/Imaginary-Biolabs/Grumpy/CI?branch=main&style=for-the-badge&label=build&color=484240&logo=githubactions&logoColor=E3E1DE" alt="build status" /></a>
  <img src="https://img.shields.io/badge/coverage-100%25-777067?style=for-the-badge&logo=pytest&logoColor=E3E1DE" alt="coverage 100%" />
  <a href="https://github.com/Imaginary-Biolabs/Grumpy/releases"><img src="https://img.shields.io/badge/version-0.1.0-C8C4BF?style=for-the-badge&logo=python&logoColor=2A2725" alt="version 0.1.0" /></a>
</p>

<p align="center">
  <a href="LICENSE-FAQ.md">License FAQ</a> ·
  <a href="docs/">Documentation</a> ·
  <a href="benchmarks/README.md">Benchmarks</a> ·
  <a href="CONTRIBUTING.md">Contributing</a>
</p>

**Grumpy** is developed by [Imaginary Biolabs](https://www.imaginary.bio) as layout-first infrastructure for biomolecular machine learning — and as a general-purpose library for **ragged**, **nested**, and **nullable** scientific arrays.

It shares Awkward Array’s buffer-tree mental model, with deliberate differences: **mutable** arrays, **strong dtypes**, homogeneous leaves, explicit **validity bitmaps**, integrated **Zarr** storage, and **streaming** transforms that can fuse into Rust execution plans.

## Features

- **Ragged arrays** — arbitrary nesting via `ListOffset` layouts; NumPy-like ops with broadcasting
- **DataFrames** — named columns, optional multi-level **schema**, dot-notation access
- **I/O** — save/load Zarr stores; axis-0 **streaming** with parallel `apply`
- **Compilation** — `@gr.compile` and `Stream.apply(compile="auto")` fuse supported transforms in Rust
- **Neighbors** — kNN / radius graph edges for 0D and grouped 1D point clouds

## Install (from source)

```bash
git clone https://github.com/imaginary-bio/grumpy.git
cd grumpy
python -m venv .venv && source .venv/bin/activate
pip install -U pip maturin
maturin develop --release
pip install -e ".[dev]"
pytest
```

Published wheels are not yet on PyPI; build with [maturin](https://www.maturin.rs/) as above.

## Quickstart

```python
import grumpy as gr

print(gr.__version__)

x = gr.array([[1, 2, 3], [4, 5]], dtype=gr.int32)
print(x.to_list())
print(x.mean(dim=1).to_list())

df = gr.dataframe(
    {"id": ["a", "b"], "vals": [[1, 2], [3, 4, 5]]},
)
gr.save(df, "data.gr")
for batch in gr.stream("data.gr", batch_size=32):
    batch = batch * 2  # or @gr.compile transform
```

## Documentation

- [Getting started](docs/getting-started.md)
- [Arrays](docs/arrays.md)
- [DataFrames & schema](docs/dataframes.md)
- [I/O & streaming](docs/io-streaming.md)
- [Compilation](docs/compilation.md)

Build the site locally: `pip install -e ".[dev]" && mkdocs serve`.

## Benchmarks

See [benchmarks/README.md](benchmarks/README.md). Quick run:

```bash
make bench
```

Grumpy targets NumPy-class kernel performance on hot paths; Awkward comparisons help validate ragged-layout competitiveness (construction overhead reported separately).

## Development

```bash
make develop
make coverage   # 100% on python/grumpy/
make bench-all
```

Rust code lives in `src/`; Python bindings in `python/grumpy/`. See [CONTRIBUTING.md](CONTRIBUTING.md) and [AGENTS.md](AGENTS.md).

## License

Business Source License 1.1 — see [LICENSE](LICENSE) and [License FAQ](LICENSE-FAQ.md). Copyright Imaginary Biolabs GmbH. For commercial or partnership licensing, <a href="mailto:licensing&#64;imaginary&#46;bio?subject=Grumpy%20licensing%20inquiry">contact licensing</a>.
