<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/grumpy_logo_horizontal_dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="docs/assets/grumpy_logo_horizontal_light.svg">
    <img src="docs/assets/grumpy_logo_horizontal_light.svg" alt="Grumpy" width="280">
  </picture>
</p>

<p align="center">
  <strong>High-performance numerical computing on ragged and nested data</strong><br/>
  Rust core · Python bindings · Zarr I/O · optional compile-time fusion
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-BSL--1.1-2A2725?style=for-the-badge&logo=opensourceinitiative&logoColor=E3E1DE" alt="license BSL 1.1" /></a>
  <a href="https://github.com/Imaginary-Biolabs/Grumpy/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/Imaginary-Biolabs/Grumpy/ci.yml?branch=main&style=for-the-badge&label=build&color=484240&logo=githubactions&logoColor=E3E1DE" alt="build status" /></a>
  <a href="https://codecov.io/gh/Imaginary-Biolabs/Grumpy"><img src="https://img.shields.io/codecov/c/github/Imaginary-Biolabs/Grumpy/main?style=for-the-badge&color=777067&logo=codecov&logoColor=E3E1DE" alt="codecov coverage" /></a>
  <a href="https://github.com/Imaginary-Biolabs/Grumpy/releases"><img src="https://img.shields.io/badge/version-0.1.0-C8C4BF?style=for-the-badge&logo=python&logoColor=2A2725" alt="version 0.1.0" /></a>
</p>

<p align="center">
  <a href="LICENSE-FAQ.md">License FAQ</a> ·
  <a href="https://imaginary-biolabs.github.io/Grumpy/">Documentation</a> ·
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
- [Dataframes](docs/dataframes.md)
- [Saving and loading](docs/saving-loading.md)
- [Compilation](docs/compilation.md)
- [API Reference](docs/api.md)

Build the site locally: `pip install -e ".[dev]" && mkdocs serve`.

Published docs: [imaginary-biolabs.github.io/Grumpy](https://imaginary-biolabs.github.io/Grumpy/)

## Benchmarks

See [benchmarks/README.md](benchmarks/README.md). Quick run:

```bash
make bench
```

Grumpy targets NumPy-class kernel performance on hot paths; Awkward comparisons help validate ragged-layout competitiveness (construction overhead reported separately).

## Development

```bash
make develop
make coverage   # ≥95% on python/grumpy/
make bench-all
```

Rust code lives in `src/`; Python bindings in `python/grumpy/`. See [CONTRIBUTING.md](CONTRIBUTING.md) and [AGENTS.md](AGENTS.md).

## License

Business Source License 1.1 — see [LICENSE](LICENSE) and [License FAQ](LICENSE-FAQ.md). Copyright Imaginary Biolabs GmbH. For commercial or partnership licensing, <a href="mailto:licensing&#64;imaginary&#46;bio?subject=Grumpy%20licensing%20inquiry">contact licensing</a>.
