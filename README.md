<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/grumpy_logo_horizontal_dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="docs/assets/grumpy_logo_horizontal_light.svg">
    <img src="docs/assets/grumpy_logo_horizontal_light.svg" alt="Grumpy" width="280">
  </picture>
</p>

<p align="center">
  <strong>High-performance numerical computing on ragged and nested data</strong>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-BSL--1.1-2A2725?style=for-the-badge&logo=opensourceinitiative&logoColor=E3E1DE" alt="license BSL 1.1" /></a>
  <a href="https://github.com/Imaginary-Biolabs/Grumpy/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/Imaginary-Biolabs/Grumpy/ci.yml?branch=main&style=for-the-badge&label=build&color=484240&logo=githubactions&logoColor=E3E1DE" alt="build status" /></a>
  <a href="https://codecov.io/gh/Imaginary-Biolabs/Grumpy"><img src="https://img.shields.io/codecov/c/github/Imaginary-Biolabs/Grumpy/main?style=for-the-badge&color=777067&logo=codecov&logoColor=E3E1DE" alt="codecov coverage" /></a>
  <a href="https://github.com/Imaginary-Biolabs/Grumpy/releases"><img src="https://img.shields.io/badge/version-0.1.1-C8C4BF?style=for-the-badge&logo=python&logoColor=2A2725" alt="version 0.1.1" /></a>
</p>

<p align="center">
  <a href="LICENSE-FAQ.md">License FAQ</a> ·
  <a href="https://imaginary-biolabs.github.io/Grumpy/">Documentation</a> ·
  <a href="benchmarks/README.md">Benchmarks</a> ·
  <a href="CONTRIBUTING.md">Contributing</a>
</p>

**Grumpy** is a Python library (Rust core) for **ragged**, **nested**, and **nullable** arrays — layout-first infrastructure for structural ML and general scientific computing. Mutable typed leaves, Zarr I/O, streaming batches, and optional `@gr.compile` fusion.

```bash
pip install grumpy
```

```python
import grumpy as gr

x = gr.array([[1, 2, 3], [4, 5]], dtype=gr.int32)
print(x.mean(dim=1).to_list())  # [2.0, 4.5]
```

## License

Business Source License 1.1 — see [LICENSE](LICENSE) and [License FAQ](LICENSE-FAQ.md). Copyright © Imaginary Biolabs GmbH.
