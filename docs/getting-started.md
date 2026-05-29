# Getting started

## Install from source

Grumpy is distributed as a maturin-built extension module.

```bash
git clone https://github.com/imaginary-bio/grumpy.git
cd grumpy
python -m venv .venv
source .venv/bin/activate
pip install -U pip maturin
maturin develop --release
pip install -e ".[dev]"
pytest
```

Requirements: Python ≥ 3.10, Rust stable toolchain.

## First array

```python
import grumpy as gr

x = gr.array([[1, 2, 3], [4, 5]], dtype=gr.int32)
print(x.to_list())
print(x.shape(dim=1))
```

## Version

```python
import grumpy as gr
print(gr.__version__)
```
