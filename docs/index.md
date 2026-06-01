# Home

Grumpy is layout-first infrastructure for **ragged and nested** numerical data: protein tables, variable-length sequences, mixed annotation fields, and the batches that feed structural ML pipelines. Unlike ad-hoc Python lists, Grumpy keeps a typed layout tree in Rust so elementwise ops, reductions, neighbor search, and streaming I/O stay fast without hand-written loops.

This guide walks from installation through arrays, dataframes, Zarr streaming, and optional compile-time fusion. Each page builds on the previous one.

## Installation

Grumpy requires **Python ≥ 3.10** and a Rust toolchain when building from source.

### PyPI

```bash
pip install grumpy
```

If a wheel is not yet available for your platform, build from source as below.

### From source

```bash
git clone https://github.com/Imaginary-Biolabs/Grumpy.git
cd Grumpy
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip maturin
maturin develop --release
pip install -e ".[dev]"     # optional: pytest, mkdocs
```

Verify the install:

```python
import grumpy as gr

print(gr.__version__)
```

## A tour of the main features

The snippets below are self-contained. Later pages explain each topic in depth.

### Ragged arrays

Grumpy arrays mirror nested Python lists but carry a **homogeneous dtype** on every leaf and run kernels in Rust:

```python
import grumpy as gr

x = gr.array([[1, 2, 3], [4, 5]], dtype=gr.int32)
print(x.to_list())           # [[1, 2, 3], [4, 5]]
print(x.mean(dim=1).to_list())  # [2.0, 4.5]
```

### Union layouts (mixed scalar and list rows)

When one axis mixes singletons and lists — common for GO terms, isoform IDs, or mixture SMILES — use a **union** layout from the same constructor:

```python
u = gr.array([1, [2, 3], 4], dtype=gr.int64)
print((u * 2).to_list())     # [2, [4, 6], 8]
```

### Dataframes with schema

Named columns share outer list structure; an optional **schema** names nesting levels for dot notation:

```python
df = gr.dataframe(
    {"id": ["a", "b"], "coords": [[1.0, 2.0], [3.0, 4.0, 5.0]]},
    schema=["molecule", "atom"],
)
print(df.molecule.coords.to_list())
```

### Save, stream, and transform batches

Persist to a Zarr directory, then iterate batches for training — with optional parallel `apply`:

```python
gr.save(df, "data.gr", chunk_size=64)

for batch in gr.stream("data.gr", batch_size=32, workers=2):
    batch = batch * 2.0
    train_step(batch)
```

### Compile fused transforms

When a batch function is simple enough, `@gr.compile` fuses it into one Rust plan (see [Compilation](compilation.md)):

```python
@gr.compile
def scale(batch):
    return batch * 2.0 + 1.0

st = gr.stream("data.gr", batch_size=32)
for batch in st.apply(scale, compile="auto", scheduler="auto"):
    train_step(batch)
```

## Performance

Representative **public API** timings on slightly ragged data (Grumpy, Awkward) vs rectangular NumPy with the same leaf count. Bar groups are **Grumpy · NumPy · Awkward**; lower is better. Charts are regenerated on each docs build.

<iframe class="perf-chart-frame perf-chart-frame--home" src="generated/performance/summary.html" title="Representative benchmarks"></iframe>

Full benchmark suites live in [`benchmarks/`](https://github.com/Imaginary-Biolabs/Grumpy/tree/main/benchmarks) — see [`benchmarks/README.md`](https://github.com/Imaginary-Biolabs/Grumpy/blob/main/benchmarks/README.md) for setup.

---

**Next:** [Arrays](arrays.md) — construct ragged arrays, run elementwise ops, and index into nested data.
