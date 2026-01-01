## Grumpy

Grumpy is a Python library (Rust core + Python bindings) for high-performance numerical computing on ragged/nested data.

### Quickstart

Build + install locally (Rust extension):

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -U pip
python -m pip install maturin
maturin develop --release
python -m pip install -e '.[dev]'
pytest
```

Basic arrays:

```python
import grumpy as gr

x = gr.array([[1, 2, 3], [4, 5]], dtype=gr.int32)
y = gr.array([[1, 2, 3], [[None, 5], [6]]])  # dtype inferred, supports None/nulls

print(x.to_list())         # [[1, 2, 3], [4, 5]]
print(x.to_numpy())        # numpy array if rectangular + all-valid, else object array
print(x.shape(dim=1))      # [3, 2]
```

Strings:

```python
s = gr.array([["one", None], ["two", "three"]], dtype=gr.string)
print(s.to_list())  # [['one', None], ['two', 'three']]
```

Elementwise ops + broadcasting:

```python
x = gr.array([[1, 2, 3], [4, 5]])
print((x * 2).to_list())         # [[2, 4, 6], [8, 10]]
print((x + x).to_list())         # [[2, 4, 6], [8, 10]]
```

Reductions (examples):

```python
x = gr.array([[1, 2, 3], [4, 5]])
print(x.sum(dim=1).to_list())    # [6, 9]
print(x.mean(dim=1).to_list())   # [2.0, 4.5]
```

DataFrames + schema + dot-notation:

```python
df = gr.dataframe(
    {
        "molecule_id": ["one", "two"],
        "residue_name": [["A", "B", "C"], ["D", "E"]],
        "atom_number": [[[1, 2], [3, 4, 5], [6]], [[7, 8], [9]]],
    },
    schema=["molecule", ("residue", "group"), "atom"],
)

print(df.atom_number.to_list())          # [1,2,3,4,5,6,7,8,9]
print(df.residue.atom_number.to_list())  # [[1,2],[3,4,5],[6],[7,8],[9]]

df.residue.residue_weight = [0.5, 0.7, 0.8, 0.9, 1.0]
print(df["residue_weight"].to_dict()["residue_weight"])  # [[0.5,0.7,0.8],[0.9,1.0]]
```

Saving + loading (Zarr-backed):

```python
gr.save(df, "mydata.gr", chunk_size=1024)
df2 = gr.load("mydata.gr")
print(df2.to_dict() == df.to_dict())  # True
```

Streaming + parallel apply (CPU threads):

```python
st = gr.stream("mydata.gr", batch_size=32, drop_last=False)

def transform(batch):
    # do work on a batch (best if it releases the GIL, e.g. I/O or Rust-backed kernels)
    return batch

st2 = st.apply(transform, cpu=8)  # preserves order
for batch in st2:
    ...
```

### Local dev

Install (builds the Rust extension):

```bash
python -m pip install -U pip
python -m pip install maturin
maturin develop
python -m pip install -e '.[dev]'
pytest
```

