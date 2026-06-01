# Arrays

Arrays are the core of Grumpy. Biomolecular and scientific workloads rarely fit a single rectangular matrix: residue counts vary per protein, atoms vary per residue, and annotation fields may be a single ID on one row and a list on the next. Grumpy represents this as **ragged nested lists** with a fixed **dtype** on every leaf, stored in a compact layout tree that Rust kernels traverse without Python per-element overhead.

This page covers construction, elementwise math, and indexing. Dataframes (named columns with optional schema) build on the same layout machinery.

## Construction

Pass nested Python lists or tuples to `gr.array`. When you omit `dtype`, Grumpy infers it from non-null leaves (`int` → `int64`, `float` → `float64`).

### List-chain (fixed nesting depth)

Use a **list-chain** when every row at a given level has the same nesting depth — the usual case for coordinates, atom tables, and fixed-depth tensors:

```python
import grumpy as gr

x = gr.array([[1, 2, 3], [4, 5]], dtype=gr.int32)
print(x.to_list())        # [[1, 2, 3], [4, 5]]
print(x.shape(dim=1))     # outer length along axis 1
```

`None` in nested input becomes a **null** leaf; validity is tracked separately from the numeric buffer:

```python
y = gr.array([[1, None, 3], [4, 5]], dtype=gr.int32)
print(y.to_list())        # [[1, None, 3], [4, 5]]
```

### Union (mixed scalar and list at one axis)

Use a **union** when one axis mixes scalars and lists — for example one GO term vs many on the same column:

```python
go = gr.array(["GO:0003674", ["GO:0003674", "GO:0005524"], []], dtype=gr.string)
nums = gr.array([1, [2, 3], 4], dtype=gr.int64)
print(nums.mean().to_list())   # 2.5 — reduction over all leaves
```

Both list-chains and unions are constructed with the same `gr.array` call; Grumpy picks the layout from the Python structure.

## Elementwise operations

Grumpy exposes NumPy-like elementwise ops. They **broadcast** across compatible ragged shapes, including mixed list-chain ↔ union pairs when outer lengths align.

Start with unary and binary ops on a list-chain:

```python
x = gr.array([[1, 2, 3], [4, 5]], dtype=gr.int32)

print((x * 2).to_list())       # [[2, 4, 6], [8, 10]]
print((x + 1).to_list())       # [[2, 3, 4], [5, 6]]
print(x.mean(dim=1).to_list()) # [2.0, 4.5] — reduce along inner axis
```

Free functions mirror methods where useful:

```python
a = gr.array([[1, 2]], dtype=gr.int32)
b = gr.array([[10, 20]], dtype=gr.int32)
print(gr.add(a, b).to_list())  # [[11, 22]]
```

On unions, elementwise ops preserve the scalar-vs-list structure:

```python
u = gr.array([1, [2, 3], 4], dtype=gr.int64)
print((u * 2).to_list())       # [2, [4, 6], 8]
```

For dtype rules and casting, see [Developer — dtypes](developer.md#dtypes-and-casting).

## Indexing

Indexing selects sub-trees without copying entire datasets. Grumpy supports **array indexing** (rows, slices, fancy indices) and **coordinate indexing** (row + column within ragged rows).

### Getting values

Select outer rows with integers or slices:

```python
x = gr.array([[1, 2, 3], [4, 5]], dtype=gr.int32)

print(x[0].to_list())      # [1, 2, 3]
print(x[[0, 1]].to_list()) # [[1, 2, 3], [4, 5]]
```

Coordinate indexing picks one element per row:

```python
print(x[[0, 1], 0].to_list())  # [1, 4] — first element of each row
```

Unions support the same patterns; fancy indices address union rows consistently:

```python
u = gr.array([1, [2, 3], 4], dtype=gr.int64)
print(u[[0, 2]].to_list())     # [1, 4]
print(u[[1, 1], 0].to_list())  # [2, 2]
```

### Setting values

Assignment mutates leaves in place (Grumpy arrays are **mutable**, unlike many immutable columnar libraries):

```python
x = gr.array([[1, 2, 3], [4, 5]], dtype=gr.int32)
x[0] = 100
print(x.to_list())   # [[100, 2, 3], [4, 5]]

x[1, 0] = 99
print(x.to_list())   # [[100, 2, 3], [99, 5]]
```

Out-of-range indices raise actionable errors with `cause:` and `fix:` hints — see [Developer — error handling](developer.md#error-handling).

---

**Next:** [Dataframes](dataframes.md) — group columns, use dot notation, and enforce schema across nesting levels.
