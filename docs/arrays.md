# Arrays

Grumpy arrays represent **ragged nested lists** using a small layout tree (`ListOffset` → … → `Leaf`).

## Construction

```python
import grumpy as gr

x = gr.array([[1, 2, 3], [4, 5]], dtype=gr.int32)
y = gr.array([[1, 2], [[None, 5], [6]]])  # dtype inferred; None → null
```

## Indexing and mutation

Coordinate vs array indexing matches the API examples in the project design doc:

```python
x[0]          # [1, 2, 3]
x[[0, 1]]     # [[1, 2, 3], [4, 5]]
x[[0, 1], 0]  # [1, 4]
x[0] = 100
```

## Operations

Elementwise ops broadcast like NumPy on compatible ragged shapes:

```python
(x * 2 + 1).to_list()
x.mean(dim=1).to_list()
```

## Known limitations

- **`UnionScalarList`** (mixed scalar/list depth) is supported for elementwise/unary/compare ops on matching structure, scalar multiply, sum/mean, unique, shuffle, streaming slice loads, and axis-0 concat/append.
- Prefer pure **list-chains** for reductions beyond sum, neighbors, einsum, and full schema validation.
