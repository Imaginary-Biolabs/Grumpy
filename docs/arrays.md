# Arrays

Grumpy arrays represent **ragged nested lists** using a small layout tree. See [Dtypes and casting](dtypes.md) for dtype rules, promotion, and ``astype`` modes.

| Path | When it appears | Example Python input |
|------|-----------------|----------------------|
| **List-chain** | Every row has the same nesting depth | `[[1, 2, 3], [4, 5]]` |
| **`UnionScalarList`** | One axis mixes scalars and lists | `[1, [2, 3], 4]` |

Both are constructed with `gr.array`, saved to Zarr, streamed, and used in dataframes. Kernels implement **both paths**; choose the layout that matches your data rather than normalizing for compatibility.

## Construction

### List-chain (fixed depth per axis)

```python
import grumpy as gr

x = gr.array([[1, 2, 3], [4, 5]], dtype=gr.int32)
y = gr.array([[1, 2], [[None, 5], [6]]])  # dtype inferred; None → null
```

### Union (mixed scalar/list at one axis)

Use when annotation columns or exports mix singletons and lists at the same logical level (GO terms, variant consequences, mixture SMILES, etc.):

```python
# One GO term vs many on the same axis
go = gr.array(["GO:0003674", ["GO:0003674", "GO:0005524"], []], dtype=gr.string)

# Numeric union: scalar row vs list row
z = gr.array([1, [2, 3], 4], dtype=gr.int64)
z.mean().to_list()          # 2.5
z.min(dim=0).to_list()      # [1, 2, 4]
```

Unions can also appear **inside** list-chains (e.g. a molecule column that is union-shaped while residue/atom columns remain pure list-chains).

## Indexing and mutation

Coordinate vs array indexing matches the API examples in the project design doc. Both layout paths support axis-0 fancy selection; unions compact scalar/list pools on gather.

```python
x = gr.array([[1, 2, 3], [4, 5]], dtype=gr.int32)
x[0]          # [1, 2, 3]
x[[0, 1]]     # [[1, 2, 3], [4, 5]]
x[[0, 1], 0]  # [1, 4]
x[0] = 100

u = gr.array([1, [2, 3], 4], dtype=gr.int64)
u[[0, 2]].to_list()       # [1, 4]
u[[1, 1], 0].to_list()    # [2, 2]  (coordinate fancy on union)
```

## Operations

Elementwise ops broadcast like NumPy on compatible ragged shapes, including **union ↔ list-chain** and **union ↔ union** pairs:

```python
# List-chain
(x * 2 + 1).to_list()
x.mean(dim=1).to_list()

# Union
u = gr.array([1, [2, 3], 4], dtype=gr.int64)
(u * 2).to_list()                    # [2, [4, 6], 8]
(u + gr.array([10, [20, 30], 40])).to_list()

# Broadcast union with list-chain (same outer length)
lc = gr.array([[1], [2, 3], [4]], dtype=gr.int64)
(u + lc).to_list()
```

### Layout support matrix

The table below lists current kernel coverage. Gaps apply to **both** paths where noted; neither layout is deprecated.

| Category | List-chain | `UnionScalarList` |
|----------|------------|-------------------|
| Elementwise / unary / compare | yes | yes |
| Broadcast (mixed layouts) | yes | yes |
| `sum`, `mean`, `min`, `max`, `ptp` | yes | yes (`dim=0` / all-axis) |
| `var`, `std` | yes | yes |
| Sort / argsort / argmin / argmax (`dim=-1`) | yes | yes |
| `searchsorted`, `unique`, shuffle | yes | yes |
| Axis-0 concat / append / fancy index | yes | yes |
| Streaming slice / `batch_on` | yes | yes (compact partial I/O) |
| `einsum` / `tensordot` (1D contraction patterns) | yes | yes |
| `neighbors` | yes (`dim=0`, `dim=1`) | yes (`dim=0`; rect2d subtrees) |
| `histogram`, `where`, strict 1D leaf helpers | yes | not yet |
| `partition` on flat 1D leaf | yes | not yet |
| Fast leaf-only assignment fast paths | yes | use general mutation path |

When adding new ops, implement list-chain **and** union traversal (or document a shared wrapper walk) so both paths stay in parity.
