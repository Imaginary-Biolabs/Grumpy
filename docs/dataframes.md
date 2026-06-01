# Dataframes

A dataframe groups named **GrumpyArray** columns that share outer list structure. In structural biology this maps naturally to tables such as molecule → residue → atom: each column holds the same nesting depth at every schema level, and dot notation lets you write `df.residue.coords` instead of manual axis dropping.

This page walks through construction, dot notation, column operations, and schema rules.

## Construction

Pass a mapping of column names to array-like values. Columns may be flat lists or nested lists; Grumpy converts each column to a layout-backed array.

```python
import grumpy as gr

df = gr.dataframe({
    "molecule_id": ["prot_a", "prot_b"],
    "residue_name": [["ALA", "GLY", "SER"], ["THR", "VAL"]],
    "residue_x": [[0.1, 0.2, 0.3], [1.0, 1.1]],
})
print(df.residue_name.to_list())
```

Union columns (mixed scalar/list per row) are allowed when annotations vary row by row:

```python
df = gr.dataframe({
    "molecule_id": ["m1", "m2", "m3"],
    "go_term": [
        "GO:0003674",
        ["GO:0003674", "GO:0005524"],
        [],
    ],
    "residue_name": [["A", "B"], ["C"], ["D", "E"]],
})
assert len(df) == 3
```

## Dot notation

Attribute access **peels** nesting axes. `df.residue` drops the outer molecule axis shared by all columns at that level; `df.residue.residue_name` reaches the residue-named field across molecules.

```python
df = gr.dataframe(
    {
        "molecule_id": ["one", "two"],
        "residue_name": [["A", "B", "C"], ["D", "E"]],
        "atom_number": [[[1, 2], [3, 4, 5], [6]], [[7, 8], [9]]],
    },
    schema=["molecule", ("residue", "group"), "atom"],
)

print(df.atom_number.to_list())
print(df.residue.atom_number.to_list())
```

On union columns, one level of inner structure is stacked when you peel — useful for heterogeneous annotation fields while keeping molecule-level alignment.

## Operations

Dataframe columns are ordinary Grumpy arrays. Use elementwise ops, reductions, and neighbor search on any column:

```python
df = gr.dataframe({
    "id": ["a", "b"],
    "vals": [[1.0, 2.0], [3.0, 4.0, 5.0]],
})

scaled = df.vals * 2.0
print(scaled.to_list())

means = df.vals.mean(dim=1)
print(means.to_list())
```

Assign new columns with matching outer shape (use schema-level paths when a schema is set):

```python
df = gr.dataframe(
    {
        "id": ["a", "b"],
        "vals": [[1.0, 2.0], [3.0, 4.0, 5.0]],
    },
    schema=["molecule", "residue"],
)
df.residue.residue_weight = [[0.5, 0.7], [0.9, 1.0, 1.1]]
print(df.residue.residue_weight.to_list())
```

For kNN graph edges on coordinate columns, pass a column array to `gr.neighbors` (documented in the [API Reference](api.md)):

```python
coords = gr.array([[[0, 0, 0], [1, 1, 1]], [[2, 2, 2]]], dtype=gr.float64)
edges = gr.neighbors(coords, coords, k=2, dim=1, loop=False)
```

Compiled pipelines can fuse dataframe dot-assignments such as `batch.residue.center = batch.residue.coords.mean(dim=1)` — see [Compilation](compilation.md).

## Schema

An optional **schema** lists nesting level names. It enforces that columns sharing a prefix have compatible outer list offsets (for list-chain columns) and consistent **outer length** (including union columns).

```python
df = gr.dataframe(
    {
        "molecule_id": ["one", "two"],
        "residue_name": [["A", "B", "C"], ["D", "E"]],
        "atom_number": [[[1, 2], [3, 4, 5], [6]], [[7, 8], [9]]],
    },
    schema=["molecule", ("residue", "group"), "atom"],
)
```

Rules in brief:

- Column names must start with a valid schema prefix (`molecule_id` → level `molecule`).
- List-chain columns at the same level share stored outer offsets.
- Union columns skip inner length checks at the union axis but must match **outer length** with sibling columns.
- New columns must respect schema depth; union columns count as one outer element per row at their level.

Schema violations raise `grumpy.SchemaViolation` with a concrete `fix:` — see [Developer — error handling](developer.md#error-handling).

---

**Next:** [Saving and loading](saving-loading.md) — persist dataframes to Zarr and stream batches through transforms.
