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

## Schema indexing

When a dataframe is constructed with `schema=`, indexing **subsets one schema level at a time** and preserves the original nested column structure. Drill down by chaining accessors or repeated `[]` on the dataframe.

### Single-level subset (`df[...]`)

At each step, pass one index at the **current** schema level: int, slice, fancy list, or boolean mask.

```python
df = gr.dataframe(
    {
        "scene_id": ["S0", "S1"],
        "molecule_id": [["M0", "M1"], ["M2"]],
        "residue_name": [[["A", "B"], ["C"]], [["D", "E"]]],
        "atom_number": [[[[1, 2], [3]], [[4, 5, 6]]], [[[7, 8], [9]]]],
    },
    schema=["scene", "molecule", "residue", "atom"],
)

# Subset scenes 0 and 1 (fancy at the outermost level)
sub = df[[0, 1]]
# or: sub = df.scene[[0, 1]]

# Drill down: scene 0, then molecule 1
sub = df.scene[0].molecule[1]
```

Multi-level tuples (`df[i, j]`), nested lists, `:` / `...` skip-level indexing, and coordinate-fancy zip across levels are **not** supported. To combine two coordinate selections, index twice and work with two separate dataframes.

Column selection by string is unchanged: `df["col"]`, `df["a", "b"]`.

Without `schema=`, a single int/slice/bool index still selects **axis-0 rows** as before (`df[:2]`, `df[[True, False]]`).

### Drill-down accessors (`df.level[i].level[j]`)

Attribute access on schema level names composes with single-level indexing:

```python
sub = df.scene[0].molecule[1]
fancy = df.scene[[0, 1]]   # same as df[[0, 1]] on the root dataframe
```

Shallow columns (e.g. `scene_id` at scene depth only) broadcast as length-1 scalars once outer levels are narrowed. Deeper columns keep their nested structure; fancy/slice/bool subsetting uses `Indexed`/`OffsetView` wrappers rather than materializing new layout stacks.

`len(df)` after drill-down reflects the logical row count at the current schema depth (the minimum outer axis length across columns). Use drill-down accessors (`df.scene`, `df.molecule`, …) when you need the per-column view at a specific nesting level.

## Shape (`df.shape(dim=…)`)

Dataframes expose the same **per-axis entity counts** as arrays, using canonical shape metadata (persisted in `grumpy.json` on save) and resident column layouts after indexing:

```python
df = gr.dataframe({...}, schema=["scene", "molecule", "residue", "atom"])

df.shape(dim=0)              # scenes (outer axis at current depth)
df.shape(dim="molecule")     # molecules per scene — nested int64 array
df.shape(dim=1).to_list()    # same as named level when index_depth=0

sub = df.scene[0]
sub.shape(dim=0)             # molecules in scene 0
sub.shape(dim="residue")     # residues per molecule
```

`dim` may be a non-negative integer (relative to the current nesting context) or a **schema level name**. `nshape` ignores nulls at the target axis (like arrays).

On `gr.open`, `shape` reads from stored canon metadata without loading leaf buffers.

### Semantics summary

| Form | Meaning |
|------|---------|
| `df[i]` | Subset entity `i` at the current schema level |
| `df[[i, j]]` | Fancy subset at the current schema level |
| `df.scene[i].molecule[j]` | Drill-down: subset scene, then molecule |
| `df[i, j]` | Not supported (use chained indexing) |
| `df[:1]` (with schema) | Slice at the current schema level |
| `df[:1]` (no schema) | Axis-0 row slice (unchanged) |

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

**Next:** [Saving and loading](saving-loading.md) — persist dataframes to Zarr and lazy `gr.open` access.
