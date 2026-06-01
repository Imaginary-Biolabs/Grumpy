# Dataframes

`gr.dataframe` groups named `GrumpyArray` columns. An optional **schema** enforces shared outer shapes per prefix. Columns may use **list-chains** or **`UnionScalarList`** layouts; both are supported for save, load, streaming, and dot notation.

## List-chain schema (fixed nesting)

Typical biomolecule tables: every row at a given schema level has the same list depth.

```python
import grumpy as gr

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

df.residue.residue_weight = [[0.5, 0.7, 0.8], [0.9, 1.0]]
```

## Union columns (mixed scalar/list per row)

Use when a column mirrors heterogeneous database fields: one GO term vs many, one accession vs isoform list, one SMILES vs mixture components.

```python
df = gr.dataframe(
    {
        "molecule_id": ["m1", "m2", "m3"],
        "go_term": [
            "GO:0003674",
            ["GO:0003674", "GO:0005524"],
            [],
        ],
        "residue_name": [["A", "B"], ["C"], ["D", "E"]],
    },
    schema=["molecule", "residue"],
)

# Outer length and molecule prefix still enforced
assert len(df) == 3
df.molecule.go_term.to_list()  # dot notation stacks union innards one level
```

Dot notation (`df.residue.col`) peels nesting axes via `drop_layout_axes`. On union columns, one level is flattened by stacking inner elements; list-chain columns follow the usual peel rules.

## Schema rules

- Column names must start with a valid schema prefix.
- Columns sharing a prefix share outer list offsets (stored once per dataframe) when they are pure list-chains at that level.
- **`UnionScalarList` columns** skip list-offset shape checks at the union axis (scalar vs list rows may differ in inner length) but still enforce **outer length** and **prefix** rules with sibling columns.
- New columns must match schema depth expectations; union columns count as one outer element per row at their schema level.
