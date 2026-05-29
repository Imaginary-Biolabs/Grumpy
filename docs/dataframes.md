# DataFrames and schema

`gr.dataframe` groups named `GrumpyArray` columns. An optional **schema** enforces shared outer shapes per prefix.

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

Dot notation selects nesting levels; assignment must match the indicated depth.

## Schema rules

- Column names must start with a valid schema prefix.
- Columns sharing a prefix share outer list offsets (stored once per dataframe).
