# Saving and loading

Grumpy persists **list-chain** and **`UnionScalarList`** layouts with the same Zarr store format. Streaming, partial I/O, and `batch_on` work on both.

## Save and load

### List-chain dataframe

```python
import grumpy as gr

df = gr.dataframe({"a": [1, 2, 3]})
gr.save(df, "data.gr", chunk_size=1024)
df2 = gr.load("data.gr")
```

### Union-root array

```python
x = gr.array(["P12345", ["P12345-1", "P12345-2"], None], dtype=gr.string)
gr.save(x, "union.gr", chunk_size=64)
assert gr.load("union.gr").to_list() == x.to_list()
```

Stores are Zarr directories with a `grumpy.json` layout manifest.

### Chunk sizing

By default every 1-D buffer uses `chunk_size` (default 1024). Pass `chunk_dim` to chunk only at a specific nesting depth — useful when outer axes are small and inner leaves are large:

```python
gr.save(df, "data.gr", chunk_size=64, chunk_dim="atom")  # schema level name
gr.save(arr, "data.gr", chunk_size=128, chunk_dim=1)     # numeric depth
```

### Incremental writes

Save the first batch normally, then append subsequent batches from a generator or iterator. Works for list-chain and union layouts:

```python
def batches():
    for i in range(10):
        yield gr.array([[i, i + 1]], dtype=gr.int64)

gr.save(batches(), "list.gr", chunk_size=256)

def union_batches():
    for i in range(5):
        yield gr.array([i, [i + 1, i + 2]], dtype=gr.int64)

gr.save(union_batches(), "union.gr", chunk_size=2)
assert len(gr.load("union.gr")) == 5
```

Each append loads the existing store, concatenates axis 0, and rewrites. This is suitable for moderate dataset sizes; very large incremental writes may prefer writing batches to separate files.

## Streaming

### List-chain dataset

```python
st = gr.stream("data.gr", batch_size=32, drop_last=False)
print(len(st))  # number of batches (metadata only)

for batch in st:
    process(batch)

# Subset of batches (after DDP sharding)
for batch in st[0:4]:
    process(batch)
```

### Union-root dataset with `batch_on`

Union stores use **compact partial I/O**: each batch reads only the selected outer rows, referenced scalar indices, and referenced list segments (not full scalar/list pools). Entity counting for `batch_on` walks union tags at each schema depth.

```python
df = gr.dataframe({
    "molecule_id": ["a", "b"],
    "residue_count": [3, [1, 2]],  # union column: scalar vs list per molecule
})
gr.save(df, "union_df.gr", chunk_size=1)

st = gr.stream("union_df.gr", batch_size=1, batch_on="molecule")
for batch in st:
    process(batch)
```

Nested unions inside list-chains use the same partial path when batched.

### Streaming behavior (both layouts)

- **`len(st)`** uses on-disk metadata (axis-0 offsets or union length); no full leaf load.
- **Each batch** is loaded with partial leaf I/O — only the value ranges needed for that batch are read from Zarr.
- **`batch_on`**, **`shuffle`** / **`seed`**, **`world_size`** / **`rank`**, and **`workers`** (prefetch) are supported; see the API reference.
- **`st[index]`** returns a new stream over selected batch indices.

### Current limitations

- **`Indexed`** layouts are not supported for streaming slice loads.

## Parallel apply

Compiled scalar elementwise ops (`batch * 2`, `batch + 1`, …) run on union batches the same way as list-chain batches when the pipeline fuses to a supported plan:

```python
def transform(batch):
    return batch * 2.0

st2 = st.apply(transform, cpu=8, compile="auto", scheduler="auto")
for batch in st2:
    train(batch)
```

Rust scheduling (`scheduler="rust"`) loads input batches with the same partial I/O path as `Stream.__iter__`.

See [Compilation](compilation.md) for `@gr.compile` and fusion rules.
