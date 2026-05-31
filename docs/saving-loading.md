# Saving and loading

## Save and load

```python
import grumpy as gr

df = gr.dataframe({"a": [1, 2, 3]})
gr.save(df, "data.gr", chunk_size=1024)
df2 = gr.load("data.gr")
```

Stores are Zarr directories with a `grumpy.json` layout manifest.

### Chunk sizing

By default every 1-D buffer uses `chunk_size` (default 1024). Pass `chunk_dim` to chunk only at a specific nesting depth — useful when outer axes are small and inner leaves are large:

```python
gr.save(df, "data.gr", chunk_size=64, chunk_dim="atom")  # schema level name
gr.save(arr, "data.gr", chunk_size=128, chunk_dim=1)     # numeric depth
```

### Incremental writes

Save the first batch normally, then append subsequent batches from a generator or iterator:

```python
def batches():
    for i in range(10):
        yield gr.array([[i, i + 1]], dtype=gr.int64)

gr.save(batches(), "data.gr", chunk_size=256)
assert len(gr.load("data.gr")) == 10
```

Each append loads the existing store, concatenates axis 0, and rewrites. This is suitable for moderate dataset sizes; very large incremental writes may prefer writing batches to separate files.

## Streaming

```python
st = gr.stream("data.gr", batch_size=32, drop_last=False)
print(len(st))  # number of batches (metadata only)

for batch in st:
    process(batch)

# Subset of batches (after DDP sharding)
for batch in st[0:4]:
    process(batch)
```

### Current behavior

- **`len(st)`** uses on-disk metadata (axis-0 offsets); no full leaf load.
- **Each batch** is loaded with partial leaf I/O — only the value ranges needed for that batch are read from Zarr.
- **`batch_on`**, **`shuffle`** / **`seed`**, **`world_size`** / **`rank`**, and **`workers`** (prefetch) are supported; see the API reference.
- **`st[index]`** returns a new stream over selected batch indices.

### Limitations

- `UnionScalarList` and `Indexed` layouts are not supported for streaming slice loads.

## Parallel apply

```python
st2 = st.apply(transform, cpu=8, compile="auto", scheduler="auto")
for batch in st2:
    train(batch)
```

Rust scheduling (`scheduler="rust"`) loads input batches with the same partial I/O path as `Stream.__iter__`.

See [Compilation](compilation.md) for `@gr.compile` and fusion rules.
