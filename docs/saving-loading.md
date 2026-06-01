# Saving and loading

Training and analysis pipelines need data that outlives a single Python session. Grumpy persists **list-chain** and **union** layouts to **Zarr** directory stores with a `grumpy.json` layout manifest, then reads back only the leaf ranges required for each batch. That combination — compact on-disk layout plus partial I/O — is what makes large protein datasets practical without loading everything into RAM.

This page covers one-shot save/load, then streaming with `apply` for batched transforms.

## Save and load

The simplest path writes an array or dataframe to a directory and reads it back:

```python
import grumpy as gr

df = gr.dataframe({
    "id": ["a", "b", "c"],
    "vals": [[1, 2], [3, 4, 5], [6]],
})
gr.save(df, "data.gr", chunk_size=1024)

df2 = gr.load("data.gr")
print(df2.vals.to_list())
```

Union-root arrays round-trip the same way:

```python
x = gr.array(["P12345", ["P12345-1", "P12345-2"], None], dtype=gr.string)
gr.save(x, "union.gr", chunk_size=64)
assert gr.load("union.gr").to_list() == x.to_list()
```

### Chunk sizing

By default every 1-D leaf buffer uses `chunk_size` (default 1024). Tune this when outer axes are small and inner leaves are large — typical for atom-level coordinates:

```python
gr.save(df, "data.gr", chunk_size=64, chunk_dim="atom")   # schema level name
gr.save(arr, "data.gr", chunk_size=128, chunk_dim=1)    # numeric depth
```

### Incremental writes

Save the first batch normally, then append from a generator for moderate-sized incremental builds:

```python
def batches():
    for i in range(10):
        yield gr.array([[i, i + 1]], dtype=gr.int64)

gr.save(batches(), "list.gr", chunk_size=256)
assert len(gr.load("list.gr")) == 10
```

Each append loads the existing store, concatenates axis 0, and rewrites. For very large corpora, prefer writing separate shards or a single upfront save.

## Streaming with apply

`gr.stream` opens a saved store and yields **batches** along axis 0 (or along a schema entity via `batch_on`). Transforms run through `apply`, which supports Python threading and optional Rust scheduling when the pipeline compiles.

### Basic streaming

```python
st = gr.stream("data.gr", batch_size=32, drop_last=False)
print(len(st))  # number of batches — metadata only, no full load

for batch in st:
    process(batch)
```

Subset batches without reloading the file:

```python
for batch in st[0:4]:
    process(batch)
```

### Parallel apply

Pass one or more callables; `cpu` controls transform parallelism (distinct from I/O `workers` on `Stream`):

```python
def transform(batch):
    return batch.vals * 2.0

st2 = st.apply(transform, cpu=4)
for batch in st2:
    train_step(batch)
```

Prefetch I/O while transforms run:

```python
st = gr.stream("data.gr", batch_size=32, workers=2)
for batch in st.apply(transform, cpu=4):
    train_step(batch)
```

### Training-oriented options

For reproducible epoch order, set `shuffle` and `seed`. For multi-GPU data parallel training, partition batches with `world_size` and `rank`:

```python
st = gr.stream(
    "data.gr",
    batch_size=32,
    shuffle="molecule",   # schema level name, or True for axis 0
    seed=42,
    world_size=4,
    rank=0,
)
```

Pack batches by schema entity instead of flat axis 0 when molecules should stay whole:

```python
st = gr.stream("proteins.gr", batch_size=8, batch_on="molecule")
```

Load the full dataset once when RAM allows and batches should be zero-copy slices:

```python
st = gr.stream("data.gr", batch_size=32, in_memory=True)
```

Union datasets use **compact partial I/O**: each batch reads only selected outer rows and the scalar/list segments they reference.

### Current limitations

- **`Indexed`** layouts are not supported for streaming slice loads.

Fusing transforms into a single Rust plan — and when that pays off — is covered in [Compilation](compilation.md).

---

**Next:** [Compilation](compilation.md) — fuse batch transforms and schedule them with Rust across CPU cores.
