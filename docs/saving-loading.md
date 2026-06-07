# Saving and loading

Training and analysis pipelines need data that outlives a single Python session. Grumpy persists **list-chain** and **union** layouts to **Zarr** directory stores with a `grumpy.json` layout manifest, then reads back only the leaf ranges required for each access. That combination — compact on-disk layout plus partial I/O — is what makes large protein datasets practical without loading everything into RAM.

This page covers one-shot save/load, lazy `open`, and incremental append.

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

### Canonical shape metadata

Dataframes with `schema=` persist **canonical nested shape** (`canon`) in `grumpy.json` — axis-0 length and list-offset vectors per schema level. This powers `df.shape(dim=…)` and `gr.open(…).shape(dim=…)` without reading a reference column from disk.

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

## Lazy open (`gr.open`)

`gr.open` returns an **OpenDataFrame** — a lazy handle over a saved dataframe. Row and schema indexing **materialize** subset dataframes; column selection returns an **OpenColumn** proxy until indexed.

```python
with gr.open("proteins.gr") as h:
    print(len(h))                      # axis-0 length from canon metadata
    print(h.shape(dim="molecule"))     # nested counts without leaf I/O
    batch = h.scene[[0, 5, 12]]        # materialized GrumpyDataFrame
    pos = h.residue_pos[[True, False]] # partial column load via OpenColumn

# Or manage the handle explicitly:
h = gr.open("proteins.gr")
full = h.load()
h.close()
```

Schema drill-down matches in-memory dataframes:

```python
scene = h.scene[0]                 # materialized subset
mol = scene.molecule[1]
```

Dot notation on `open` returns lazy column proxies (`open.residue_pos`); bracket column select (`open["residue_pos"]`) keeps full nesting until indexed.

### Partial axis-0 reads

Low-level `gr._core.load_slice(path, start, stop)` loads a contiguous axis-0 range for arrays or dataframes without opening a long-lived handle. Union datasets read only the referenced scalar/list segments.

### Current limitations

- **`Indexed`** layouts are not supported for on-disk slice loads (materialize before save).
- **`gr.open`** is dataframe-only; use `gr.load` or `load_slice` for arrays.

Fusing transforms into a single Rust plan is covered in [Compilation](compilation.md). Training-time batching, shuffle, and DDP live in **Fabric** (outside Grumpy).

---

**Next:** [Compilation](compilation.md) — fuse transforms into a Rust execution plan.
