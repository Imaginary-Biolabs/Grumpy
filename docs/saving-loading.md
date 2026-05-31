# Saving and loading

## Save and load

```python
import grumpy as gr

df = gr.dataframe({"a": [1, 2, 3]})
gr.save(df, "data.gr", chunk_size=1024)
df2 = gr.load("data.gr")
```

Stores are Zarr directories with a `grumpy.json` layout manifest.

## Streaming

```python
st = gr.stream("data.gr", batch_size=32, drop_last=False)
print(len(st))  # number of batches (metadata only)

for batch in st:
    process(batch)
```

### Current behavior

- **`len(st)`** uses `stored_len` (axis-0 metadata, no leaf load).
- **Each batch** is loaded with `load_slice(start, stop)` — not the full dataset in RAM.
- **Limitation:** `load_slice` still walks the on-disk layout tree; chunked leaf I/O per batch is future work.

## Parallel apply

```python
st2 = st.apply(transform, cpu=8, compile="auto", scheduler="auto")
for batch in st2:
    train(batch)
```

See [Compilation](compilation.md) for `@gr.compile` and fusion rules.
