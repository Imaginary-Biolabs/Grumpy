# Compilation

## `@gr.compile`

Decorate batch transforms to run a fused Rust `CompiledPlan` when the source is statically analyzable:

```python
import grumpy as gr

@gr.compile
def transform(batch):
    batch = batch * 2
    batch = batch + 1
    return batch
```

Supported (MVP):

- Straight-line code, single `batch` argument
- `batch <op> scalar` for `+ - * / %` (list-chain **and** union layouts)
- `batch.sum/mean/min/max/ptp(dim=...)`
- `gr.neighbors(batch, batch, k=..., dim=..., loop=...)`
- Dataframe dot assignments: `batch.level.col = batch.level.other.mean(dim=-1)`

Unsupported constructs fall back to Python with a one-time warning.

### Union batches

Scalar elementwise fusion works on **`UnionScalarList`** inputs loaded from stream or memory:

```python
x = gr.array([1.0, [2.0, 3.0], 4.0], dtype=gr.float64)
gr.save(x, "u.gr", chunk_size=2)

@gr.compile
def double(batch):
    return batch * 2.0

st = gr.stream("u.gr", batch_size=1)
out = list(st.apply(double, compile=True, scheduler="rust"))
```

Reduction and neighbor opcodes in compiled plans follow the same layout rules as eager execution (union support where the underlying kernel supports it).

## `Stream.apply(compile=...)`

- `"auto"` — compile when the full pipeline fuses into one plan
- `"force"` / `True` — require compilation where possible
- `"never"` / `False` — Python only

## Schedulers

- `"python"` — `ThreadPoolExecutor` over batches
- `"rust"` — Rayon scheduling for fully compiled pipelines (`cpu > 1`)
- `"auto"` — pick Rust when opcodes are supported
