# Compilation and parallel apply

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
- `batch <op> scalar` for `+ - * / %`
- `batch.sum/mean/min/max/ptp(dim=...)`
- `gr.neighbors(batch, batch, k=..., dim=..., loop=...)`
- Dataframe dot assignments: `batch.level.col = batch.level.other.mean(dim=-1)`

Unsupported constructs fall back to Python with a one-time warning.

## `Stream.apply(compile=...)`

- `"auto"` — compile when the full pipeline fuses into one plan
- `"force"` / `True` — require compilation where possible
- `"never"` / `False` — Python only

## Schedulers

- `"python"` — `ThreadPoolExecutor` over batches
- `"rust"` — Rayon scheduling for fully compiled pipelines (`cpu > 1`)
- `"auto"` — pick Rust when opcodes are supported
