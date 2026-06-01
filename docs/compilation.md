# Compilation

Python batch loops spend time crossing the interpreter boundary on every operation. **`@gr.compile`** and **`Stream.apply(compile=...)`** analyze a restricted subset of your transform, build a **CompiledPlan** of Rust opcodes, and execute them in one fused pass per batch — often with Rayon scheduling when `cpu > 1`.

Compilation matters most in **Zarr streaming** pipelines where the same transform runs thousands of times across epochs. Eager one-off calls on in-memory arrays rarely need it.

## What compilation does

When compilation succeeds, Grumpy replaces your Python function body with a fixed opcode sequence — scalar elementwise math, reductions, kNN neighbors, and certain dataframe dot-assignments — executed entirely in Rust while the GIL is released.

Decorate a function or pass it to `apply`:

```python
import grumpy as gr

@gr.compile
def scale(batch):
    return batch * 2.0 + 1.0

x = gr.array([[1, 2], [3]], dtype=gr.float64)
print(scale(x).to_list())   # [[3.0, 5.0], [7.0]]
print(scale.is_compiled)    # True
```

The same function inside a stream:

```python
gr.save(x, "data.gr")

st = gr.stream("data.gr", batch_size=1)
for out in st.apply(scale, compile="auto"):
    train_step(out)
```

If analysis fails, Grumpy falls back to plain Python and emits a **one-time warning**; the transform still runs correctly.

## When compilation kicks in

`Stream.apply` accepts `compile=`:

| Value | Behavior |
|-------|----------|
| `"auto"` (default) | Compile when the full pipeline fuses into one supported plan |
| `True` / `"force"` | Require compilation; warn or fall back if unsupported |
| `False` / `"never"` | Always run Python callables |

Scheduling is separate via `scheduler=`:

| Value | Behavior |
|-------|----------|
| `"auto"` | Use Rust Rayon batch scheduling when the plan is fully compiled and `cpu > 1` |
| `"python"` | `ThreadPoolExecutor` over batches |
| `"rust"` | Require Rust scheduling (falls back with a warning if the plan is not fully compiled) |

Compilation pays off primarily when **multiple ops fuse** and **`cpu > 1`** with `scheduler="auto"` — the homepage compile benchmark chart compares Python vs compiled paths on a protein-like stream.

Union batches support the same scalar elementwise opcodes as list-chains when loaded from stream or memory:

```python
u = gr.array([1.0, [2.0, 3.0], 4.0], dtype=gr.float64)
gr.save(u, "u.gr", chunk_size=2)

@gr.compile
def double(batch):
    return batch * 2.0

st = gr.stream("u.gr", batch_size=1)
out = list(st.apply(double, compile=True, scheduler="rust"))
```

## Writing compilable functions

Follow these rules so static analysis can build a plan:

1. **Straight-line code only** — no `if`, `for`, `while`, `try`, imports, or nested function definitions.
2. **Single argument** named by convention `batch` (the stream batch object).
3. **Supported statements** — see list below.

### Supported constructs (MVP)

- `batch <op> scalar` for `+`, `-`, `*`, `/`, `%` (list-chain and union layouts)
- `batch.sum()`, `batch.mean()`, `batch.min()`, `batch.max()`, `batch.ptp()` with optional `dim=`
- `gr.neighbors(batch, batch, k=..., dim=..., loop=...)`
- Dataframe dot assignments, e.g.  
  `batch.residue.center = batch.residue.coords.mean(dim=-1)`

Chaining fuses into one plan:

```python
@gr.compile
def normalize_and_pool(batch):
    batch = batch * 0.01
    batch = batch + 1.0
    return batch.mean(dim=1)
```

Multi-function pipelines in one `apply` call fuse when each step is compilable:

```python
def stage_a(batch):
    return batch * 2.0

def stage_b(batch):
    return batch + 1.0

for out in st.apply([stage_a, stage_b], compile="auto", cpu=4, scheduler="auto"):
    train_step(out)
```

### Unsupported (falls back to Python)

- Control flow and exception handling
- Arbitrary method calls outside the supported set
- Multiple parameters or closures capturing external state
- Ops not yet implemented for the batch layout (see [Developer](developer.md) for layout notes)

Inspect compilation status on the wrapper:

```python
@gr.compile
def maybe(batch):
    return batch * 2.0

print(maybe.is_compiled, maybe.compile_error)
```

---

**Next:** [API Reference](api.md) — generated documentation for every public function and core type.
