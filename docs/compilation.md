# Compilation

Python batch loops spend time crossing the interpreter boundary on every operation. **`@gr.compile`** analyzes a restricted subset of your transform, builds a **CompiledPlan** of Rust opcodes, and executes them in one fused pass per batch — with the GIL released for array and dataframe batches.

Compilation matters most when the same transform runs thousands of times across training epochs. Pair **`@gr.compile`** with **`gr.open`** for lazy Zarr-backed batches, or call compiled wrappers on in-memory dataframes directly.

## What compilation does

When compilation succeeds, Grumpy replaces your Python function body with a fixed opcode sequence — scalar elementwise math, reductions, kNN neighbors, and certain dataframe dot-assignments — executed entirely in Rust.

```python
import grumpy as gr

@gr.compile
def scale(batch):
    return batch * 2.0 + 1.0

x = gr.array([[1, 2], [3]], dtype=gr.float64)
print(scale(x).to_list())   # [[3.0, 5.0], [7.0]]
print(scale.is_compiled)    # True
```

### With lazy open

Materialize axis-0 slices from a saved dataset and run the compiled wrapper on each batch:

```python
gr.save(x, "data.gr")

@gr.compile
def scale(batch):
    return batch * 2.0 + 1.0

with gr.open("data.gr") as h:
    for start in range(0, len(h), 32):
        out = scale(h[start : start + 32])
        train_step(out)
```

If analysis fails at decoration time, the wrapper still runs as plain Python and emits a **one-time warning** on the first call; the transform remains correct.

## When compilation pays off

The homepage compile benchmark chart compares eager Python vs compiled paths on a protein-like **`gr.open`** mini-epoch. Gains show up when **multiple ops fuse** into one plan (elementwise chains, normalize + kNN + pool, and similar).

Union batches support the same scalar elementwise opcodes as list-chains:

```python
u = gr.array([1.0, [2.0, 3.0], 4.0], dtype=gr.float64)
gr.save(u, "u.gr", chunk_size=2)

@gr.compile
def double(batch):
    return batch * 2.0

with gr.open("u.gr") as h:
    out = double(h[[0]])
```

Multi-function pipelines fuse when each step is compilable:

```python
def stage_a(batch):
    return batch * 2.0

def stage_b(batch):
    return batch + 1.0

run = gr.compiler.compile_pipeline([stage_a, stage_b])
with gr.open("data.gr") as h:
    out = run(h[[0]])
```

Epoch-level shuffle, DDP sharding, and parallel batch scheduling live in **Fabric** (outside Grumpy).

## Writing compilable functions

Follow these rules so static analysis can build a plan:

1. **Straight-line code only** — no `if`, `for`, `while`, `try`, imports, or nested function definitions.
2. **Single argument** named by convention `batch` (a `GrumpyArray` or `GrumpyDataFrame` batch).
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
