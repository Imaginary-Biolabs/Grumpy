# GPU Acceleration

Grumpy accelerates select heavy geometry kernels on GPU when available. Today this is **kNN neighbors** (`gr.neighbors`, `gpu="auto"`). Metal is used on macOS; CUDA when built with `--features cuda`.

## Usage

```python
import grumpy as gr

gr.gpu_available()   # True when Metal/CUDA runtime is present
gr.gpu_backend()     # "metal" | "cuda" | None

# Per-call override
gr.neighbors(x, x, k=16, dim=1, gpu="auto")   # default for Stream.apply
gr.neighbors(x, x, k=16, dim=1, gpu="force")  # always GPU (errors if unavailable)
gr.neighbors(x, x, k=16, dim=1, gpu=False)    # CPU only

# Streaming: gpu propagates into neighbors inside apply()
for batch in gr.stream("coords.gr", gpu="auto").apply(fn, compile="auto"):
    ...
```

## Auto selection (`gpu="auto"`)

Each GPU kernel defines its own minimum work estimate so small batches stay on CPU and avoid fixed launch/sync overhead.

| Op | Gates | Rationale |
|---|---|---|
| **kNN dim=0** | ≥ 128²×3 distance evals | Single launch; moderate clouds amortize host setup |
| **kNN dim=1** | ≥ 4M distance evals **and** ≥ 32 groups | Stream batches (32 proteins × 256 residues) and full in-memory sets |
| **pairwise distances** (future) | ≥ 24M evals **and** ≥ 32 groups | Higher bar for all-pairs work |

*Distance evals* ≈ (query×data pair comparisons) × coordinate dimension.

`gpu="force"` bypasses these thresholds. `gpu="never"` / `False` always uses CPU.

## Limitations

- Metal kernels use float32 internally (Metal has no double); host I/O remains float64.
- CUDA kNN is dim=0 only today; dim=1 falls back to CPU.
- dim=0 kNN still prefers CPU kd-tree when `n ≥ 2048` (GPU applies to brute-force path only).

For multi-core CPU parallelism on streams, see [Compilation](compilation.md) and `Stream.apply(..., cpu=N, scheduler="rust")`.
