# GPU Acceleration

Grumpy accelerates **kNN neighbors** (`gr.neighbors`) on GPU when available. Metal is used on macOS; CUDA when built with `--features cuda`.

CPU-only geometry helpers:

- **Pairwise distances** (`gr.pairwise_distances`)
- **Grid pooling / voxelization** (`gr.grid_pool`)

## Usage

```python
import grumpy as gr

gr.gpu_available()   # True when Metal/CUDA runtime is present
gr.gpu_backend()     # "metal" | "cuda" | None

# Per-call override (kNN only)
gr.neighbors(x, x, k=16, dim=1, gpu="auto")   # default for Stream.apply
gr.neighbors(x, x, k=16, dim=1, gpu="force")  # always GPU (errors if unavailable)
gr.neighbors(x, x, k=16, dim=1, gpu=False)    # CPU only

gr.pairwise_distances(x, dim=1)
gr.grid_pool(x, grid_size=(32, 32, 32), origin=(-2, -3, -3), voxel_size=(3.5, 3.5, 3.5))

# Streaming: gpu propagates into neighbors inside apply()
for batch in gr.stream("coords.gr", gpu="auto").apply(fn, compile="auto"):
    ...
```

## Auto selection (`gpu="auto"`)

Each GPU kernel defines its own minimum work estimate so small batches stay on CPU and avoid fixed launch/sync overhead.

| Op | Gates | Rationale |
|---|---|---|
| **kNN dim=0** | ≥ 128²×3 distance evals | Single launch; moderate clouds amortize host setup |
| **kNN dim=1** | ≥ 4M distance evals **and** ≥ 32 groups | Stream batches and full in-memory sets |

*Distance evals* ≈ (query×data pair comparisons) × coordinate dimension.

`gpu="force"` bypasses these thresholds. `gpu="never"` / `False` always uses CPU.

## Limitations

- Metal kernels use float32 internally (Metal has no double); host I/O remains float64.
- CUDA kNN is dim=0 only today; dim=1 falls back to CPU.
- dim=0 kNN still prefers CPU kd-tree when `n ≥ 2048` (GPU applies to brute-force path only).

For multi-core CPU parallelism on streams, see [Compilation](compilation.md) and `Stream.apply(..., cpu=N, scheduler="rust")`.
