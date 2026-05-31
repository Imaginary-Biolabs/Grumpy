# GPU Acceleration

GPU execution for Grumpy kernels is planned but **not available in the current release**. Today all layout kernels, reductions, and compiled streaming plans run on the CPU (Rust + optional Rayon parallelism).

## Roadmap

- **Device buffers** — layout trees backed by GPU memory with the same ragged semantics as CPU arrays.
- **Kernel fusion** — extend `CompiledPlan` scheduling to CUDA (and optionally Metal) for training-time pipelines already expressed with `@gr.compile` and `Stream.apply`.
- **Host ↔ device I/O** — efficient transfers for Zarr-backed streaming without materializing full datasets.

## Current alternative

Use `Stream.apply(..., cpu=N, scheduler="rust")` for multi-core CPU parallelism on compiled batch transforms. See [Compilation](compilation.md) and [Saving and loading](saving-loading.md).
