# Grumpy agent notes

Grumpy is a Rust + Python (maturin/pyo3) library for ragged/nested numerical data.

For architecture, API examples, and implementation checklists, see the monorepo project file:

- `../.ai/project/grumpy.md`

Coding principles: performance first, thin Python layer, layout-aware kernels, 100% coverage on `python/grumpy/`, descriptive errors.
