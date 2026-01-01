Project Summary

We are going to build “Grumpy”, a Python library for high-performance numerical computing on ragged data. It is meant to be used in various scientific computing scenarios, but mainly for use in another library that deals with biomolecule data. Examples will therefore be based on biological concepts, but the Grumpy library itself is agnostic to the kind of data. It is meant to be a re-implementation of Awkward Arrays and shares many concepts, with few important differences, first and foremost mutability of arrays, and further strong typing and type homogoneity, another definition of Records, and some different indexing semantics. The Grumpy library further extends to efficient data storage and reading based on Zarr, and offers advanced parallel data loading and transformation for applications in machine learning.

Repository structure (high-level)

- Rust core (`src/`)
  - `src/lib.rs`: module wiring + Python module registration.
  - `src/dtype.rs`: `DType` enum + dtype inference/casting helpers. Includes `char` (single unicode scalar) and `string` (variable-length UTF-8).
  - `src/layout.rs`: Awkward-like buffer layout tree:
    - `Leaf` with typed `LeafBuffer` (`Arc<Vec<T>>`) + `validity` bitmap (`Arc<BitVec>`) for nulls.
    - `ListOffset` for ragged lists.
    - `UnionScalarList` for mixed scalar/list structures (variable depth).
    - `Indexed` + `OffsetView` wrappers for view-style transforms without materializing buffers.
  - `src/ops.rs`: elementwise binary ops with rectangular and ragged 2D fast paths, plus broadcasting support.
  - `src/reduce.rs`, `src/unary.rs`, `src/compare.rs`, `src/setops.rs`, `src/stats.rs`, `src/hist.rs`, `src/sortsearch.rs`, `src/whereops.rs`, `src/linalg.rs`, `src/einsum.rs`, `src/neighbors.rs`: NumPy-like functionality implemented in Rust with parity tests.
  - `src/dataframe.rs`: `GrumpyDataFrame` (named columns of `GrumpyArray`) + optional schema that enforces shape constraints.
  - `src/io.rs`: Zarr-backed save/load for `GrumpyArray` and `GrumpyDataFrame` (Milestone 11).
- Python package (`python/grumpy/`)
  - `python/grumpy/__init__.py`: thin user-facing API wrappers and `__all__`.
  - `python/grumpy/stream.py`: minimal streaming + parallel apply (Milestone 12).
- Tests (`tests/`): milestone-based pytest suites for parity + edge cases.
- Benchmarks (`benchmarks/`): kernel-oriented microbenchmarks for elementwise/indexing/reductions/etc.

Design philosophy and key design choices

- Layout-first (Awkward-like): operations are expressed over a small set of layout nodes that describe buffers + offsets, rather than Python lists.
- Typed leaf buffers: leaf storage uses `LeafBuffer` variants (`Arc<Vec<i32>>`, `Arc<Vec<f64>>`, etc.) for fast tight loops and compiler auto-vectorization/SIMD. Strings use `Arc<Vec<String>>` (variable-length).
- Explicit nullability: all dtypes (including numeric) can carry nulls via a validity bitmap; this is propagated by ops where required.
- Views over copies: `OffsetView` and `Indexed` allow slicing/indexing to return cheap views, deferring copies until mutation.
- DataFrame schema: schema defines shared outer shapes by level names (and aliases), enabling dot-notation access/assignment that drops/keeps specified nesting levels.
- Zarr as storage layer (Milestone 11): saved datasets are a directory-based Zarr store with a small `grumpy.json` file describing the layout tree and buffer paths. Buffers are stored as 1D arrays with chunking (`chunk_size`), including experimental Zarr V3 string arrays for `dtype=string`.
- Streaming + parallel apply (Milestone 12): a minimal `Stream` loads a saved dataset and yields axis-0 batches; `apply(..., cpu=N)` supports Python scheduling and Rust scheduling for fully compiled pipelines (bounded prefetch, preserves order).
- Compilation + fusion: `Stream.apply(..., compile='auto')` attempts to compile and fuse consecutive transforms into a single Rust `CompiledPlan` segment; unsupported transforms fall back to Python with a one-time warning (reason included).
- Rust scheduling for compiled segments: for fully compiled pipelines and `cpu>1`, Grumpy can run batches in parallel in Rust (`_core.compiled_stream_apply`) with a Rayon thread pool, avoiding Python executor overhead.
- Neighbors backend: `neighbors(dim=0)` uses an adaptive strategy (brute-force for small `n`, kd-tree for larger `n` in low dimensions); results are stable-sorted by `(distance, index)`; `loop=False` excludes self if query/data share storage. `neighbors(dim=1)` supports grouped point clouds and streamed `OffsetView` batches. **API returns `edge_index`** (and optionally `distances`) for graph construction.
- No-GIL deep reductions: `reduce_array` provides a no-GIL reduction-to-layout engine for arbitrary-depth pure list-chains, enabling reductions in Rust scheduling without creating Python scalars.
- Realistic payload testing: tests/benchmarks include “protein-like” schemas and transforms (scene→molecule→chain→residue→atom; residue centers; residue kNN graphs; compiled dataframe assignment under streaming).

---

How to implement a new op (performance checklist)

This project’s performance comes from combining layout-aware kernels, fast paths, views, and (optionally) compilation/scheduling. When adding a new op, follow this checklist:

1) Pick the right module + API surface
- Put the kernel in a focused Rust module (`src/<group>.rs`).
- Expose it in `src/py_api.rs` as:
  - a `PyGrumpyArray` method (preferred for NumPy-like ops), or
  - a free `#[pyfunction]` if it’s a top-level function (`gr.neighbors`, `gr.where`, …).
- Add a thin wrapper in `python/grumpy/__init__.py` (keep Python logic minimal).

2) Decide layout support (and document it)
- Start with **pure list-chains** (`ListOffset -> … -> Leaf`) and explicitly error on `UnionScalarList` if not supported.
- Make sure the op accepts `OffsetView`/`Indexed` views produced by slicing/streaming (either handle them directly or normalize).
- Prefer extending support by adding wrapper-aware traversal instead of materializing.

3) Implement fast paths first (the “hot loops”)
- If there is a common hot case, write a **single tight loop** kernel:
  - Rectangular 2D list->leaf fast path (one contiguous loop; skip validity checks if all-valid).
  - Ragged 2D list->leaf fast path (row-by-row with per-row broadcast rules).
- Keep branches out of inner loops; specialize per dtype where needed.

4) Use copy-on-write + views correctly
- Leaf buffers and offsets are `Arc<...>`. Mutations must use `Arc::make_mut`.
- If you need to mutate in compiled pipelines, add an **in-place** variant (e.g. scalar ops via `ops::elementwise_scalar_inplace`) to avoid intermediates.
- Avoid copying offsets/content unless required; prefer `OffsetView` and `take_range`.
- If your kernel needs canonical offsets, normalize views with `layout::offsetview_to_listoffset`.

5) Null/NaN semantics
- Nulls are tracked via a validity bitmap; ensure you propagate validity correctly.
- Decide whether NaNs are treated as values or as missing (Grumpy treats NaN as a value; “nan*” ops handle NaN explicitly).
- Be explicit about placeholder semantics for `dim=0` reductions on ragged data (see `src/reduce.rs`).

6) Compilation + Rust scheduling (optional, but important for training-time pipelines)
- If the op is common in streaming transforms, consider making it compilable:
  - Add an IR opcode to `python/grumpy/compiler.py`
  - Add a `PlanOp` in `src/py_api.rs` and execution in `PyCompiledPlan.run`
  - If it should run under Rust scheduling, add it to `run_plan_array_rust` / `run_plan_df_rust` and to the supported-op list in `python/grumpy/stream.py`.
- Ensure the scheduled path does not need Python scalars/GIL (use no-GIL kernels like `reduce_array`).

7) Tests + benchmarks
- Add parity tests (NumPy where possible) and include `OffsetView` batches (streamed slicing).
- Add at least one “protein-like payload” test for the op if it will be used in biomolecule pipelines.
- Add a kernel-only benchmark (avoid output allocation in the timed region) and a pipeline benchmark (streaming + `compile`/`scheduler` modes).

Key Coding Principles
- Write concise and modular code, it should be a joy to read and extend the codebase.
- Extensively document code and use docstrings that result in nicely formatted documentation with ReadTheDocs.
- Do not go the lazy route. Implement a principled library from the ground up. Do not use stubs or placeholders.
- Highest priority is performance. All operations should reach NumPy-like runtime.
- Second highest priority is usability. The API should be elegant, intuitive, and simple to use.
- The codebase is flexible. Check with every new implementation if the structure is still the best. You can restructure the entire codebase if necessary.
- Everything should be Rust-implemented with a thin Python-layer for bindings. Test in Python.
- Write and check tests as you go. Target is 100% coverage. Include all edge-cases you can think of. Check validity and performance against Numpy with rectangular data, but treat it as ragged in Grumpy. Optimize ragged operations and do not cheat performance benchmarks by optimizing only a separate path for rectangular data.
- Clear error reporting. For the developer experience, it is critical that errors and warnings are descriptive and if possible recommend a way to fix it. Catch errors early and implement tons of assertions.
- Before changes to the codebase, re-consider these coding principles.
- You can edit this AGENTS.md file yourself and add summarizing information of the code structure, new rules, and coding principles.

Desired API Structure

```
import grumpy as gr

# ---- Arrays (ragged / nested) ----
# Arrays can be ragged and nested to arbitrary level. If the dtype is not given, it is inferred from the data.
x = gr.array([[1,2,3],[4,5]], dtype=gr.int32)
y = gr.array([[1,2,3],[[None,5],[6]]])
z = gr.array([["a","b"],["c"]], dtype=gr.string)

# ---- Reductions ----
x.mean(dim=1) # returns [2, 4.5]
x.ptp(dim=0) # returns [3,3,None]

# ---- Random (planned / milestone) ----
rng = gr.rng(seed=42)
rng.choice(x, size=2, replace=False, dim=1) # returns [[3,1],[5,4]]
rng.choice(x, size=3, replace=False, dim=1) # raises sample count error
rng.choice(x, size=1, dim=0) # returns [4,5]
rng.choice(x, size=0.5, dim=0) # returns 0.5*length number of elements
rng.choice(x, size=[2,1], dim=1) # returns [[2,1],[5]]
rng.uniform_like(x)

# ---- Neighbors (graph edges) ----
# Returns an edge index (src,dst). Optionally also return distances per edge.
edge_index = gr.neighbors(query, data, k=3, dim=0) # or radius=3.5
edge_index, dists = gr.neighbors(query, data, k=3, dim=0, return_distances=True)

# 2D / grouped example
coords = gr.array([[[1,1],[0,0],[2,3]],[[0,0],[2,2]]]) # 2D coordinates
# grouped (dim=1) returns local indices per group: points -> k -> [src,dst]
gr.neighbors(coords, coords, k=1, loop=False, dim=1) # returns [[[ [0,1] ], [ [1,0] ], [ [2,1] ]], ...]

# The shape of a grumpy array counts the number of elements on the axis
# x.nanshape is the same but ignores NaN in the values
x.shape(dim=0) # returns 2
x.shape(dim=1) # returns [3,2]
y.shape(dim=2) # returns [[] [2, 1]]
y.nanshape(dim=2) # returns [[] [1, 1]]

# ---- Elementwise + broadcasting ----
u = x * 2 + 1
v = (x + x) / 2

# Arrays can be flattened at specified dimensions, where the dimension can be given as a number, negative number, or tuple/list/array. They can also be explicitly exluded.
y.flatten() # returns [1,2,3,None,5,6]
y.flatten(dim=2) # returns [[1,2,3],[None,5,6]]
y.flatten(but=-1) # returns [1,2,3,[None, 5],[6]]
y.flatten(dim=[1,2]) # returns [1,2,3,None,5,6]

# Flattening can be inverted with unflatten
gr.array([1,2,3,4,5,6]).unflatten(sizes=[4,2], dim=0) # returns [[1,2,3,4],[5,6]]
gr.array([[1,2,3],[4,5,6]]).unflatten(sizes=[[2,1],[1,2]], dim=1) # returns [[[1,2],[3],[4],[5,6]]], the sizes must match the number of elements
gr.array([1,2,3,4,5,6]).unflatten(sizes=y.shape(dim=1), dim=0) # returns [[1,2,3],[4,5,6]]

# Arrays can be concatentated
gr.cat([x,y], dim=1) # returns [[1,2,3,1,2,3],[4,5,[None, 5],[6]]]

# There are two ways of indexing in Grumpy: array-indexing and coordinate-indexing
# If the index is an array or list, the result has the same structure as the index
# If indexed with a tuple of multiple numbers or multiple arrays, they are interpreted as coordinates and must be same length
x[0] # coordinate-indexing, returns [1,2,3]
x[[0]] # array-indexing, returns [[1,2,3]]
x[[0,1]] # array-indexing, returns [[1], [5]]
x[[0,1],] # coordinate-indexing, returns [[1,2,3], [4,5]]
x[[:2,1]] # array-indexing, returns [[1,2],[5]]
x[0,0] # coordinate-indexing, returns 1
x[[0,1],[0,0]] # coordinate-indexing, returns [1,4]
x[[0,1,0,1],[0,0,1,1]] # coordinate-indexing, returns [1,4,2,5]
x[[True, False]] # array-indexing, returns [[1,2,3]]

# Arrays are mutable in the same way
x[0] = 100 # [100,[4,5]]
x[[0,1]] = [10,20] # [[10,2,3],[4,20]]
x[[0,1],[0,0]] = [10,20] # [[10,2,3],[20,5]]

# Arrays are broadcasted, also for size arguments in functions and in indexing
x[[0,1]] = 10 # [[10,2,3],[4,10]]
rng.choice(x, size=2, dim=1) # returns [[2,1],[4,5]]
x[[0,1],0] # returns [1,4]

# ---- Unary / comparisons / logical ----
a = gr.array([[0.0, 1.0], [2.0]], dtype=gr.float64)
gr.sin(a)
gr.isnan(a)
gr.less(a, 0.5)
gr.logical_and(gr.isfinite(a), gr.greater_equal(a, 0.0))

# ---- Set routines / statistics / histogram ----
gr.unique(gr.array([3, 1, 1, 2], dtype=gr.int32))
gr.isin(gr.array([1, 2, 3], dtype=gr.int32), gr.array([2, 4], dtype=gr.int32))
gr.std(gr.array([[1.0, 2.0], [3.0]], dtype=gr.float64), dim=1)
gr.histogram(gr.array([0.1, 0.2, 0.9], dtype=gr.float64), bins=3, range=(0.0, 1.0))

# ---- Sorting / searching ----
gr.sort(gr.array([3, 1, 2], dtype=gr.int32))
gr.argsort(gr.array([3.0, 1.0, 2.0], dtype=gr.float64))
gr.partition(gr.array([3, 1, 2], dtype=gr.int32), kth=1)
gr.where(gr.array([True, False, True], dtype=gr.bool), gr.array([1, 2, 3], dtype=gr.int32), gr.array([0, 0, 0], dtype=gr.int32))
gr.argwhere(gr.array([True, False, True], dtype=gr.bool))

# ---- Linear algebra / einsum / tensordot (restricted patterns) ----
gr.dot(gr.array([1.0, 2.0], dtype=gr.float64), gr.array([3.0, 4.0], dtype=gr.float64))
gr.outer(gr.array([1.0, 2.0], dtype=gr.float64), gr.array([3.0, 4.0], dtype=gr.float64))
gr.einsum("ij,jk->ik", gr.array([[1.0, 2.0]], dtype=gr.float64), gr.array([[3.0],[4.0]], dtype=gr.float64))

# The other data container is the grumpy dataframe, which organizes several grumpy arrays in named columns and stores some metadata
df = gr.dataframe({'a':[1,2,3], 'b':[4,[5,6]]})

# operations on the dataframe are applied to all columns
df.max() # returns {'a':3, 'b':6}
df[:2] # returns {'a':[1,2], 'b':[4,[5,6]]}

# dataframes can be subset with string indices
df['a'][:2] # returns {'a':[1,2]}
df['b','a'][:2] # returns {'b':[4,[5,6]], 'a':[1,2]}

# optionally, a schema can be provided at construction, which enforces some shape constraints
# the given schema levels (strings) are used as prefixes for column names, and all columns must start with a valid prefix
# the prefix indicates that all columns with the same prefix share the same shape
# the schema defines prefixes as a list in the order of dimensions, where a tuple indicates same dimension (aliases)
# in this example, the atom columns therefore need to have the same number of elements as the residues in the dimension before, and the same number of elements as the molecules in the dimensions before that.
df = gr.dataframe({'molecule_id':['one','two'], 'residue_name':[['A','B','C'],['D','E']], 'atom_number':[[[1,2],[3,4,5],[6]],[[7,8],[9]]]}, schema=['molecule',('residue','group'),'atom'])

# new columns must have valid shapes
df['molecule_weight'] = [0.5] # error
df_filtered = df[[True, False]] # {'molecule_id':['one'], 'residue_name':[['A','B','C']], 'atom_number':[[[1,2],[3,4,5],[6]]]}
df_filtered['molecule_weight'] = [0.5] # now it works

# internally, the offsets for schema columns are only saved once per dataframe to save space
# however, columns can be further nested, in which case there is an additional offset file saved
df['molecule_measurements'] = [[1,2,3],[4,5]] # "molecule" prefix constraints only apply to the first dimension here, values can be futher nested

# Using string indices for columns results in full nesting as above
# For convenience, dataframe fields can be accessed with a nested dot-notation which indicates the desired nesting level of the returned array
# Only nesting levels explicitly indicated are applied, all others are flattened
df.atom_number # [1,2,3,4,5,6,7,8,9]
df.residue.atom_number # [[1,2],[3,4,5],[6],[7,8],[9]]

# The same principle applies when setting values
# The value needs to have the nesting given by the dot-notation
df.molecule.residue_weight = [[0.5,0.7,0.8],[0.9,1.0]]

# This simplifies some batched operations and avoids explicitly dealing with nesting levels, for example:
df.residue.residue_center = df.residue.atom_pos.mean(dim=-1)

# Arrays and dataframes can be saved, optionally in chunks
# Chunk size can be defined on schema level names or dimension number
gr.save(df, 'path/to/dataframe.gr', chunk_size=1024, chunk_dim='atom')

# Arrays and dataframes can be loaded fully into memory, or streamed over chunks
# Chunks are consumed to fill batch_size and batches are then yielded, loading chunks as necessary
# By default, batching is computed on the outermost dimension, but by providing the "batch_on" argument, batches can be size-limited on other schema columns, to get more equal memory consumption across elements with very different sizes (for example batch_on=residue for proteins of different sizes)
# The batch is then filled with elements until the batch_size is reached, which might result in slightly larger batches than batch_size
# Optionally, the stream can be shuffled, which implements pseudo-shuffling by accessing chunks in random order and then shuffling in memory within chunks after loading, possibly on a given axis of the dataframe
# Chunk loading can be parallelized over multiple workers which reduce load time from disk
# The grumpy stream interface therefore effectively acts as a dataloader that can pipe straight into ML training
df_memory = gr.load('path/to/dataframe.gr') # whole dataframe in memory
st = gr.stream('path/to/dataframe.gr', batch_size=32, batch_on='molecule', drop_last=False, shuffle='molecule', seed=42, workers=4)

# If used in a DDP setting/process from another framework, world_size and rank can be provided and the streamer will load the respective data parts (seed must be the same across ranks)
st = gr.stream('path/to/dataframe.gr', batch_size=32, batch_on='molecule', drop_last=False, shuffle='molecule', seed=42, workers=4, world_size=16, rank=2)

# Streamed dataframes/arrays are generators and are loaded on the fly, but have a length attribute that is gathered from metadata
len(st) # returns number of batches
for batch in st:
	...

# A stream can be indexed like a regular dataframe/array, in which case the order of iteration and yielded data parts are user-defined
# This allows for random access and selective data loading
for batch in st[index]:
	...

# If a generator is provided to gr.save, it is stream-saved to disk filling up the chunks batch by batch, such that the dataset does not need to fit into memory completely
gr.save(generator, 'path/to/dataframe.gr', chunk_size=1024, chunk_dim='atom')

# Grumpy is primarily a library to do fast data processing of structured data
# Therefore, its main purpose it to apply user-written functions and modify data in loops over streams
# These can be parallelized with multiple workers, both on CPU and GPU
# The interface is similar to joblib
st = gr.stream('path/to/dataframe.gr', batch_size=32, batch_on='molecule', shuffle='molecule', workers=4)
def transform(batch):
	batch.residue.residue_center = df.residue.atom_pos.mean(dim=-1)
	return batch
st = st.apply(transform, cpu=8, compile="auto", scheduler="auto") # compilation + rust scheduling when possible
```
