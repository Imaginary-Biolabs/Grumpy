"""Grumpy: high-performance numerical computing on ragged and nested data.

Grumpy provides Awkward-like layout semantics with strong typing, explicit nullability,
mutable arrays, Zarr-backed lazy I/O via :func:`open`, and optional compilation of transforms.

Layouts
-------
Arrays use either **list-chains** (``ListOffset -> … -> Leaf``) or **`UnionScalarList``**
(mixed scalar/list rows at one axis). Both are constructed with :func:`array`, persisted to
Zarr, streamed, and used in dataframes.

Notes
-----
- :func:`open` exposes lazy on-disk dataframes; row/schema indexing materializes subsets;
  column access returns lazy column proxies until indexed.
- ``gr.compile`` accepts a restricted subset of Python (see :func:`compile`); scalar
  elementwise opcodes fuse on union batches as well as list-chains.
"""

from __future__ import annotations

from ._version import __version__

from ._core import (
    DType,
    GrumpyArray,
    GrumpyDataFrame,
    Generator,
    array as _array,
    multiply as _multiply,
    add_arrays as _add_arrays,
    subtract as _subtract,
    cat as _cat,
    full_like as _full_like,
    ones_like as _ones_like,
    zeros_like as _zeros_like,
    unique as _unique,
    isin as _isin,
    setdiff as _setdiff,
    setunion as _setunion,
    setxor as _setxor,
    bincount as _bincount,
    digitize as _digitize,
    histogram as _histogram,
    nonzero as _nonzero,
    search_sorted as _search_sorted,
    where_ as _where,
    argwhere as _argwhere,
    dot as _dot,
    inner as _inner,
    outer as _outer,
    trace as _trace,
    norm as _norm,
    cross as _cross,
    det as _det,
    inv as _inv,
    einsum as _einsum,
    tensordot as _tensordot,
    neighbors as _neighbors,
    pairwise_distances as _pairwise_distances,
    grid_pool as _grid_pool,
    gpu_available as _gpu_available,
    gpu_backend as _gpu_backend,
    dataframe as _dataframe,
    save as _save,
    append_batch as _append_batch,
    load as _load,
    open_dataset as _open,
    OpenDataFrame,
    OpenColumn,
    rng as _rng,
    py_can_cast as _can_cast,
    py_promote_types as _promote_types,
)

from . import compiler as _compiler_mod
compile = _compiler_mod.compile

# Public dtype singletons (match the API examples).
int8 = DType.int8()
int16 = DType.int16()
int32 = DType.int32()
int64 = DType.int64()

uint8 = DType.uint8()
uint16 = DType.uint16()
uint32 = DType.uint32()
uint64 = DType.uint64()

float16 = DType.float16()
float32 = DType.float32()
float64 = DType.float64()

bool_ = DType.bool_()
char = DType.char()
string = DType.string()


def array(obj, dtype: DType | None = None) -> GrumpyArray:
    """
    Construct a GrumpyArray from Python scalars / nested lists or tuples.

    Parameters
    ----------
    obj:
        Python scalar or nested Python sequences (lists/tuples) of arbitrary depth.
    dtype:
        Optional explicit dtype. If omitted, dtype is inferred from non-null leaves.
    """
    return _array(obj, dtype)


def can_cast(from_dtype: DType, to_dtype: DType, casting: str = "safe") -> bool:
    """Return whether ``from_dtype`` can be cast to ``to_dtype`` under ``casting``."""
    return _can_cast(from_dtype, to_dtype, casting)


def promote_types(a: DType, b: DType) -> DType:
    """NumPy-style binary result dtype for two dtypes."""
    return _promote_types(a, b)


def multiply(a: GrumpyArray, b: GrumpyArray, out: GrumpyArray | None = None) -> GrumpyArray:
    """Elementwise multiply with optional pre-allocated ``out`` (NumPy ``out=`` style)."""
    return _multiply(a, b, out)


def add(a: GrumpyArray, b: GrumpyArray, out: GrumpyArray | None = None) -> GrumpyArray:
    """Elementwise add with optional pre-allocated ``out``."""
    return _add_arrays(a, b, out)


def subtract(a: GrumpyArray, b: GrumpyArray, out: GrumpyArray | None = None) -> GrumpyArray:
    """Elementwise subtract with optional pre-allocated ``out``."""
    return _subtract(a, b, out)


def cat(arrays: list[GrumpyArray], dim: int = 0) -> GrumpyArray:
    """Concatenate arrays along a ragged dimension."""
    return _cat(arrays, dim)


def full_like(x: GrumpyArray, fill_value, dtype: DType | None = None) -> GrumpyArray:
    """Create an array with the same ragged structure as `x`, filled with `fill_value`."""
    return _full_like(x, fill_value, dtype)


def zeros_like(x: GrumpyArray, dtype: DType | None = None) -> GrumpyArray:
    """Create an array with the same ragged structure as `x`, filled with zeros."""
    return _zeros_like(x, dtype)


def ones_like(x: GrumpyArray, dtype: DType | None = None) -> GrumpyArray:
    """Create an array with the same ragged structure as `x`, filled with ones."""
    return _ones_like(x, dtype)


def unique(x: GrumpyArray) -> GrumpyArray:
    return _unique(x)


def isin(x: GrumpyArray, test_elements: GrumpyArray) -> GrumpyArray:
    return _isin(x, test_elements)


def setdiff(a: GrumpyArray, b: GrumpyArray) -> GrumpyArray:
    return _setdiff(a, b)


def setunion(a: GrumpyArray, b: GrumpyArray) -> GrumpyArray:
    return _setunion(a, b)


def setxor(a: GrumpyArray, b: GrumpyArray) -> GrumpyArray:
    return _setxor(a, b)


def var(x: GrumpyArray, dim: int = 0, ddof: int = 0) -> GrumpyArray:
    return x.var(dim, ddof)


def std(x: GrumpyArray, dim: int = 0, ddof: int = 0) -> GrumpyArray:
    return x.std(dim, ddof)


def nanvar(x: GrumpyArray, dim: int = 0, ddof: int = 0) -> GrumpyArray:
    return x.nanvar(dim, ddof)


def nanstd(x: GrumpyArray, dim: int = 0, ddof: int = 0) -> GrumpyArray:
    return x.nanstd(dim, ddof)


def quantile(x: GrumpyArray, q: float, dim: int = 0) -> GrumpyArray:
    return x.quantile(q, dim)


def nanquantile(x: GrumpyArray, q: float, dim: int = 0) -> GrumpyArray:
    return x.nanquantile(q, dim)


def percentile(x: GrumpyArray, q: float, dim: int = 0) -> GrumpyArray:
    return x.percentile(q, dim)


def nanpercentile(x: GrumpyArray, q: float, dim: int = 0) -> GrumpyArray:
    return x.nanpercentile(q, dim)


def median(x: GrumpyArray, dim: int = 0) -> GrumpyArray:
    return x.median(dim)


def nanmedian(x: GrumpyArray, dim: int = 0) -> GrumpyArray:
    return x.nanmedian(dim)


def bincount(x: GrumpyArray, weights: GrumpyArray | None = None, minlength: int = 0) -> GrumpyArray:
    return _bincount(x, weights, minlength)


def digitize(x: GrumpyArray, bins: GrumpyArray, right: bool = False) -> GrumpyArray:
    return _digitize(x, bins, right)


def histogram(
    x: GrumpyArray,
    bins: int = 10,
    range: tuple[float, float] | None = None,  # noqa: A002 - match NumPy API
    density: bool = False,
    weights: GrumpyArray | None = None,
) -> tuple[GrumpyArray, GrumpyArray]:
    return _histogram(x, bins, range, density, weights)


def nonzero(x: GrumpyArray) -> GrumpyArray:
    return _nonzero(x)


def search_sorted(x: GrumpyArray, v: GrumpyArray, right: bool = False) -> GrumpyArray:
    return _search_sorted(x, v, right)


def where(cond: GrumpyArray, x: GrumpyArray | None = None, y: GrumpyArray | None = None):
    # Matches NumPy-style: where(cond) -> indices; where(cond,x,y) -> selected array.
    return _where(cond, x, y)


def argwhere(cond: GrumpyArray) -> GrumpyArray:
    return _argwhere(cond)

def dot(a: GrumpyArray, b: GrumpyArray):
    return _dot(a, b)


def inner(a: GrumpyArray, b: GrumpyArray):
    return _inner(a, b)


def outer(a: GrumpyArray, b: GrumpyArray) -> GrumpyArray:
    return _outer(a, b)


def trace(a: GrumpyArray):
    return _trace(a)


def norm(a: GrumpyArray):
    return _norm(a)


def cross(a: GrumpyArray, b: GrumpyArray) -> GrumpyArray:
    return _cross(a, b)


def det(a: GrumpyArray):
    return _det(a)


def inv(a: GrumpyArray) -> GrumpyArray:
    return _inv(a)

def einsum(subscripts: str, *operands):
    return _einsum(subscripts, *operands)


def tensordot(a: GrumpyArray, b: GrumpyArray, axes: int = 2):
    return _tensordot(a, b, axes)

def neighbors(
    query: GrumpyArray,
    data: GrumpyArray,
    k: int | None = None,
    radius: float | None = None,
    dim: int = 0,
    loop: bool = True,
    return_distances: bool = False,
    gpu: bool | str = False,
):
    """
    Compute neighbors and return an **edge index** (and optionally distances).

    Parameters
    ----------
    query, data:
        Coordinate arrays. For ``dim=1``, shape is ``(n_groups, n_points, d)``.
    k:
        Number of nearest neighbors (mutually exclusive with ``radius``).
    radius:
        Include all neighbors within this distance (mutually exclusive with ``k``).
    dim:
        Axis along which groups of points live (``0`` or ``1`` for grouped clouds).
    loop:
        If ``True``, include self-matches where query and data share the same point index.
    return_distances:
        If ``True``, also return distances aligned with the neighbor axis.
    gpu:
        ``False`` (default) uses CPU. ``True`` always uses GPU when available
        (Metal on macOS, CUDA on Linux with ``--features cuda``).
        ``'auto'`` selects GPU only for large enough batches.
        Use :func:`gpu_available` to check runtime support.

    Returns
    -------
    edge_index:
        Ragged edge index with last axis length 2: ``[src, dst]``.
    distances (optional):
        Returned when ``return_distances=True``.
    """
    if gpu not in (True, False, "auto"):
        from .errors import arg_one_of

        arg_one_of("gpu", gpu, ("True", "False", "'auto'"))
    gpu_arg = "auto" if gpu == "auto" else ("true" if gpu else "false")
    return _neighbors(query, data, k, radius, dim, loop_=loop, return_distances=return_distances, gpu=gpu_arg)


def pairwise_distances(x: GrumpyArray, *, dim: int = 1) -> GrumpyArray:
    """
    All-pairs Euclidean distances within each point cloud (group).

    For ``dim=1``, input shape is ``(n_groups, n_points, d)``; output is
    ``(n_groups, n_points, n_points)`` distance matrices.
    """
    return _pairwise_distances(x, dim)


def grid_pool(
    x: GrumpyArray,
    grid_size: tuple[int, int, int],
    *,
    origin: tuple[float, float, float] | None = None,
    voxel_size: tuple[float, float, float] | None = None,
    dim: int = 1,
) -> GrumpyArray:
    """
    Voxelize point clouds by counting points per grid cell (occupancy pooling).

    Returns ``(n_groups, nx*ny*nz)`` occupancy grids per group.
    """
    return _grid_pool(x, grid_size, origin, voxel_size, dim)


def gpu_available() -> bool:
    """Return True when a GPU backend (Metal or CUDA) is available."""
    return _gpu_available()


def gpu_backend() -> str | None:
    """Return ``'metal'``, ``'cuda'``, or ``None`` if no GPU backend is active."""
    return _gpu_backend()

def dataframe(mapping: dict, schema=None):
    return _dataframe(mapping, schema)


def save(obj, path: str, chunk_size: int = 1024, chunk_dim=None):
    """Save a GrumpyArray/DataFrame, or incrementally write batches from a generator."""
    import types

    chunk_arg = None if chunk_dim is None else str(chunk_dim)
    if isinstance(obj, (GrumpyArray, GrumpyDataFrame)):
        return _save(obj, path, chunk_size, chunk_arg)
    if isinstance(obj, types.GeneratorType) or (
        hasattr(obj, "__iter__") and hasattr(obj, "__next__") and not isinstance(obj, (str, bytes))
    ):
        it = iter(obj)
        try:
            first = next(it)
        except StopIteration as exc:
            from .errors import format_grumpy_error

            raise ValueError(
                format_grumpy_error(
                    "ArgumentInvalid",
                    "save(generator): iterator produced no batches",
                    cause="gr.save from a generator requires at least one yielded batch to infer schema and layout.",
                    fix="yield at least one GrumpyArray or GrumpyDataFrame before the generator ends.",
                )
            ) from exc
        _save(first, path, chunk_size, chunk_arg)
        for batch in it:
            _append_batch(batch, path, chunk_size, chunk_arg)
        return None
    return _save(obj, path, chunk_size, chunk_arg)


def load(path: str):
    """Load a saved dataset fully into memory."""
    return _load(path)


def open(path: str) -> OpenDataFrame:
    """Open a saved dataset as a lazy on-disk handle.

    Row and schema indexing materialize subset dataframes. Column selection
    (``df['col']`` or ``df.col``) returns an :class:`OpenColumn` proxy until
    the column is indexed.

    Call :meth:`~grumpy.OpenDataFrame.close` when finished, or use a context
    manager::

        with gr.open("data.gr") as h:
            batch = h[[0, 1, 2]]
    """
    return _open(path)


def rng(seed: int = 0) -> Generator:
    """Create a reproducible random :class:`~grumpy.Generator`."""
    return _rng(seed)


def sin(x: GrumpyArray) -> GrumpyArray:
    return x.sin()


def cos(x: GrumpyArray) -> GrumpyArray:
    return x.cos()


def tan(x: GrumpyArray) -> GrumpyArray:
    return x.tan()


def exp(x: GrumpyArray) -> GrumpyArray:
    return x.exp()


def log(x: GrumpyArray) -> GrumpyArray:
    return x.log()


def log10(x: GrumpyArray) -> GrumpyArray:
    return x.log10()


def log2(x: GrumpyArray) -> GrumpyArray:
    return x.log2()


def sqrt(x: GrumpyArray) -> GrumpyArray:
    return x.sqrt()


def abs(x: GrumpyArray) -> GrumpyArray:  # noqa: A001 - match NumPy API
    return x.abs()


def sign(x: GrumpyArray) -> GrumpyArray:
    return x.sign()


def floor(x: GrumpyArray) -> GrumpyArray:
    return x.floor()


def ceil(x: GrumpyArray) -> GrumpyArray:
    return x.ceil()


def round(x: GrumpyArray) -> GrumpyArray:  # noqa: A001 - match NumPy API
    return x.round()


def reciprocal(x: GrumpyArray) -> GrumpyArray:
    return x.reciprocal()


def angle(x: GrumpyArray) -> GrumpyArray:
    return x.angle()


__all__ = [
    "__version__",
    "compile",
    "GrumpyArray",
    "DType",
    "array",
    "can_cast",
    "promote_types",
    "cat",
    "full_like",
    "zeros_like",
    "ones_like",
    "unique",
    "isin",
    "setdiff",
    "setunion",
    "setxor",
    "var",
    "std",
    "nanvar",
    "nanstd",
    "quantile",
    "nanquantile",
    "percentile",
    "nanpercentile",
    "median",
    "nanmedian",
    "bincount",
    "digitize",
    "histogram",
    "nonzero",
    "search_sorted",
    "where",
    "argwhere",
    "dot",
    "inner",
    "outer",
    "trace",
    "norm",
    "cross",
    "det",
    "inv",
    "einsum",
    "tensordot",
    "neighbors",
    "dataframe",
    "save",
    "load",
    "open",
    "OpenDataFrame",
    "OpenColumn",
    "rng",
    "Generator",
    "sin",
    "cos",
    "tan",
    "exp",
    "log",
    "log10",
    "log2",
    "sqrt",
    "abs",
    "sign",
    "floor",
    "ceil",
    "round",
    "reciprocal",
    "angle",
    "isnan",
    "isfinite",
    "isinf",
    "equal",
    "not_equal",
    "less",
    "less_equal",
    "greater",
    "greater_equal",
    "logical_and",
    "logical_or",
    "logical_xor",
    "logical_not",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float16",
    "float32",
    "float64",
    "bool_",
    "char",
    "string",
]


def isnan(x: GrumpyArray) -> GrumpyArray:
    return x.isnan()


def isfinite(x: GrumpyArray) -> GrumpyArray:
    return x.isfinite()


def isinf(x: GrumpyArray) -> GrumpyArray:
    return x.isinf()


def equal(a: GrumpyArray, b: GrumpyArray) -> GrumpyArray:
    return a.equal(b)


def not_equal(a: GrumpyArray, b: GrumpyArray) -> GrumpyArray:
    return a.not_equal(b)


def less(a: GrumpyArray, b: GrumpyArray) -> GrumpyArray:
    return a.less(b)


def less_equal(a: GrumpyArray, b: GrumpyArray) -> GrumpyArray:
    return a.less_equal(b)


def greater(a: GrumpyArray, b: GrumpyArray) -> GrumpyArray:
    return a.greater(b)


def greater_equal(a: GrumpyArray, b: GrumpyArray) -> GrumpyArray:
    return a.greater_equal(b)


def logical_and(a: GrumpyArray, b: GrumpyArray) -> GrumpyArray:
    return a.logical_and(b)


def logical_or(a: GrumpyArray, b: GrumpyArray) -> GrumpyArray:
    return a.logical_or(b)


def logical_xor(a: GrumpyArray, b: GrumpyArray) -> GrumpyArray:
    return a.logical_xor(b)


def logical_not(a: GrumpyArray) -> GrumpyArray:
    return a.logical_not()


def _apply_function_docs() -> None:
    for name, text in FUNCTION_DOCS.items():
        fn = globals().get(name)
        if fn is not None and callable(fn):
            fn.__doc__ = text


from ._docinit import FUNCTION_DOCS  # noqa: E402
from ._docinject import inject_all  # noqa: E402

inject_all()
_apply_function_docs()
