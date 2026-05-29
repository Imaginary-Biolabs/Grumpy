"""Grumpy: high-performance numerical computing on ragged and nested data.

Grumpy provides Awkward-like layout semantics with strong typing, explicit nullability,
mutable arrays, Zarr-backed I/O, and optional compilation of streaming transforms.

Known limitations
-----------------
- ``UnionScalarList`` layouts are not supported for most ops (use pure list-chains).
- Streaming is axis-0 batching only; advanced dataloader features are planned.
- ``gr.compile`` accepts a restricted subset of Python (see :func:`compile`).
"""

from __future__ import annotations

from ._version import __version__

from ._core import (
    DType,
    GrumpyArray,
    array as _array,
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
    dataframe as _dataframe,
    save as _save,
    load as _load,
)

from . import compiler as _compiler_mod
from .stream import Stream, StreamApply

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
):
    """
    Compute neighbors and return an **edge index** (and optionally distances).

    Returns
    -------
    edge_index:
        Ragged edge index with last axis length 2: [src, dst].
    distances (optional):
        If return_distances=True, also returns distances aligned with the neighbors axis.
    """
    return _neighbors(query, data, k, radius, dim, loop_=loop, return_distances=return_distances)

def dataframe(mapping: dict, schema=None):
    return _dataframe(mapping, schema)


def save(obj, path: str, chunk_size: int = 1024):
    return _save(obj, path, chunk_size)


def load(path: str):
    return _load(path)


def stream(path: str, batch_size: int = 32, drop_last: bool = False) -> Stream:
    return Stream(path=path, batch_size=batch_size, drop_last=drop_last)

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
    "stream",
    "Stream",
    "StreamApply",
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

