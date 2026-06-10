"""Docstrings for :mod:`grumpy` top-level functions."""

from __future__ import annotations

from ._docutil import doc

FUNCTION_DOCS: dict[str, str] = {
    "array": doc(
        "Construct a :class:`GrumpyArray` from Python scalars or nested sequences.",
        params=[
            "obj : scalar or nested sequence",
            "    Python scalar or nested lists/tuples of arbitrary depth.",
            "dtype : DType, optional",
            "    Explicit dtype. Inferred from non-null leaves when omitted.",
        ],
        returns="GrumpyArray\n    New array.",
        examples=[
            ">>> import grumpy as gr",
            ">>> gr.array([[1, 2], [3]], dtype=gr.int32).to_list()",
            "[[1, 2], [3]]",
        ],
    ),
    "multiply": doc(
        "Elementwise multiply with optional pre-allocated output.",
        params=[
            "a, b : GrumpyArray",
            "    Input arrays.",
            "out : GrumpyArray, optional",
            "    Pre-allocated output (NumPy ``out=`` style).",
        ],
        returns="GrumpyArray\n    ``a * b`` with broadcasting.",
        examples=[
            ">>> import grumpy as gr",
            ">>> a = gr.array([1, 2, 3])",
            ">>> gr.multiply(a, a).to_list()",
            "[1, 4, 9]",
        ],
    ),
    "add": doc(
        "Elementwise add with optional pre-allocated output.",
        params=[
            "a, b : GrumpyArray",
            "    Input arrays.",
            "out : GrumpyArray, optional",
            "    Pre-allocated output.",
        ],
        returns="GrumpyArray\n    ``a + b`` with broadcasting.",
        examples=[
            ">>> import grumpy as gr",
            ">>> gr.add(gr.array([1]), gr.array([2])).to_list()",
            "[3]",
        ],
    ),
    "subtract": doc(
        "Elementwise subtract with optional pre-allocated output.",
        params=[
            "a, b : GrumpyArray",
            "    Input arrays.",
            "out : GrumpyArray, optional",
            "    Pre-allocated output.",
        ],
        returns="GrumpyArray\n    ``a - b`` with broadcasting.",
        examples=[
            ">>> import grumpy as gr",
            ">>> gr.subtract(gr.array([3]), gr.array([1])).to_list()",
            "[2]",
        ],
    ),
    "cat": doc(
        "Concatenate arrays along a ragged dimension.",
        params=[
            "arrays : list[GrumpyArray]",
            "    Arrays to concatenate.",
            "dim : int, default 0",
            "    Axis along which to concatenate.",
        ],
        returns="GrumpyArray\n    Concatenated array.",
        examples=[
            ">>> import grumpy as gr",
            ">>> a = gr.array([1, 2])",
            ">>> b = gr.array([3])",
            ">>> gr.cat([a, b]).to_list()",
            "[1, 2, 3]",
        ],
    ),
    "full_like": doc(
        "Create an array with the same layout as ``x``, filled with a constant.",
        params=[
            "x : GrumpyArray",
            "    Template array.",
            "fill_value : scalar",
            "    Value to write into every leaf.",
            "dtype : DType, optional",
            "    Output dtype (defaults to ``x.dtype``).",
        ],
        returns="GrumpyArray\n    Filled array.",
        examples=[
            ">>> import grumpy as gr",
            ">>> x = gr.array([[1, 2], [3]])",
            ">>> gr.full_like(x, 0).to_list()",
            "[[0, 0], [0]]",
        ],
    ),
    "zeros_like": doc(
        "Create a zero-filled array with the same layout as ``x``.",
        params=[
            "x : GrumpyArray",
            "    Template array.",
            "dtype : DType, optional",
            "    Output dtype.",
        ],
        returns="GrumpyArray\n    Zero-filled array.",
        examples=[
            ">>> import grumpy as gr",
            ">>> gr.zeros_like(gr.array([1, 2])).to_list()",
            "[0, 0]",
        ],
    ),
    "ones_like": doc(
        "Create a one-filled array with the same layout as ``x``.",
        params=[
            "x : GrumpyArray",
            "    Template array.",
            "dtype : DType, optional",
            "    Output dtype.",
        ],
        returns="GrumpyArray\n    One-filled array.",
        examples=[
            ">>> import grumpy as gr",
            ">>> gr.ones_like(gr.array([1, 2])).to_list()",
            "[1, 1]",
        ],
    ),
    "unique": doc(
        "Return sorted unique leaf values.",
        params=["x : GrumpyArray", "    Input array."],
        returns="GrumpyArray\n    Unique values.",
        examples=[
            ">>> import grumpy as gr",
            ">>> gr.unique(gr.array([1, 2, 1])).to_list()",
            "[1, 2]",
        ],
    ),
    "isin": doc(
        "Test whether each leaf of ``x`` is contained in ``test_elements``.",
        params=[
            "x : GrumpyArray",
            "    Values to test.",
            "test_elements : GrumpyArray",
            "    Candidate values.",
        ],
        returns="GrumpyArray\n    Boolean mask.",
        examples=[
            ">>> import grumpy as gr",
            ">>> gr.isin(gr.array([1, 2]), gr.array([2, 3])).to_list()",
            "[False, True]",
        ],
    ),
    "setdiff": doc(
        "Set difference of two 1D leaf arrays.",
        params=["a, b : GrumpyArray", "    Input arrays."],
        returns="GrumpyArray\n    Values in ``a`` not present in ``b``.",
        examples=[
            ">>> import grumpy as gr",
            ">>> gr.setdiff(gr.array([1, 2, 3]), gr.array([2])).to_list()",
            "[1, 3]",
        ],
    ),
    "setunion": doc(
        "Sorted union of two 1D leaf arrays.",
        params=["a, b : GrumpyArray", "    Input arrays."],
        returns="GrumpyArray\n    Union of values.",
        examples=[
            ">>> import grumpy as gr",
            ">>> gr.setunion(gr.array([1, 2]), gr.array([2, 3])).to_list()",
            "[1, 2, 3]",
        ],
    ),
    "setxor": doc(
        "Symmetric difference of two 1D leaf arrays.",
        params=["a, b : GrumpyArray", "    Input arrays."],
        returns="GrumpyArray\n    Values in exactly one input.",
        examples=[
            ">>> import grumpy as gr",
            ">>> gr.setxor(gr.array([1, 2]), gr.array([2, 3])).to_list()",
            "[1, 3]",
        ],
    ),
    "var": doc(
        "Variance along a ragged axis.",
        params=[
            "x : GrumpyArray",
            "    Input array.",
            "dim : int, default 0",
            "    Axis to reduce.",
            "ddof : int, default 0",
            "    Delta degrees of freedom.",
        ],
        returns="GrumpyArray\n    Variance along ``dim``.",
        examples=[">>> import grumpy as gr", ">>> x = gr.array([[1, 2], [3, 4]])", ">>> gr.var(x, dim=1).to_list()"],
    ),
    "std": doc(
        "Standard deviation along a ragged axis.",
        params=["x : GrumpyArray", "dim : int, default 0", "ddof : int, default 0"],
        returns="GrumpyArray\n    Standard deviation along ``dim``.",
        examples=[">>> import grumpy as gr", ">>> gr.std(gr.array([1, 2, 3])).to_list()"],
    ),
    "nanvar": doc(
        "NaN-aware variance along a ragged axis.",
        params=["x : GrumpyArray", "dim : int, default 0", "ddof : int, default 0"],
        returns="GrumpyArray\n    Variance ignoring NaN leaves.",
        examples=[">>> import grumpy as gr", ">>> gr.nanvar(gr.array([1.0, float('nan')])).to_list()"],
    ),
    "nanstd": doc(
        "NaN-aware standard deviation along a ragged axis.",
        params=["x : GrumpyArray", "dim : int, default 0", "ddof : int, default 0"],
        returns="GrumpyArray\n    Standard deviation ignoring NaN leaves.",
        examples=[">>> import grumpy as gr", ">>> gr.nanstd(gr.array([1.0, float('nan')])).to_list()"],
    ),
    "quantile": doc(
        "Quantile along a ragged axis.",
        params=["x : GrumpyArray", "q : float", "dim : int, default 0"],
        returns="GrumpyArray\n    Quantile values.",
        examples=[">>> import grumpy as gr", ">>> gr.quantile(gr.array([1, 2, 3]), 50).to_list()"],
    ),
    "nanquantile": doc(
        "NaN-aware quantile along a ragged axis.",
        params=["x : GrumpyArray", "q : float", "dim : int, default 0"],
        returns="GrumpyArray\n    Quantile ignoring NaN leaves.",
        examples=[">>> import grumpy as gr", ">>> gr.nanquantile(gr.array([1.0, float('nan'), 3.0]), 50).to_list()"],
    ),
    "percentile": doc(
        "Percentile along a ragged axis.",
        params=["x : GrumpyArray", "q : float", "dim : int, default 0"],
        returns="GrumpyArray\n    Percentile values.",
        examples=[">>> import grumpy as gr", ">>> gr.percentile(gr.array([1, 2, 3]), 50).to_list()"],
    ),
    "nanpercentile": doc(
        "NaN-aware percentile along a ragged axis.",
        params=["x : GrumpyArray", "q : float", "dim : int, default 0"],
        returns="GrumpyArray\n    Percentile ignoring NaN leaves.",
        examples=[">>> import grumpy as gr", ">>> gr.nanpercentile(gr.array([1.0, float('nan')]), 50).to_list()"],
    ),
    "median": doc(
        "Median along a ragged axis.",
        params=["x : GrumpyArray", "dim : int, default 0"],
        returns="GrumpyArray\n    Median values.",
        examples=[">>> import grumpy as gr", ">>> gr.median(gr.array([1, 3, 2])).to_list()"],
    ),
    "nanmedian": doc(
        "NaN-aware median along a ragged axis.",
        params=["x : GrumpyArray", "dim : int, default 0"],
        returns="GrumpyArray\n    Median ignoring NaN leaves.",
        examples=[">>> import grumpy as gr", ">>> gr.nanmedian(gr.array([1.0, float('nan'), 3.0])).to_list()"],
    ),
    "bincount": doc(
        "Count occurrences of non-negative integer leaves.",
        params=[
            "x : GrumpyArray",
            "    Non-negative integer indices.",
            "weights : GrumpyArray, optional",
            "    Weights applied to each bin.",
            "minlength : int, default 0",
            "    Minimum length of the output histogram.",
        ],
        returns="GrumpyArray\n    Bin counts.",
        examples=[
            ">>> import grumpy as gr",
            ">>> gr.bincount(gr.array([0, 1, 1])).to_list()",
            "[1, 2]",
        ],
    ),
    "digitize": doc(
        "Bin leaf values into discrete intervals.",
        params=[
            "x : GrumpyArray",
            "    Values to bin.",
            "bins : GrumpyArray",
            "    Monotonic bin edges.",
            "right : bool, default False",
            "    If ``True``, intervals are ``(a[i], a[i+1]]``.",
        ],
        returns="GrumpyArray\n    Bin indices.",
        examples=[
            ">>> import grumpy as gr",
            ">>> gr.digitize(gr.array([0.2, 1.5]), gr.array([0.0, 1.0, 2.0])).to_list()",
        ],
    ),
    "histogram": doc(
        "Compute a histogram of leaf values.",
        params=[
            "x : GrumpyArray",
            "    Input values.",
            "bins : int, default 10",
            "    Number of equal-width bins.",
            "range : tuple[float, float], optional",
            "    ``(min, max)`` range of the histogram.",
            "density : bool, default False",
            "    If ``True``, normalize to form a density.",
            "weights : GrumpyArray, optional",
            "    Per-leaf weights.",
        ],
        returns="tuple[GrumpyArray, GrumpyArray]\n    ``(counts, bin_edges)``.",
        examples=[
            ">>> import grumpy as gr",
            ">>> counts, edges = gr.histogram(gr.array([0.1, 1.2, 1.9]), bins=2)",
        ],
    ),
    "nonzero": doc(
        "Return indices of non-zero / True leaves.",
        params=["x : GrumpyArray", "    Input array."],
        returns="GrumpyArray\n    Index structure of non-zero entries.",
        examples=[">>> import grumpy as gr", ">>> gr.nonzero(gr.array([0, 1, 0])).to_list()"],
    ),
    "search_sorted": doc(
        "Find insertion indices to maintain sorted order.",
        params=[
            "x : GrumpyArray",
            "    Sorted reference array.",
            "v : GrumpyArray",
            "    Values to insert.",
            "right : bool, default False",
            "    Side of the interval to use when values match edges.",
        ],
        returns="GrumpyArray\n    Insertion indices.",
        examples=[
            ">>> import grumpy as gr",
            ">>> gr.search_sorted(gr.array([1, 3]), gr.array([2])).to_list()",
        ],
    ),
    "where": doc(
        "NumPy-like conditional selection or index lookup.",
        params=[
            "cond : GrumpyArray",
            "    Boolean condition.",
            "x, y : GrumpyArray, optional",
            "    Values chosen where ``cond`` is true/false.",
        ],
        returns="GrumpyArray\n    Selected values, or indices when ``x`` and ``y`` are omitted.",
        examples=[
            ">>> import grumpy as gr",
            ">>> c = gr.array([True, False])",
            ">>> gr.where(c, gr.array([1, 2]), gr.array([9, 9])).to_list()",
            "[1, 9]",
        ],
    ),
    "argwhere": doc(
        "Return indices where ``cond`` is true.",
        params=["cond : GrumpyArray", "    Boolean condition."],
        returns="GrumpyArray\n    Index pairs for true entries.",
        examples=[">>> import grumpy as gr", ">>> gr.argwhere(gr.array([True, False]))"],
    ),
    "dot": doc(
        "Dot product / matrix multiply where layouts permit.",
        params=["a, b : GrumpyArray", "    Input arrays."],
        returns="GrumpyArray or scalar layout\n    Dot product result.",
        examples=[">>> import grumpy as gr", ">>> gr.dot(gr.array([1, 2]), gr.array([3, 4])).to_list()"],
    ),
    "inner": doc(
        "Inner product of two arrays.",
        params=["a, b : GrumpyArray", "    Input arrays."],
        returns="GrumpyArray\n    Inner product.",
        examples=[">>> import grumpy as gr", ">>> gr.inner(gr.array([1, 2]), gr.array([3, 4])).to_list()"],
    ),
    "outer": doc(
        "Outer product of two 1D arrays.",
        params=["a, b : GrumpyArray", "    Input vectors."],
        returns="GrumpyArray\n    Outer product matrix as ragged layout.",
        examples=[">>> import grumpy as gr", ">>> gr.outer(gr.array([1, 2]), gr.array([3, 4])).to_list()"],
    ),
    "trace": doc(
        "Sum of diagonal elements for 2D layouts.",
        params=["a : GrumpyArray", "    Input matrix-like array."],
        returns="GrumpyArray\n    Trace value.",
        examples=[">>> import grumpy as gr", ">>> gr.trace(gr.array([[1, 2], [3, 4]])).to_list()"],
    ),
    "norm": doc(
        "Vector or matrix norm.",
        params=["a : GrumpyArray", "    Input array."],
        returns="GrumpyArray\n    Norm value.",
        examples=[">>> import grumpy as gr", ">>> gr.norm(gr.array([3.0, 4.0])).to_list()"],
    ),
    "cross": doc(
        "Cross product of vectors.",
        params=["a, b : GrumpyArray", "    Input 3-vectors."],
        returns="GrumpyArray\n    Cross product.",
        examples=[">>> import grumpy as gr", ">>> gr.cross(gr.array([1, 0, 0]), gr.array([0, 1, 0])).to_list()"],
    ),
    "det": doc(
        "Determinant of a square 2D layout.",
        params=["a : GrumpyArray", "    Input matrix."],
        returns="GrumpyArray\n    Determinant.",
        examples=[">>> import grumpy as gr", ">>> gr.det(gr.array([[1, 2], [3, 4]])).to_list()"],
    ),
    "inv": doc(
        "Matrix inverse for square 2D layouts.",
        params=["a : GrumpyArray", "    Input matrix."],
        returns="GrumpyArray\n    Inverse matrix.",
        examples=[">>> import grumpy as gr", ">>> gr.inv(gr.array([[1, 0], [0, 1]])).to_list()"],
    ),
    "einsum": doc(
        "Einstein summation over array operands.",
        params=[
            "subscripts : str",
            "    Index notation string.",
            "*operands : GrumpyArray",
            "    Input arrays.",
        ],
        returns="GrumpyArray\n    Contraction result.",
        examples=[
            ">>> import grumpy as gr",
            ">>> a = gr.array([[1, 2], [3, 4]])",
            ">>> gr.einsum('ij->i', a).to_list()",
        ],
    ),
    "tensordot": doc(
        "Tensor dot product along the innermost axes.",
        params=[
            "a, b : GrumpyArray",
            "    Input arrays.",
            "axes : int, default 2",
            "    Number of axes to sum over.",
        ],
        returns="GrumpyArray\n    Tensor contraction result.",
        examples=[">>> import grumpy as gr", ">>> gr.tensordot(gr.array([1, 2]), gr.array([3, 4])).to_list()"],
    ),
    "neighbors": doc(
        "Compute kNN or radius neighbors and return graph edges.",
        params=[
            "query : GrumpyArray",
            "    Query points.",
            "data : GrumpyArray",
            "    Candidate points.",
            "k : int, optional",
            "    Number of nearest neighbors.",
            "radius : float, optional",
            "    Radius search threshold.",
            "dim : int, default 0",
            "    Point-cloud dimensionality grouping.",
            "loop : bool, default True",
            "    Include self matches when query and data share storage.",
            "return_distances : bool, default False",
            "    Also return neighbor distances.",
        ],
        returns="GrumpyArray or tuple[GrumpyArray, GrumpyArray]\n    Edge index ``[src, dst]`` and optional distances.",
        examples=[
            ">>> import grumpy as gr",
            ">>> q = gr.array([[0.0, 0.0], [1.0, 1.0]])",
            ">>> edges = gr.neighbors(q, q, k=1, dim=0)",
        ],
    ),
    "dataframe": doc(
        "Create a column-oriented dataframe from a mapping of arrays.",
        params=[
            "mapping : dict[str, array-like]",
            "    Column name to values mapping.",
            "schema : optional",
            "    Optional schema enforcing shared outer shapes.",
        ],
        returns="GrumpyDataFrame\n    New dataframe.",
        examples=[
            ">>> import grumpy as gr",
            ">>> df = gr.dataframe({'id': ['a', 'b'], 'v': [[1], [2, 3]]})",
        ],
    ),
    "save": doc(
        "Save an array or dataframe to a Zarr directory store.",
        params=[
            "obj : GrumpyArray or GrumpyDataFrame",
            "    Object to persist.",
            "path : str",
            "    Output directory path.",
            "chunk_size : int, default 1024",
            "    Leaf chunk size for Zarr buffers.",
        ],
        returns="None",
        examples=[
            ">>> import grumpy as gr",
            ">>> gr.save(gr.array([1, 2, 3]), 'data.gr')",
        ],
    ),
    "load": doc(
        "Load a saved array or dataframe from disk.",
        params=["path : str", "    Path passed to :func:`save`."],
        returns="GrumpyArray or GrumpyDataFrame\n    Loaded object.",
        examples=[
            ">>> import grumpy as gr",
            ">>> gr.save(gr.array([1, 2]), 'data.gr')",
            ">>> gr.load('data.gr').to_list()",
            "[1, 2]",
        ],
    ),
    "open": doc(
        "Open a saved dataframe as a lazy on-disk handle.",
        params=[
            "path : str",
            "    Saved dataset path.",
            "cache : str, default 'chunks'",
            "    I/O cache mode: ``'chunks'``, ``'metadata'``, or ``'none'``.",
            "chunk_budget_mb : int, default 256",
            "    Byte budget for decoded chunk LRU when ``cache='chunks'``.",
        ],
        returns="OpenDataFrame\n    Lazy handle; row/schema indexing materializes subsets.",
        examples=[
            ">>> import grumpy as gr",
            ">>> with gr.open('data.gr') as h:",
            "...     len(h)",
        ],
    ),
    "compile": doc(
        "Compile a batch transform into a fused Rust execution plan.",
        params=[
            "fn : callable",
            "    Function ``fn(batch) -> batch`` with straight-line Python.",
        ],
        returns="CompiledTransform\n    Callable wrapper that runs the compiled plan when possible.",
        examples=[
            ">>> import grumpy as gr",
            ">>> @gr.compile",
            "... def f(batch):",
            "...     return batch * 2",
            "... ",
            ">>> f(gr.array([1, 2])).to_list()",
            "[2, 4]",
        ],
    ),
}
for _name in (
    "sin", "cos", "tan", "exp", "log", "log10", "log2", "sqrt", "abs", "sign",
    "floor", "ceil", "round", "reciprocal", "angle",
    "isnan", "isfinite", "isinf",
    "equal", "not_equal", "less", "less_equal", "greater", "greater_equal",
    "logical_and", "logical_or", "logical_xor", "logical_not",
):
    FUNCTION_DOCS.setdefault(
        _name,
        doc(
            f"Apply :meth:`GrumpyArray.{_name}` to ``x``.",
            params=["x : GrumpyArray", "    Input array."],
            returns="GrumpyArray\n    Result array.",
            examples=[
                ">>> import grumpy as gr",
                f">>> gr.{_name}(gr.array([1, 2]))",
            ],
        ),
    )

# Top-level wrappers that delegate to GrumpyArray methods.