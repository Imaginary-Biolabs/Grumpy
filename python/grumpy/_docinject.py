"""Attach NumPy-style docstrings to Rust extension types and functions."""

from __future__ import annotations

from . import _core
from ._docutil import doc, inject, inject_many


def _same_layout(name: str, verb: str) -> str:
    return doc(
        f"Elementwise {verb}.",
        params=["x : GrumpyArray", "    Input array."],
        returns="GrumpyArray\n    Result with the same ragged layout as ``x``.",
        examples=[
            ">>> import grumpy as gr",
            f">>> x = gr.array([0.0, 1.0])",
            f">>> gr.{name}(x).to_list()",
        ],
    )


def _binary(name: str, verb: str) -> str:
    return doc(
        f"Elementwise {verb} of two arrays with broadcasting.",
        params=[
            "a : GrumpyArray",
            "    Left-hand array.",
            "b : GrumpyArray",
            "    Right-hand array.",
        ],
        returns="GrumpyArray\n    Boolean or numeric result aligned with broadcast rules.",
        examples=[
            ">>> import grumpy as gr",
            ">>> a = gr.array([1, 2, 3])",
            ">>> b = gr.array([2, 2, 2])",
            f">>> a.{name}(b).to_list()",
        ],
    )


def _reduce(name: str, summary: str, *, extra_params: list[str] | None = None) -> str:
    params = [
        "dim : int, default 0",
        "    Ragged axis to reduce along.",
    ]
    if extra_params:
        params.extend(extra_params)
    return doc(
        summary,
        params=params,
        returns="GrumpyArray\n    Reduced array along ``dim``.",
        examples=[
            ">>> import grumpy as gr",
            ">>> x = gr.array([[1, 2], [3, 4, 5]])",
            f">>> x.{name}(dim=1).to_list()",
        ],
    )


def inject_grumpy_array_docs() -> None:
    cls = _core.GrumpyArray
    cls.__doc__ = doc(
        "Ragged nested array backed by a typed layout tree.",
        params=[
            "GrumpyArray objects are created with :func:`grumpy.array` and expose",
            "NumPy-like methods for elementwise math, reductions, sorting, and shape queries.",
        ],
        examples=[
            ">>> import grumpy as gr",
            ">>> x = gr.array([[1, 2], [3]], dtype=gr.int32)",
            ">>> x.shape(dim=1)",
            "2",
        ],
    )

    unary = {
        "sin": "sine",
        "cos": "cosine",
        "tan": "tangent",
        "exp": "exponential",
        "log": "natural logarithm",
        "log10": "base-10 logarithm",
        "log2": "base-2 logarithm",
        "sqrt": "square root",
        "abs": "absolute value",
        "sign": "sign",
        "floor": "floor",
        "ceil": "ceiling",
        "round": "round to nearest integer",
        "reciprocal": "reciprocal",
        "angle": "complex angle (for numeric leaves)",
    }
    inject_many(cls, {k: _same_layout(k, v) for k, v in unary.items()})

    for name, verb in (
        ("isnan", "NaN indicator"),
        ("isfinite", "finite-value indicator"),
        ("isinf", "infinity indicator"),
    ):
        inject(
            cls,
            name,
            doc(
                f"Return a boolean array marking {verb} leaves.",
                params=["x : GrumpyArray", "    Input array."],
                returns="GrumpyArray\n    Boolean mask with the same layout as ``x``.",
                examples=[
                    ">>> import grumpy as gr",
                    ">>> x = gr.array([1.0, float('nan')])",
                    f">>> x.{name}().to_list()",
                ],
            ),
        )

    inject_many(
        cls,
        {
            "equal": _binary("equal", "equality"),
            "not_equal": _binary("not_equal", "inequality"),
            "less": _binary("less", "less-than comparison"),
            "less_equal": _binary("less_equal", "less-than-or-equal comparison"),
            "greater": _binary("greater", "greater-than comparison"),
            "greater_equal": _binary("greater_equal", "greater-than-or-equal comparison"),
            "logical_and": _binary("logical_and", "logical AND"),
            "logical_or": _binary("logical_or", "logical OR"),
            "logical_xor": _binary("logical_xor", "logical XOR"),
        },
    )
    inject(
        cls,
        "logical_not",
        doc(
            "Elementwise logical NOT.",
            params=["x : GrumpyArray", "    Input boolean array."],
            returns="GrumpyArray\n    Inverted boolean mask.",
            examples=[
                ">>> import grumpy as gr",
                ">>> x = gr.array([True, False])",
                ">>> x.logical_not().to_list()",
            ],
        ),
    )

    inject_many(
        cls,
        {
            "sum": _reduce("sum", "Sum along a ragged axis.", extra_params=["dim : int, optional", "    Axis to sum. Omit to sum all leaves to a scalar layout."]),
            "mean": _reduce("mean", "Mean along a ragged axis."),
            "min": _reduce("min", "Minimum along a ragged axis."),
            "max": _reduce("max", "Maximum along a ragged axis."),
            "ptp": _reduce("ptp", "Peak-to-peak (max - min) along a ragged axis."),
            "var": _reduce("var", "Variance along a ragged axis.", extra_params=["ddof : int, default 0", "    Delta degrees of freedom."]),
            "std": _reduce("std", "Standard deviation along a ragged axis.", extra_params=["ddof : int, default 0", "    Delta degrees of freedom."]),
            "nanvar": _reduce("nanvar", "Variance along a ragged axis, ignoring NaN leaves.", extra_params=["ddof : int, default 0", "    Delta degrees of freedom."]),
            "nanstd": _reduce("nanstd", "Standard deviation along a ragged axis, ignoring NaN leaves.", extra_params=["ddof : int, default 0", "    Delta degrees of freedom."]),
            "median": _reduce("median", "Median along a ragged axis."),
            "nanmedian": _reduce("nanmedian", "Median along a ragged axis, ignoring NaN leaves."),
        },
    )

    for name, label in (
        ("quantile", "quantile"),
        ("nanquantile", "NaN-aware quantile"),
        ("percentile", "percentile"),
        ("nanpercentile", "NaN-aware percentile"),
    ):
        inject(
            cls,
            name,
            doc(
                f"Compute the {label} along a ragged axis.",
                params=[
                    "q : float",
                    "    Quantile in ``[0, 100]``.",
                    "dim : int, default 0",
                    "    Axis to reduce along.",
                ],
                returns="GrumpyArray\n    Reduced array along ``dim``.",
                examples=[
                    ">>> import grumpy as gr",
                    ">>> x = gr.array([[1, 2], [3, 4, 5]])",
                    f">>> x.{name}(50, dim=1).to_list()",
                ],
            ),
        )

    inject_many(
        cls,
        {
            "shape": doc(
                "Return the length along an outer ragged axis.",
                params=["dim : int, default 0", "    Axis index."],
                returns="int\n    Number of elements along ``dim``.",
                examples=[
                    ">>> import grumpy as gr",
                    ">>> x = gr.array([[1, 2], [3]])",
                    ">>> x.shape(dim=1)",
                    "2",
                ],
            ),
            "nshape": doc(
                "Return per-row lengths along an axis (including null slots).",
                params=["dim : int, default 0", "    Axis index."],
                returns="GrumpyArray\n    Lengths encoded as an array.",
                examples=[
                    ">>> import grumpy as gr",
                    ">>> x = gr.array([[1, 2], [3]])",
                    ">>> x.nshape(dim=1).to_list()",
                ],
            ),
            "nanshape": doc(
                "Return per-row non-null counts along an axis.",
                params=["dim : int, default 0", "    Axis index."],
                returns="GrumpyArray\n    Non-null counts encoded as an array.",
                examples=[
                    ">>> import grumpy as gr",
                    ">>> x = gr.array([[1, None], [3]])",
                    ">>> x.nanshape(dim=1).to_list()",
                ],
            ),
            "dtype": doc(
                "Return the array dtype.",
                returns="DType\n    Leaf dtype of the array.",
                examples=[
                    ">>> import grumpy as gr",
                    ">>> gr.array([1, 2], dtype=gr.int32).dtype.name()",
                    "'int32'",
                ],
            ),
            "copy": doc(
                "Return a deep copy of the array buffers.",
                returns="GrumpyArray\n    Independent copy.",
                examples=[
                    ">>> import grumpy as gr",
                    ">>> x = gr.array([1, 2])",
                    ">>> y = x.copy()",
                ],
            ),
            "astype": doc(
                "Cast leaves to another dtype (layout-preserving).",
                params=[
                    "dtype : DType",
                    "    Target dtype.",
                    "casting : str, optional",
                    "    ``'safe'`` (default), ``'same_kind'``, or ``'unsafe'``.",
                ],
                returns="GrumpyArray\n    Array with converted leaves.",
                examples=[
                    ">>> import grumpy as gr",
                    ">>> x = gr.array([1, 2], dtype=gr.int32)",
                    ">>> x.astype(gr.float64).dtype.name",
                    "'float64'",
                ],
            ),
            "to_list": doc(
                "Materialize the array as nested Python lists.",
                returns="list\n    Nested Python structure mirroring the layout.",
                examples=[
                    ">>> import grumpy as gr",
                    ">>> gr.array([[1, 2], [3]]).to_list()",
                    "[[1, 2], [3]]",
                ],
            ),
            "to_numpy": doc(
                "Convert to NumPy when the layout is rectangular.",
                returns="numpy.ndarray\n    Dense NumPy array.",
                examples=[
                    ">>> import grumpy as gr",
                    ">>> x = gr.array([[1, 2], [3, 4]])",
                    ">>> x.to_numpy().shape",
                    "(2, 2)",
                ],
            ),
            "flatten": doc(
                "Flatten nested lists to a 1D leaf array.",
                params=[
                    "dim : int or sequence of int, optional",
                    "    Axis or axes to flatten. Omit to flatten all nested levels.",
                    "but : int or sequence of int, optional",
                    "    Axes to exclude from flattening (Awkward-style ``but=``).",
                ],
                returns="GrumpyArray\n    One-dimensional view of selected leaf values.",
                examples=[
                    ">>> import grumpy as gr",
                    ">>> gr.array([[1, 2], [3]]).flatten().to_list()",
                    "[1, 2, 3]",
                ],
            ),
            "unflatten": doc(
                "Restore nested structure from a flattened leaf array.",
                params=[
                    "sizes : sequence or GrumpyArray",
                    "    Per-row lengths along ``dim``.",
                    "dim : int, default 0",
                    "    Axis at which to rebuild nesting.",
                ],
                returns="GrumpyArray\n    Array with nesting restored along ``dim``.",
                examples=[
                    ">>> import grumpy as gr",
                    ">>> x = gr.array([[1, 2], [3]])",
                    ">>> x.flatten().unflatten([2, 1]).to_list()",
                    "[[1, 2], [3]]",
                ],
            ),
        },
    )

    for name in ("sort", "argsort", "argmax", "argmin", "nanargmax", "nanargmin"):
        inject(
            cls,
            name,
            doc(
                f"NumPy-like ``{name}`` along an optional ragged axis.",
                params=["dim : int, optional", "    Axis to operate on."],
                returns="GrumpyArray or int\n    Sorted values, indices, or partitions depending on the method.",
                examples=[
                    ">>> import grumpy as gr",
                    ">>> x = gr.array([[3, 1], [2]])",
                    f">>> x.{name}(dim=1)",
                ],
            ),
        )

    for name in ("partition", "argpartition"):
        inject(
            cls,
            name,
            doc(
                f"Partial sort / index selection via ``{name}`` along an optional axis.",
                params=[
                    "kth : int",
                    "    Partition index (NumPy ``kth`` argument).",
                    "dim : int, optional",
                    "    Axis to operate on.",
                ],
                returns="GrumpyArray\n    Partially sorted values or index array.",
                examples=[
                    ">>> import grumpy as gr",
                    ">>> x = gr.array([[3, 1, 2], [2]])",
                    f">>> x.{name}(1, dim=1)",
                ],
            ),
        )

    for name in ("mod_", "remainder"):
        inject(
            cls,
            name,
            doc(
                "Elementwise modulo with broadcasting.",
                params=[
                    "other : GrumpyArray",
                    "    Divisor array.",
                ],
                returns="GrumpyArray\n    Remainder with the same layout rules as elementwise ops.",
                examples=[
                    ">>> import grumpy as gr",
                    ">>> a = gr.array([5, 7])",
                    ">>> b = gr.array([2, 3])",
                    f">>> a.{name}(b).to_list()",
                ],
            ),
        )


def inject_grumpy_array_dunder_docs() -> None:
    cls = _core.GrumpyArray
    inject_many(
        cls,
        {
            "__len__": doc(
                "Return the number of elements along axis 0.",
                returns="int\n    Outer axis length.",
                examples=[
                    ">>> import grumpy as gr",
                    ">>> len(gr.array([[1, 2], [3]]))",
                    "2",
                ],
            ),
            "__repr__": doc(
                "Return a concise debug representation.",
                returns="str\n    Summary including dtype and nested data preview.",
                examples=[
                    ">>> import grumpy as gr",
                    ">>> repr(gr.array([1, 2]))",
                    "'GrumpyArray(dtype=int64, data=[1, 2])'",
                ],
            ),
            "__getitem__": doc(
                "Index into the array with integers, slices, or boolean masks.",
                params=[
                    "index : int, slice, or GrumpyArray",
                    "    Row/column selection along ragged axes.",
                ],
                returns="GrumpyArray or scalar leaf\n    Selected sub-array or value.",
                examples=[
                    ">>> import grumpy as gr",
                    ">>> x = gr.array([[1, 2], [3]])",
                    ">>> x[0].to_list()",
                    "[1, 2]",
                ],
            ),
            "__setitem__": doc(
                "Assign values into a selected region of the array.",
                params=[
                    "index : int, slice, or GrumpyArray",
                    "    Target region.",
                    "value : scalar or GrumpyArray",
                    "    Value(s) to write.",
                ],
                returns="None",
                examples=[
                    ">>> import grumpy as gr",
                    ">>> x = gr.array([[1, 2], [3]])",
                    ">>> x[0] = gr.array([9, 8])",
                ],
            ),
            "__add__": doc(
                "Elementwise addition with broadcasting.",
                params=["other : GrumpyArray or scalar", "    Right-hand operand."],
                returns="GrumpyArray\n    Sum.",
                examples=[
                    ">>> import grumpy as gr",
                    ">>> (gr.array([1, 2]) + gr.array([3, 4])).to_list()",
                    "[4, 6]",
                ],
            ),
            "__sub__": doc(
                "Elementwise subtraction with broadcasting.",
                params=["other : GrumpyArray or scalar", "    Right-hand operand."],
                returns="GrumpyArray\n    Difference.",
                examples=[
                    ">>> import grumpy as gr",
                    ">>> (gr.array([3, 4]) - gr.array([1, 2])).to_list()",
                    "[2, 2]",
                ],
            ),
            "__mul__": doc(
                "Elementwise multiplication with broadcasting.",
                params=["other : GrumpyArray or scalar", "    Right-hand operand."],
                returns="GrumpyArray\n    Product.",
                examples=[
                    ">>> import grumpy as gr",
                    ">>> (gr.array([1, 2]) * 2).to_list()",
                    "[2, 4]",
                ],
            ),
            "__truediv__": doc(
                "Elementwise true division with broadcasting.",
                params=["other : GrumpyArray or scalar", "    Divisor."],
                returns="GrumpyArray\n    Quotient (promotes integers to float when needed).",
                examples=[
                    ">>> import grumpy as gr",
                    ">>> (gr.array([1, 2]) / 2).to_list()",
                    "[0.5, 1.0]",
                ],
            ),
            "__mod__": doc(
                "Elementwise modulo with broadcasting.",
                params=["other : GrumpyArray or scalar", "    Divisor."],
                returns="GrumpyArray\n    Remainder.",
                examples=[
                    ">>> import grumpy as gr",
                    ">>> (gr.array([5, 7]) % 2).to_list()",
                    "[1, 1]",
                ],
            ),
        },
    )


def inject_dtype_docs() -> None:
    cls = _core.DType
    cls.__doc__ = doc(
        "Strongly typed leaf dtype descriptor for Grumpy arrays.",
        returns="Use factory methods such as ``DType.int32()`` or module singletons like ``gr.int32``.",
        examples=[
            ">>> import grumpy as gr",
            ">>> gr.int32.name()",
            "'int32'",
        ],
    )
    for name in (
        "int8", "int16", "int32", "int64",
        "uint8", "uint16", "uint32", "uint64",
        "float16", "float32", "float64",
        "bool_", "char", "string",
    ):
        inject(
            cls,
            name,
            doc(
                f"Return the ``{name}`` dtype singleton.",
                returns="DType\n    Dtype instance.",
                examples=[
                    ">>> import grumpy as gr",
                    f">>> gr.DType.{name}().name()",
                    f"'{name.replace('_', '') if name != 'bool_' else 'bool'}'",
                ],
            ),
        )
    inject(
        cls,
        "name",
        doc(
            "Return the canonical dtype name.",
            returns="str\n    Name such as ``'int32'`` or ``'string'``.",
            examples=[
                ">>> import grumpy as gr",
                ">>> gr.int32.name()",
                "'int32'",
            ],
        ),
    )
    inject(
        cls,
        "__repr__",
        doc(
            "Return a debug representation of the dtype.",
            returns="str\n    Canonical dtype name.",
            examples=[
                ">>> import grumpy as gr",
                ">>> repr(gr.int32)",
                "'DType(int32)'",
            ],
        ),
    )


def inject_dataframe_docs() -> None:
    cls = _core.GrumpyDataFrame
    cls.__doc__ = doc(
        "Column-oriented container of named :class:`GrumpyArray` columns.",
        examples=[
            ">>> import grumpy as gr",
            ">>> df = gr.dataframe({'a': [1, 2], 'b': [3, 4]})",
            ">>> df.to_dict()['a'].to_list()",
            "[1, 2]",
        ],
    )
    inject(
        cls,
        "to_dict",
        doc(
            "Return columns as a Python dict of name → array.",
            returns="dict[str, GrumpyArray]\n    Column mapping.",
            examples=[
                ">>> import grumpy as gr",
                ">>> df = gr.dataframe({'x': [1]})",
                ">>> list(df.to_dict())",
                "['x']",
            ],
        ),
    )
    inject(
        cls,
        "max",
        doc(
            "Reduce all columns with ``max`` (dataframe-level helper).",
            returns="GrumpyArray\n    Reduced result.",
            examples=[
                ">>> import grumpy as gr",
                ">>> df = gr.dataframe({'a': [1, 3], 'b': [2, 4]})",
                ">>> df.max()",
            ],
        ),
    )
    inject_many(
        cls,
        {
            "__len__": doc(
                "Return the number of rows in the dataframe.",
                returns="int\n    Row count.",
                examples=[
                    ">>> import grumpy as gr",
                    ">>> len(gr.dataframe({'a': [1, 2, 3]}))",
                    "3",
                ],
            ),
            "__repr__": doc(
                "Return a concise debug representation.",
                returns="str\n    Summary listing column names.",
                examples=[
                    ">>> import grumpy as gr",
                    ">>> repr(gr.dataframe({'a': [1], 'b': [2]}))",
                    "'grumpy.dataframe(a, b)'",
                ],
            ),
            "__getitem__": doc(
                "Select columns, rows, or schema levels.",
                params=[
                    "key : str, tuple[str, ...], int, slice, or boolean mask",
                    "    Column name(s) or row index expression.",
                ],
                returns="GrumpyDataFrame or GrumpyArray\n    Sub-frame or flattened column array.",
                examples=[
                    ">>> import grumpy as gr",
                    ">>> df = gr.dataframe({'a': [1, 2], 'b': [3, 4]})",
                    ">>> df['a'].to_list()",
                    "[1, 2]",
                ],
            ),
            "__setitem__": doc(
                "Assign or replace a column by name.",
                params=[
                    "key : str",
                    "    Column name.",
                    "value : array-like or GrumpyArray",
                    "    Column values.",
                ],
                returns="None",
                examples=[
                    ">>> import grumpy as gr",
                    ">>> df = gr.dataframe({'a': [1]})",
                    ">>> df['b'] = gr.array([2])",
                ],
            ),
            "__getattr__": doc(
                "Access schema levels or columns via attribute syntax.",
                params=["name : str", "    Schema level or column name."],
                returns="GrumpyDataFrame accessor or GrumpyArray\n    Nested accessor or flattened column.",
                examples=[
                    ">>> import grumpy as gr",
                    ">>> df = gr.dataframe({'x': [[1], [2, 3]]})",
                    ">>> df.x.to_list()",
                    "[[1], [2, 3]]",
                ],
            ),
        },
    )


def inject_compiled_plan_docs() -> None:
    cls = _core.CompiledPlan
    cls.__doc__ = doc(
        "Fused Rust execution plan built by :func:`grumpy.compile` or ``Stream.apply(compile=...)``.",
        examples=[
            ">>> import grumpy as gr",
            ">>> @gr.compile",
            "... def f(batch):",
            "...     return batch * 2",
            "... ",
            ">>> f.is_compiled",
            "True",
        ],
    )
    inject(
        cls,
        "run",
        doc(
            "Execute the plan on a batch (GIL released for array batches).",
            params=["batch : GrumpyArray or GrumpyDataFrame", "    Input batch."],
            returns="GrumpyArray or GrumpyDataFrame\n    Transformed batch.",
            examples=[
                ">>> import grumpy as gr",
                ">>> @gr.compile",
                "... def double(batch):",
                "...     return batch * 2",
                "... ",
                ">>> double(gr.array([1, 2])).to_list()",
                "[2, 4]",
            ],
        ),
    )
    inject(
        cls,
        "__repr__",
        doc(
            "Return a debug summary of the compiled plan.",
            returns="str\n    Opcode count summary.",
            examples=[
                ">>> import grumpy as gr",
                ">>> @gr.compile",
                "... def f(b): return b * 2",
                "... ",
                ">>> repr(f._plan)",
            ],
        ),
    )


def inject_core_function_docs() -> None:
    inject_many(
        _core,
        {
            "stored_len": doc(
                "Return axis-0 length from on-disk metadata without loading leaves.",
                params=["path : str", "    Path passed to :func:`grumpy.save`."],
                returns="int\n    Number of rows along axis 0.",
                examples=[
                    ">>> import grumpy as gr",
                    ">>> gr.save(gr.array([1, 2, 3]), 'tmp.gr')",
                    ">>> gr._core.stored_len('tmp.gr')",
                    "3",
                ],
            ),
            "load_slice": doc(
                "Load an axis-0 slice from a saved dataset.",
                params=[
                    "path : str",
                    "    Saved dataset path.",
                    "start : int",
                    "    Inclusive start index.",
                    "stop : int",
                    "    Exclusive stop index.",
                ],
                returns="GrumpyArray or GrumpyDataFrame\n    Batch covering ``[start, stop)``.",
                examples=[
                    ">>> import grumpy as gr",
                    ">>> gr.save(gr.array([1, 2, 3]), 'tmp.gr')",
                    ">>> gr._core.load_slice('tmp.gr', 0, 2).to_list()",
                    "[1, 2]",
                ],
            ),
            "compiled_stream_apply": doc(
                "Apply a fused compiled plan over streaming axis-0 batches in Rust.",
                params=[
                    "path : str",
                    "    Saved dataset path.",
                    "batch_size : int",
                    "    Batch size.",
                    "drop_last : bool",
                    "    Whether to drop the final partial batch.",
                    "cpu : int",
                    "    Rayon worker count.",
                    "prefetch : int",
                    "    Maximum in-flight batches.",
                    "ops : list[dict]",
                    "    Fused opcode specification.",
                ],
                returns="iterator\n    Iterator of transformed batches.",
                examples=[
                    ">>> import grumpy as gr",
                    ">>> st = gr.stream('data.gr', batch_size=32)",
                    ">>> for batch in st.apply(lambda b: b * 2, cpu=4, compile='auto'):",
                    "...     pass",
                ],
            ),
        },
    )


def inject_all() -> None:
    inject_grumpy_array_docs()
    inject_grumpy_array_dunder_docs()
    inject_dtype_docs()
    inject_dataframe_docs()
    inject_compiled_plan_docs()
    inject_core_function_docs()
