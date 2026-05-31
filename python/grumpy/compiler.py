"""Compile restricted batch transforms into fused Rust execution plans.

The compiler analyzes straight-line Python functions (typically ``def f(batch): ...``)
and builds :class:`~grumpy._core.CompiledPlan` opcode lists for use with
:meth:`~grumpy.stream.Stream.apply` or the :func:`compile` decorator.

Known limitations
-----------------
- No control flow (``if``/``for``/``try``), no imports, single ``batch`` parameter.
- ``UnionScalarList`` layouts are not compilable.
- Rust scheduling supports only a fixed opcode set (see ``stream.py``).
"""

from __future__ import annotations

import ast
import inspect
import linecache
import textwrap
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Optional

from . import _core


@dataclass
class _CompileResult:
    plan: Optional["_core.CompiledPlan"]
    error: Optional[str]


class CompiledTransform:
    """
    Callable wrapper that runs a compiled Rust plan when possible.

    Instances are returned by :func:`compile` and used internally by
    :meth:`~grumpy.stream.Stream.apply`.

    Attributes
    ----------
    is_compiled : bool
        Whether a Rust :class:`~grumpy._core.CompiledPlan` was built.
    compile_error : str or None
        Compilation failure message when ``is_compiled`` is ``False``.

    Examples
    --------
    >>> import grumpy as gr
    >>> @gr.compile
    ... def scale(batch):
    ...     return batch * 2
    ...
    >>> scale.is_compiled
    True
    >>> scale(gr.array([1, 2])).to_list()
    [2, 4]
    """

    def __init__(self, fn: Callable[[Any], Any], result: _CompileResult):
        self._fn = fn
        self._plan = result.plan
        self._compile_error = result.error
        self._warned = False

        # Preserve metadata reasonably.
        self.__name__ = getattr(fn, "__name__", "compiled_transform")
        self.__qualname__ = getattr(fn, "__qualname__", self.__name__)
        self.__doc__ = getattr(fn, "__doc__", None)

    @property
    def is_compiled(self) -> bool:
        """
        Return ``True`` when a Rust :class:`~grumpy._core.CompiledPlan` was built.

        Returns
        -------
        bool
            Compilation success flag.

        Examples
        --------
        >>> import grumpy as gr
        >>> @gr.compile
        ... def f(b): return b
        ...
        >>> f.is_compiled
        True
        """
        return self._plan is not None

    @property
    def compile_error(self) -> Optional[str]:
        """
        Return the compilation error message, or ``None`` on success.

        Returns
        -------
        str or None
            Error text when compilation failed.

        Examples
        --------
        >>> import grumpy as gr
        >>> @gr.compile
        ... def ok(b): return b * 2
        ...
        >>> ok.compile_error is None
        True
        """
        return self._compile_error

    def __call__(self, batch):
        """
        Run the compiled plan or fall back to the original Python function.

        Parameters
        ----------
        batch : GrumpyArray or GrumpyDataFrame
            Input batch.

        Returns
        -------
        GrumpyArray or GrumpyDataFrame
            Transformed batch.

        Examples
        --------
        >>> import grumpy as gr
        >>> @gr.compile
        ... def double(batch):
        ...     return batch * 2
        ...
        >>> double(gr.array([1, 2])).to_list()
        [2, 4]
        """
        if self._plan is not None:
            return self._plan.run(batch)
        if (not self._warned) and self._compile_error:
            warnings.warn(
                f"Stream.apply(compile=True): '{self.__qualname__}' could not be compiled; falling back to Python.\n"
                f"Reason: {self._compile_error}",
                category=UserWarning,
                stacklevel=2,
            )
            self._warned = True
        return self._fn(batch)


def compile(fn: Callable[[Any], Any]) -> CompiledTransform:
    """
    Compile a restricted batch transform into a Rust execution plan.

    Parameters
    ----------
    fn : callable
        Function ``fn(batch) -> batch`` with straight-line Python only.

    Returns
    -------
    CompiledTransform
        Callable wrapper that executes the plan when compilation succeeds.

    Examples
    --------
    >>> import grumpy as gr
    >>> @gr.compile
    ... def scale(batch):
    ...     batch = batch * 2
    ...     return batch
    ...
    >>> scale(gr.array([1, 2])).to_list()
    [2, 4]
    """
    res = _try_compile(fn)
    return CompiledTransform(fn, res)

def compile_pipeline(fns: list[Callable[[Any], Any]]) -> Callable[[Any], Any]:
    """
    Compile and fuse a sequence of batch transforms.

    Parameters
    ----------
    fns : list[callable]
        Transform functions applied in order.

    Returns
    -------
    callable
        Single callable that runs fused compiled segments and Python fallbacks.

    Examples
    --------
    >>> import grumpy as gr
    >>> def a(b): return b * 2
    >>> def b(b): return b + 1
    >>> run = gr.compiler.compile_pipeline([a, b])
    >>> run(gr.array([1])).to_list()
    [3]
    """

    segments: list[Callable[[Any], Any]] = []

    pending_ops: list[dict[str, Any]] = []

    def flush_ops():
        nonlocal pending_ops
        if not pending_ops:
            return
        plan = _core.CompiledPlan(_fuse_elementwise_ops(pending_ops))
        pending_ops = []

        def run_plan(x):
            return plan.run(x)

        segments.append(run_plan)

    for fn in fns:
        ops = _try_compile_to_ops(fn)
        if ops is not None:
            pending_ops.extend(ops)
            continue

        # Not compilable: flush current fused segment and append python fallback with warning.
        flush_ops()
        segments.append(CompiledTransform(fn, _try_compile(fn)))

    flush_ops()

    def run_all(x):
        for seg in segments:
            x = seg(x)
        return x

    return run_all


@dataclass(frozen=True)
class PipelineInfo:
    """
    Result of :func:`compile_pipeline_info`.

    Attributes
    ----------
    run_all : callable
        Callable that applies the compiled and Python fallback segments.
    fully_compiled : bool
        ``True`` when every transform compiled into fused Rust ops.
    fused_ops : list[dict] or None
        Fused opcode list when ``fully_compiled`` is ``True``.

    Examples
    --------
    >>> import grumpy as gr
    >>> @gr.compile
    ... def f(batch):
    ...     return batch * 2
    ...
    >>> info = gr.compiler.compile_pipeline_info([f])
    >>> info.fully_compiled
    True
    """

    run_all: Callable[[Any], Any]
    fully_compiled: bool
    fused_ops: Optional[list[dict[str, Any]]]


def compile_pipeline_info(fns: list[Callable[[Any], Any]]) -> PipelineInfo:
    """
    Compile a pipeline and report fusion metadata.

    Parameters
    ----------
    fns : list[callable]
        Transform functions applied in order.

    Returns
    -------
    PipelineInfo
        ``run_all`` callable, ``fully_compiled`` flag, and optional fused op list.

    Examples
    --------
    >>> import grumpy as gr
    >>> @gr.compile
    ... def f(batch):
    ...     return batch * 2
    ...
    >>> info = gr.compiler.compile_pipeline_info([f])
    >>> info.fully_compiled
    True
    """
    segments: list[Callable[[Any], Any]] = []
    pending_ops: list[dict[str, Any]] = []
    saw_python_fallback = False

    def flush_ops() -> Optional[list[dict[str, Any]]]:
        nonlocal pending_ops
        if not pending_ops:
            return None
        ops = pending_ops
        plan = _core.CompiledPlan(ops)
        pending_ops = []

        def run_plan(x):
            return plan.run(x)

        segments.append(run_plan)
        return ops

    fused_single_ops: Optional[list[dict[str, Any]]] = None

    for fn in fns:
        ops = _try_compile_to_ops(fn)
        if ops is not None:
            pending_ops.extend(ops)
            continue
        # Not compilable.
        saw_python_fallback = True
        flush_ops()
        segments.append(CompiledTransform(fn, _try_compile(fn)))

    last = flush_ops()
    if (not saw_python_fallback) and len(segments) == 1:
        # Fully compiled into one plan.
        fused_single_ops = last or []

    def run_all(x):
        for seg in segments:
            x = seg(x)
        return x

    return PipelineInfo(
        run_all=run_all,
        fully_compiled=(not saw_python_fallback) and len(segments) == 1,
        fused_ops=fused_single_ops,
    )

def _try_compile_to_ops(fn: Callable[[Any], Any]) -> Optional[list[dict[str, Any]]]:
    """
    Like _try_compile but returns raw op dicts for fusion. Returns None on failure.
    """
    src = _get_source(fn)
    if src is None:
        return None
    try:
        src = textwrap.dedent(src)
        mod = ast.parse(src)
    except Exception:
        return None

    fndef = None
    for n in mod.body:
        if isinstance(n, ast.FunctionDef):
            fndef = n
            break
    if fndef is None or len(fndef.args.args) != 1 or fndef.args.args[0].arg != "batch":
        return None

    ops: list[dict[str, Any]] = []
    cur_name = "batch"
    body = list(fndef.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) and isinstance(body[0].value.value, str):
        body = body[1:]
    for stmt in body:
        if isinstance(stmt, ast.Return):
            break
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Attribute):
            target_chain = _parse_batch_attr_chain(stmt.targets[0], cur_name)
            if target_chain is None or len(target_chain) < 2:
                return None
            level0, col_out = target_chain[0], target_chain[-1]
            rhs_ops = _compile_df_rhs(stmt.value, cur_name)
            if rhs_ops is None:
                return None
            ops.extend(rhs_ops)
            ops.append({"op": "df_set", "level0": level0, "col": col_out})
            continue
        if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
            return None
        if stmt.targets[0].id != cur_name:
            return None
        op = _compile_expr(stmt.value, cur_name)
        if op is None:
            return None
        ops.append(op)
    return ops or None


def _try_compile(fn: Callable[[Any], Any]) -> _CompileResult:
    src = _get_source(fn)
    if src is None:
        return _CompileResult(None, "Cannot get source for function.")
    try:
        src = textwrap.dedent(src)
        mod = ast.parse(src)
    except Exception as e:
        return _CompileResult(None, f"Cannot parse source to AST ({e!r}).")

    fndef = None
    for n in mod.body:
        if isinstance(n, ast.FunctionDef):
            fndef = n
            break
    if fndef is None:
        return _CompileResult(None, "Expected a Python function definition.")

    if len(fndef.args.args) != 1:
        return _CompileResult(None, "Only single-argument transforms are supported (fn(batch)).")
    arg_name = fndef.args.args[0].arg
    if arg_name != "batch":
        return _CompileResult(None, "Transform argument must be named 'batch' for now.")

    ops: list[dict[str, Any]] = []
    cur_name = "batch"

    body = list(fndef.body)
    # Drop docstring expr.
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) and isinstance(body[0].value.value, str):
        body = body[1:]

    if not body:
        return _CompileResult(None, "Empty function body.")

    for stmt in body:
        if isinstance(stmt, ast.Return):
            if not isinstance(stmt.value, ast.Name) or stmt.value.id != cur_name:
                return _CompileResult(None, "Return must be `return batch` (or the current batch variable).")
            # done
            break

        # Dot-notation dataframe assignment: batch.<level>.<col> = <expr>
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Attribute):
            target_chain = _parse_batch_attr_chain(stmt.targets[0], cur_name)
            if target_chain is None or len(target_chain) < 2:
                return _CompileResult(None, "Only assignments like `batch.<level>.<col> = ...` are supported for dataframes.")
            level0, col_out = target_chain[0], target_chain[-1]
            rhs_ops = _compile_df_rhs(stmt.value, cur_name)
            if rhs_ops is None:
                return _CompileResult(None, "Unsupported RHS for dataframe assignment.")
            ops.extend(rhs_ops)
            ops.append({"op": "df_set", "level0": level0, "col": col_out})
            continue

        if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
            return _CompileResult(None, "Only simple assignments like `batch = ...` are supported.")

        target = stmt.targets[0].id
        if target != cur_name:
            return _CompileResult(None, "Only rebinding the batch variable is supported (use `batch = ...`).")

        expr = stmt.value
        op = _compile_expr(expr, cur_name)
        if op is None:
            return _CompileResult(None, "Unsupported expression in assignment.")
        ops.append(op)

    if not ops:
        return _CompileResult(None, "No compilable operations found.")

    ops = _fuse_elementwise_ops(ops)

    try:
        plan = _core.CompiledPlan(ops)
    except Exception as e:
        return _CompileResult(None, f"Failed to build compiled plan ({e!r}).")
    return _CompileResult(plan, None)


def _compile_expr(expr: ast.AST, cur_name: str) -> Optional[dict[str, Any]]:
    # batch <binop> scalar
    if isinstance(expr, ast.BinOp):
        if not (isinstance(expr.left, ast.Name) and expr.left.id == cur_name):
            return None
        scalar = _const_number(expr.right)
        if scalar is None:
            return None
        if isinstance(expr.op, ast.Add):
            return {"op": "add_scalar", "value": float(scalar), "is_int": isinstance(scalar, int)}
        if isinstance(expr.op, ast.Sub):
            return {"op": "sub_scalar", "value": float(scalar), "is_int": isinstance(scalar, int)}
        if isinstance(expr.op, ast.Mult):
            return {"op": "mul_scalar", "value": float(scalar), "is_int": isinstance(scalar, int)}
        if isinstance(expr.op, ast.Div):
            return {"op": "div_scalar", "value": float(scalar), "is_int": isinstance(scalar, int)}
        if isinstance(expr.op, ast.Mod):
            return {"op": "mod_scalar", "value": float(scalar), "is_int": isinstance(scalar, int)}
        return None

    # gr.neighbors(batch, batch, k=..., dim=..., loop=...)
    if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute):
        if isinstance(expr.func.value, ast.Name) and expr.func.value.id in ("gr", "grumpy") and expr.func.attr == "neighbors":
            if len(expr.args) < 2:
                return None
            if not (isinstance(expr.args[0], ast.Name) and expr.args[0].id == cur_name):
                return None
            if not (isinstance(expr.args[1], ast.Name) and expr.args[1].id == cur_name):
                return None
            kw = {k.arg: k.value for k in expr.keywords if k.arg is not None}
            if "k" not in kw or "radius" in kw:
                # MVP: only kNN
                return None
            k = _const_int(kw["k"])
            dim = _const_int(kw.get("dim", ast.Constant(value=0)))
            loopv = _const_bool(kw.get("loop", kw.get("loop_", ast.Constant(value=True))))
            if k is None or dim is None or loopv is None:
                return None
            return {"op": "neighbors_knn_self", "k": int(k), "dim": int(dim), "loop": bool(loopv)}

    # batch.sum(dim=...) or batch.sum(), batch.mean(dim=...), ...
    if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute):
        if isinstance(expr.func.value, ast.Name) and expr.func.value.id == cur_name:
            red = expr.func.attr
            if red in ("sum", "mean", "min", "max", "ptp"):
                dim = _call_dim(expr)
                if red == "sum" and dim is None:
                    return {"op": "reduce", "reduce": "sum"}
                if dim is None:
                    return None
                return {"op": "reduce", "reduce": red, "dim": int(dim)}

    return None


def _fuse_elementwise_ops(ops: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fuse common op sequences into single kernels."""
    fused: list[dict[str, Any]] = []
    i = 0
    while i < len(ops):
        if (
            i + 1 < len(ops)
            and ops[i].get("op") == "mul_scalar"
            and ops[i + 1].get("op") == "reduce"
            and ops[i + 1].get("reduce") == "sum"
            and "dim" not in ops[i + 1]
        ):
            fused.append(
                {
                    "op": "mul_scalar_sum_all",
                    "value": ops[i]["value"],
                    "is_int": ops[i]["is_int"],
                }
            )
            i += 2
            continue
        fused.append(ops[i])
        i += 1
    return fused


def _parse_batch_attr_chain(node: ast.AST, cur_name: str) -> Optional[list[str]]:
    # Supports batch.residue.atom_pos etc.
    out: list[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        out.append(cur.attr)
        cur = cur.value
    if not (isinstance(cur, ast.Name) and cur.id == cur_name):
        return None
    out.reverse()
    return out


def _compile_df_rhs(expr: ast.AST, cur_name: str) -> Optional[list[dict[str, Any]]]:
    # RHS can be:
    # - batch.<level>.<col>
    # - batch.<level>.<col>.<reduce>(dim=...)
    if isinstance(expr, ast.Attribute):
        chain = _parse_batch_attr_chain(expr, cur_name)
        if chain is None or len(chain) < 2:
            return None
        return [{"op": "df_get", "level0": chain[0], "col": chain[-1]}]

    if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute):
        red = expr.func.attr
        if red not in ("sum", "mean", "min", "max", "ptp"):
            return None
        chain = _parse_batch_attr_chain(expr.func.value, cur_name)
        if chain is None or len(chain) < 2:
            return None
        dim = _call_dim(expr)
        if dim is None:
            return None
        return [
            {"op": "df_get", "level0": chain[0], "col": chain[-1]},
            {"op": "reduce_tmp", "reduce": red, "dim": int(dim)},
        ]
    return None


def _call_dim(call: ast.Call) -> Optional[int]:
    # Accept dim as keyword or first positional arg.
    for kw in call.keywords:
        if kw.arg == "dim":
            return _const_int(kw.value)
    if call.args:
        return _const_int(call.args[0])
    return None


def _get_source(fn: Callable[..., Any]) -> Optional[str]:
    """
    Best-effort source retrieval that works for nested functions under pytest too.
    """
    try:
        return inspect.getsource(fn)
    except Exception:
        pass

    filename = getattr(getattr(fn, "__code__", None), "co_filename", None)
    first = getattr(getattr(fn, "__code__", None), "co_firstlineno", None)
    if not filename or not first:
        return None
    lines = linecache.getlines(filename)
    if not lines:
        return None
    start = max(0, int(first) - 1)

    # Incrementally extend until we can parse and find our function with end_lineno.
    buf: list[str] = []
    for end in range(start + 1, min(len(lines), start + 400) + 1):
        buf.append(lines[end - 1])
        snippet = "".join(buf)
        try:
            mod = ast.parse(textwrap.dedent(snippet))
        except SyntaxError:
            continue
        for node in mod.body:
            if not (
                isinstance(node, ast.FunctionDef)
                and node.name == getattr(fn, "__name__", "")
                and getattr(node, "end_lineno", None)
            ):
                continue
            rel_end = int(node.end_lineno)  # type: ignore[arg-type]
            chunk = "".join(buf[:rel_end])
            # Avoid returning a partial function body before the final `return`.
            if "return" in chunk:
                return chunk
    return None


def _const_number(node: ast.AST) -> Optional[int | float]:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        inner = _const_number(node.operand)
        if inner is None:
            return None
        return +inner if isinstance(node.op, ast.UAdd) else -inner
    return None


def _const_int(node: ast.AST) -> Optional[int]:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        inner = _const_int(node.operand)
        if inner is None:
            return None
        return +inner if isinstance(node.op, ast.UAdd) else -inner
    return None


def _const_bool(node: ast.AST) -> Optional[bool]:
    if isinstance(node, ast.Constant) and isinstance(node.value, bool):
        return node.value
    return None


