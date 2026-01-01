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
    Wrapper returned by @gr.compile.

    If compilation succeeds, calling the wrapper runs a Rust-executed plan (GIL-released).
    If compilation fails, it falls back to the original Python function and emits a one-time warning.
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
        return self._plan is not None

    @property
    def compile_error(self) -> Optional[str]:
        return self._compile_error

    def __call__(self, batch):
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
    Decorator that attempts to compile a restricted batch transform to a Rust-executed plan.

    Supported (MVP):
    - Straight-line code only (no if/for/while/try/with)
    - Single parameter (batch)
    - Rebinding `batch = ...` and final `return batch`
    - Operations:
      - `batch <op> scalar` where op in {+,-,*,/,%,remainder}
      - `gr.neighbors(batch, batch, k=..., dim=..., loop=...)`
    """
    res = _try_compile(fn)
    return CompiledTransform(fn, res)

def compile_pipeline(fns: list[Callable[[Any], Any]]) -> Callable[[Any], Any]:
    """
    Compile and fuse a list of transforms into one or more CompiledPlan segments.

    - Consecutive compilable transforms are fused into a single _core.CompiledPlan.
    - Uncompilable transforms run as Python callables (with one-time warning describing why).
    - Output order and semantics match sequential application.
    """

    segments: list[Callable[[Any], Any]] = []

    pending_ops: list[dict[str, Any]] = []

    def flush_ops():
        nonlocal pending_ops
        if not pending_ops:
            return
        plan = _core.CompiledPlan(pending_ops)
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
    run_all: Callable[[Any], Any]
    fully_compiled: bool
    fused_ops: Optional[list[dict[str, Any]]]


def compile_pipeline_info(fns: list[Callable[[Any], Any]]) -> PipelineInfo:
    """
    Like compile_pipeline, but also returns whether the full pipeline is a single fused plan,
    and (if so) the fused raw op dict list.
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

    # batch.sum(dim=...), batch.mean(dim=...), batch.min(dim=...), batch.max(dim=...), batch.ptp(dim=...)
    if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute):
        if isinstance(expr.func.value, ast.Name) and expr.func.value.id == cur_name:
            red = expr.func.attr
            if red in ("sum", "mean", "min", "max", "ptp"):
                dim = _call_dim(expr)
                if dim is None:
                    return None
                return {"op": "reduce", "reduce": red, "dim": int(dim)}

    return None


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
            if isinstance(node, ast.FunctionDef) and node.name == getattr(fn, "__name__", ""):
                # If end_lineno exists, trim exactly.
                if getattr(node, "end_lineno", None):
                    rel_end = int(node.end_lineno)
                    return "".join(buf[:rel_end])
                return snippet
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


