"""Additional compiler coverage for error paths and opcode variants."""

from __future__ import annotations

import textwrap
import warnings

import pytest

import grumpy as gr
from grumpy.compiler import (
    _call_dim,
    _compile_df_rhs,
    _compile_expr,
    _get_source,
    _try_compile,
    _try_compile_to_ops,
    compile_pipeline,
    compile_pipeline_info,
)


def test_compile_all_scalar_binops():
    def f(batch):
        batch = batch + 1
        batch = batch - 1
        batch = batch * 2
        batch = batch / 2
        batch = batch % 2
        return batch

    ops = _try_compile_to_ops(f)
    assert ops is not None
    names = {o["op"] for o in ops}
    assert names == {"add_scalar", "sub_scalar", "mul_scalar", "div_scalar", "mod_scalar"}


def test_compile_reduce_ops():
    for red in ("sum", "mean", "min", "max", "ptp"):
        src = f"""
        def f(batch):
            batch = batch.{red}(dim=1)
            return batch
        """
        ops = _try_compile_to_ops(_make_fn(src))
        assert ops is not None and ops[-1]["reduce"] == red


def test_compile_df_get_and_reduce_tmp():
    def f(batch):
        batch.residue.out = batch.residue.pos.mean(dim=-1)
        return batch

    ops = _try_compile_to_ops(f)
    assert ops is not None
    assert any(o["op"] == "df_get" for o in ops)
    assert any(o["op"] == "reduce_tmp" for o in ops)


def test_compile_pipeline_flush_and_segments():
    def a(batch):
        batch = batch + 1
        return batch

    def b(batch):
        if True:
            return batch
        return batch

    run = compile_pipeline([a, b])
    x = gr.array([1], dtype=gr.int32)
    with pytest.warns(UserWarning):
        assert run(x).to_list() == [2]

    info = compile_pipeline_info([a, b])
    assert not info.fully_compiled


def test_try_compile_parse_and_assign_errors():
    assert _try_compile(_make_fn("def f(batch):\n    batch.m.a = 1\n    return batch\n")).error
    assert _try_compile(_make_fn("def f(batch):\n    other = batch\n    return batch\n")).error

    src = textwrap.dedent(
        """
        def f(batch):
            '''only doc'''
            return batch
        """
    )
    assert _try_compile(_make_fn(src)).error


def test_get_source_linecache_extension():
    def outer():
        def inner(batch):
            batch = batch + 1
            return batch

        return inner

    inner = outer()
    assert _get_source(inner) is not None


def test_compile_expr_and_df_rhs_direct():
    import ast

    tree = ast.parse("batch + 1")
    expr = tree.body[0].value  # type: ignore[attr-defined]
    assert _compile_expr(expr, "batch")["op"] == "add_scalar"
    assert _compile_df_rhs(ast.parse("batch.m.col").body[0].value, "batch") is not None  # type: ignore[attr-defined]

    call = ast.parse("batch.m.col.mean(dim=1)").body[0].value  # type: ignore[attr-defined]
    rhs = _compile_df_rhs(call, "batch")
    assert rhs is not None and rhs[-1]["op"] == "reduce_tmp"

    assert _call_dim(ast.parse("batch.sum()").body[0].value) is None  # type: ignore[attr-defined]


def _make_fn(src: str):
    import os
    import tempfile

    body = textwrap.dedent(src)
    fd, path = tempfile.mkstemp(suffix=".py")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(body)
        ns = {"gr": gr, "grumpy": gr}
        import builtins

        exec(builtins.compile(body, path, "exec"), ns, ns)
        return ns["f"]
    finally:
        pass  # keep path so inspect.getsource / linecache can read the file
