"""Cover remaining compiler.py branches."""

from __future__ import annotations

import textwrap

import pytest

import grumpy as gr
from grumpy.compiler import _compile_df_rhs, _compile_expr, _get_source, _try_compile, _try_compile_to_ops


def _fn(src: str):
    import os
    import tempfile
    import builtins

    body = textwrap.dedent(src)
    fd, path = tempfile.mkstemp(suffix=".py")
    with os.fdopen(fd, "w") as fh:
        fh.write(body)
    ns = {"gr": gr, "grumpy": gr}
    exec(builtins.compile(body, path, "exec"), ns, ns)
    return ns["f"]


def test_docstring_strip_in_try_compile_to_ops():
    ops = _try_compile_to_ops(_fn("def f(batch):\n    '''d'''\n    batch = batch + 1\n    return batch\n"))
    assert ops is not None


def test_df_assign_short_chain_fails():
    assert _try_compile_to_ops(_fn("def f(batch):\n    batch.x = 1\n    return batch\n")) is None


def test_try_compile_df_assign_and_bad_rhs():
    assert _try_compile(_fn("def f(batch):\n    batch.mol.x = 1\n    return batch\n")).error
    assert _try_compile(_fn("def f(batch):\n    batch.mol.out = batch + 1\n    return batch\n")).error


def test_try_compile_empty_after_docstring():
    assert _try_compile(_fn("def f(batch):\n    '''only doc'''\n")).error


def test_compile_sub_div_and_unary_usub():
    ops = _try_compile_to_ops(_fn("def f(batch):\n    batch = batch - 3\n    batch = batch / 2\n    return batch\n"))
    assert {o["op"] for o in ops or []} >= {"sub_scalar", "div_scalar"}


def test_compile_failed_plan(monkeypatch):
    import grumpy.compiler as cm

    def boom(_ops):
        raise RuntimeError("bad plan")

    monkeypatch.setattr(cm._core, "CompiledPlan", boom)
    r = _try_compile(_fn("def f(batch):\n    batch = batch + 1\n    return batch\n"))
    assert r.error


def test_compile_df_rhs_edges():
    import ast

    assert _compile_df_rhs(ast.parse("1").body[0].value, "batch") is None  # type: ignore[attr-defined]
    tree = ast.parse("batch.mol.atom_pos.other(dim=1)")
    assert _compile_df_rhs(tree.body[0].value, "batch") is None  # type: ignore[attr-defined]
    tree2 = ast.parse("batch.mol.atom_pos.mean()")
    assert _compile_df_rhs(tree2.body[0].value, "batch") is None  # type: ignore[attr-defined]


def test_get_source_linecache_empty(monkeypatch):
    import grumpy.compiler as cm

    monkeypatch.setattr(cm.inspect, "getsource", lambda _f: (_ for _ in ()).throw(OSError()))
    monkeypatch.setattr(cm.linecache, "getlines", lambda _p: [])

    def f(batch):
        batch = batch + 1
        return batch

    assert _get_source(f) is None


def test_get_source_without_end_lineno(tmp_path):
    path = tmp_path / "mod.py"
    path.write_text("def f(batch):\n    batch = batch + 1\n    return batch\n")
    ns: dict = {}
    import builtins

    exec(builtins.compile(path.read_text(), str(path), "exec"), ns, ns)
    fn = ns["f"]
    src = _get_source(fn)
    assert src is not None


def test_const_unary_minus():
    import ast
    from grumpy.compiler import _const_int, _const_number

    assert _const_number(ast.parse("-3").body[0].value) == -3  # type: ignore[attr-defined]
    assert _const_int(ast.parse("-4").body[0].value) == -4  # type: ignore[attr-defined]
