"""Cover the last uncovered compiler.py lines."""

from __future__ import annotations

import ast
import textwrap

import grumpy as gr
from grumpy.compiler import _compile_df_rhs, _compile_expr, _get_source, _parse_batch_attr_chain, _try_compile


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


def test_try_compile_df_short_assign_message():
    r = _try_compile(_fn("def f(batch):\n    batch.x = 1\n    return batch\n"))
    assert r.error and "batch.<level>.<col>" in r.error


def test_try_compile_df_bad_rhs_message():
    r = _try_compile(_fn("def f(batch):\n    batch.mol.out = [1, 2]\n    return batch\n"))
    assert r.error and "Unsupported RHS" in r.error


def test_try_compile_df_assignment_success():
    def f(batch):
        batch.mol.mol_center = batch.mol.mol_pos.mean(dim=-1)
        return batch

    r = _try_compile(f)
    assert r.plan is not None
    assert r.plan is not None and r.error is None


def test_neighbors_non_const_k():
    t = ast.parse("gr.neighbors(batch, batch, k=batch)")
    assert _compile_expr(t.body[0].value, "batch") is None  # type: ignore[attr-defined]


def test_parse_batch_attr_chain_not_batch():
    t = ast.parse("other.mol")
    assert _parse_batch_attr_chain(t.body[0].value, "batch") is None  # type: ignore[attr-defined]


def test_compile_df_rhs_short_chain():
    t = ast.parse("batch.mol")
    assert _compile_df_rhs(t.body[0].value, "batch") is None  # type: ignore[attr-defined]

    t2 = ast.parse("other.mol.pos.mean(dim=0)")
    assert _compile_df_rhs(t2.body[0].value, "batch") is None  # type: ignore[attr-defined]


def test_get_source_falls_back_to_linecache(monkeypatch):
    import inspect

    from tests.fixtures.compile_helper import batch_transform

    monkeypatch.setattr(inspect, "getsource", lambda _f: (_ for _ in ()).throw(OSError()))
    src = _get_source(batch_transform)
    assert src is not None and "return" in src


def test_get_source_returns_none_for_missing_fn(monkeypatch):
    import inspect

    def other(batch):
        return batch

    other.__name__ = "does_not_exist_in_file"
    monkeypatch.setattr(inspect, "getsource", lambda _f: (_ for _ in ()).throw(OSError()))
    assert _get_source(other) is None
