"""Hit remaining compiler.py branches for full coverage."""

from __future__ import annotations

import ast
import textwrap

import pytest

import grumpy as gr
from grumpy.compiler import (
    _compile_expr,
    _get_source,
    _try_compile,
    _try_compile_to_ops,
)


def _make(src: str):
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
        pass


def test_try_compile_to_ops_failure_modes():
    assert _try_compile_to_ops(_make("def f(batch):\n    return batch\n")) is None
    assert _try_compile_to_ops(_make("def f(batch):\n    batch = batch ** 2\n    return batch\n")) is None
    assert _try_compile_to_ops(_make("def f(batch):\n    batch.x.y = 1\n    return batch\n")) is None
    assert _try_compile_to_ops(_make("def f(batch):\n    other = batch\n    return batch\n")) is None
    assert _try_compile_to_ops(_make("def f(batch):\n    batch = gr.neighbors(batch, batch, radius=1.0)\n    return batch\n")) is None
    assert _try_compile_to_ops(_make("def f(batch):\n    batch = batch.other\n    return batch\n")) is None


def test_try_compile_return_and_docstring():
    r = _try_compile(_make("def f(batch):\n    '''d'''\n    batch = batch + 1\n    return batch\n"))
    assert r.plan is not None
    r2 = _try_compile(_make("def f(batch):\n    batch = batch + 1\n    return 0\n"))
    assert r2.error is not None


def test_compile_expr_neighbors_and_binop_edges():
    import ast as astmod

    tree = astmod.parse("batch + x")
    assert _compile_expr(tree.body[0].value, "batch") is None  # type: ignore[attr-defined]
    tree = astmod.parse("batch + 'x'")
    assert _compile_expr(tree.body[0].value, "batch") is None  # type: ignore[attr-defined]
    tree = astmod.parse("gr.neighbors(batch, other, k=1)")
    assert _compile_expr(tree.body[0].value, "batch") is None  # type: ignore[attr-defined]
    tree = astmod.parse("batch.unknown(dim=0)")
    assert _compile_expr(tree.body[0].value, "batch") is None  # type: ignore[attr-defined]


def test_get_source_incremental_parse(monkeypatch):
    def fn(batch):
        batch = batch + 1
        return batch

    # Force inspect.getsource failure then linecache path.
    import grumpy.compiler as cm

    monkeypatch.setattr(cm.inspect, "getsource", lambda _f: (_ for _ in ()).throw(OSError()))
    src = _get_source(fn)
    assert src is not None and "batch" in src


def test_get_source_no_filename():
    class C:
        pass

    assert _get_source(C()) is None  # type: ignore[arg-type]
