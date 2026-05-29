"""Direct _compile_expr / _call_dim edge coverage."""

from __future__ import annotations

import ast

from grumpy.compiler import _call_dim, _compile_expr, _const_int, _const_number


def test_compile_expr_binop_and_neighbors_edges():
    t = ast.parse("other + 1")
    assert _compile_expr(t.body[0].value, "batch") is None  # type: ignore[attr-defined]

    t = ast.parse("batch + 'x'")
    assert _compile_expr(t.body[0].value, "batch") is None  # type: ignore[attr-defined]

    t = ast.parse("batch // 2")
    assert _compile_expr(t.body[0].value, "batch") is None  # type: ignore[attr-defined]

    t = ast.parse("gr.neighbors(batch)")
    assert _compile_expr(t.body[0].value, "batch") is None  # type: ignore[attr-defined]

    t = ast.parse("gr.neighbors(other, batch, k=1)")
    assert _compile_expr(t.body[0].value, "batch") is None  # type: ignore[attr-defined]

    t = ast.parse("batch.unknown(dim=0)")
    assert _compile_expr(t.body[0].value, "batch") is None  # type: ignore[attr-defined]

    t = ast.parse("batch.sum()")
    assert _compile_expr(t.body[0].value, "batch") is None  # type: ignore[attr-defined]


def test_call_dim_positional():
    t = ast.parse("batch.sum(2)")
    assert _call_dim(t.body[0].value) == 2  # type: ignore[arg-type]


def test_const_inner_none():
    t = ast.parse("-'a'")
    node = t.body[0].value  # type: ignore[attr-defined]
    assert _const_number(node) is None
    assert _const_int(node) is None
