"""Tier 3: UnionScalarList, dtype casting, string ops, einsum parity."""

from __future__ import annotations

import numpy as np
import pytest

import grumpy as gr


def test_cast_mixed_int32_float64_add():
    a = gr.array([[1, 2], [3, 4]], dtype=gr.int32)
    b = gr.array([[1.0, 2.0], [3.0, 4.0]], dtype=gr.float64)
    out = (a + b).to_list()
    assert out == [[2.0, 4.0], [6.0, 8.0]]


def test_union_elementwise_and_unary():
    x = gr.array([[1, 2, 3], [[None, 5], [6]]], dtype=gr.int64)
    doubled = (x + x).to_list()
    assert doubled == [[2, 4, 6], [[None, 10], [12]]]


def test_union_sin():
    x = gr.array([1.0, [2.0, 3.0]], dtype=gr.float64)
    out = gr.sin(x).to_list()
    assert len(out) == 2


def test_union_compare():
    a = gr.array([1, [2, 3]], dtype=gr.int32)
    b = gr.array([1, [2, 3]], dtype=gr.int32)
    eq = gr.equal(a, b).to_list()
    assert eq == [True, [True, True]]


def test_string_concat_and_compare():
    a = gr.array([["a", "b"], ["c"]], dtype=gr.string)
    b = gr.array([["!", "?"], ["#"]], dtype=gr.string)
    out = (a + b).to_list()
    assert out == [["a!", "b?"], ["c#"]]
    lt = gr.less(a, b).to_list()
    assert lt[0][0] is False  # 'a' > '!'


def test_string_unique_isin():
    a = gr.array(["b", "a", "b", "c"], dtype=gr.string)
    u = gr.unique(a).to_list()
    assert u == ["a", "b", "c"]
    mask = gr.isin(a, gr.array(["a", "c"], dtype=gr.string)).to_list()
    assert mask == [False, True, False, True]


def test_einsum_numpy_fallback_matmul_transpose():
    a_np = np.arange(6, dtype=np.float64).reshape(2, 3)
    b_np = np.arange(12, dtype=np.float64).reshape(3, 4)
    a = gr.array(a_np.tolist(), dtype=gr.float64)
    b = gr.array(b_np.tolist(), dtype=gr.float64)
    gr_out = gr.einsum("ij,jk->ik", a, b).to_list()
    np_out = np.einsum("ij,jk->ik", a_np, b_np).tolist()
    assert gr_out == np_out
    gr_t = gr.einsum("ij->ji", a).to_list()
    np_t = np.einsum("ij->ji", a_np).tolist()
    assert gr_t == np_t


@pytest.mark.parametrize(
    "pattern,build",
    [
        ("i->", lambda: (gr.array([1.0, 2.0, 3.0], dtype=gr.float64),)),
        ("ij->", lambda: (gr.array([[1.0, 2.0], [3.0, 4.0]], dtype=gr.float64),)),
    ],
)
def test_einsum_rust_fast_paths(pattern, build):
    ops = build()
    if len(ops) == 1:
        _ = gr.einsum(pattern, ops[0])
    else:
        _ = gr.einsum(pattern, ops[0], ops[1])
