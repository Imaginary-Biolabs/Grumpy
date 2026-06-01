"""Broadcasting between different UnionScalarList structures."""

from __future__ import annotations

import grumpy as gr


def test_union_times_flat_leaf_same_outer_len():
    a = gr.array([1, [2, 3], 4], dtype=gr.int64)
    b = gr.array([10, 20, 30], dtype=gr.int64)
    assert (a * b).to_list() == [10, [40, 60], 120]


def test_union_add_mixed_union_structures():
    a = gr.array([1, [2, 3], 4], dtype=gr.int64)
    b = gr.array([[10, 11], 20, [30]], dtype=gr.int64)
    assert (a + b).to_list() == [[11, 12], [22, 23], 34]


def test_union_axis0_broadcast_len1():
    a = gr.array([1, [2, 3], 4], dtype=gr.int64)
    b = gr.array([10], dtype=gr.int64)
    assert (a + b).to_list() == [11, [12, 13], 14]


def test_union_scalar_broadcast():
    a = gr.array([1, [2, 3], 4], dtype=gr.float64)
    assert (a * 2.0).to_list() == [2.0, [4.0, 6.0], 8.0]


def test_union_broadcast_with_list_chain():
    a = gr.array([1, [2, 3], 4], dtype=gr.int64)
    b = gr.array([[10], [20], [30]], dtype=gr.int64)
    assert (a + b).to_list() == [11, [22, 23], 34]
