"""Union layout support for reduce, stats, sort/search, and einsum."""

from __future__ import annotations

import grumpy as gr
import numpy as np


def _union_int():
    return gr.array([1, [2, 3], 4, [5]], dtype=gr.int64)


def _union_float():
    return gr.array([1.0, [2.0, 3.0], 4.0], dtype=gr.float64)


def test_union_min_max_ptp_mean():
    x = _union_int()
    assert x.min() == 1
    assert x.max() == 5
    assert x.ptp() == 4
    assert x.mean() == 3.0
    assert x.min(dim=0).to_list() == [1, 2, 4, 5]
    assert x.max(dim=0).to_list() == [1, 3, 4, 5]


def test_union_var_std():
    x = _union_float()
    assert abs(x.var() - np.var([1.0, 2.0, 3.0, 4.0])) < 1e-9
    assert abs(x.std() - np.std([1.0, 2.0, 3.0, 4.0])) < 1e-9
    v0 = x.var(dim=0).to_list()
    assert len(v0) == 3
    assert v0[0] == 0.0
    assert abs(v0[1] - 0.25) < 1e-9


def test_union_sort_argsort():
    x = gr.array([3, [1, 2], 0], dtype=gr.int64)
    assert x.sort(dim=-1).to_list() == [3, [1, 2], 0]
    y = gr.array([3, [2, 1], 0], dtype=gr.int64)
    assert y.sort(dim=-1).to_list() == [3, [1, 2], 0]


def test_union_argmin_argmax_dim():
    x = gr.array([1, [3, 2], 0], dtype=gr.int64)
    assert x.argmin(dim=-1).to_list() == [0, 1, 0]
    assert x.argmax(dim=-1).to_list() == [0, 0, 0]


def test_union_search_sorted():
    # Flatten order [0, 1, 2, 3] is sorted for searchsorted.
    x = gr.array([0, [1, 2], 3], dtype=gr.int64)
    v = gr.array([2, 3], dtype=gr.int64)
    assert gr.search_sorted(x, v).to_list() == [2, 3]


def test_union_einsum_dot():
    a = _union_float()
    b = gr.array([1.0, 1.0, 1.0, 1.0], dtype=gr.float64)
    assert abs(gr.einsum("i,i->", a, b) - 10.0) < 1e-9
    assert abs(gr.tensordot(a, b, axes=1) - 10.0) < 1e-9
