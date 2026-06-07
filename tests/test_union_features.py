"""UnionScalarList feature tests (compact slice, stream, scalar ops, reduce, unique)."""

from __future__ import annotations

import grumpy as gr


def _union_array():
    return gr.array([1, [2, 3], 4, [5]], dtype=gr.int64)


def test_union_scalar_mul_compiled_path():
    x = gr.array([1.0, [2.0, 3.0], 4.0], dtype=gr.float64)

    def mul_only(batch):
        batch = batch * 2.0
        return batch

    out = gr.compile(mul_only)(x)
    assert out.to_list() == [2.0, [4.0, 6.0], 8.0]


def test_union_load_slice(tmp_path):
    x = _union_array()
    path = str(tmp_path / "u.gr")
    gr.save(x, path)
    assert gr._core.load_slice(path, 0, 2).to_list() == x[0:2].to_list()
    assert gr._core.load_slice(path, 2, 4).to_list() == x[2:4].to_list()


def test_union_sum_all():
    x = _union_array()
    assert x.sum() == 15


def test_union_unique():
    x = gr.array([1, [2, 1], 3, [2]], dtype=gr.int64)
    assert gr.unique(x).to_list() == [1, 2, 3]


def test_union_shuffle_reproducible():
    orig = _union_array().to_list()
    a = _union_array()
    b = _union_array()
    a.shuffle(dim=0, seed=7)
    b.shuffle(dim=0, seed=7)
    assert a.to_list() == b.to_list()
    assert a.to_list() != orig
    c = _union_array()
    c.shuffle(dim=0, seed=99)
    assert c.to_list() != a.to_list()


def test_union_fancy_axis0():
    x = _union_array()
    assert x[[0, 2]].to_list() == [1, 4]
    assert x[[1, 3]].to_list() == [[2, 3], [5]]
    assert x[[0, 1]].to_list() == [1, [2, 3]]
    assert x[[True, False, True, False]].to_list() == [1, 4]
    assert x[1:3].to_list() == [[2, 3], 4]


def test_union_fancy_coordinates():
    x = _union_array()
    assert x[0, 0] == 1
    assert x[1, 0] == 2
    assert x[[0, 1], [0, 0]].to_list() == [1, 2]
    assert x[[0, 1], 0].to_list() == [1, 2]


def test_union_coordinate_assignment():
    x = _union_array()
    x[1, 0] = 99
    assert x.to_list() == [1, [99, 3], 4, [5]]
    x[0, 0] = 10
    assert x[0, 0] == 10


def test_union_quantile_dim0():
    x = gr.array([1.0, [2.0, 3.0], 4.0], dtype=gr.float64)
    out = x.quantile(0.5, dim=0)
    # Per outer element: median of each row's leaves.
    assert out.to_list() == [1.0, 2.5, 4.0]


def test_union_quantile_last_axis():
    x = gr.array([1.0, [2.0, 4.0], 3.0], dtype=gr.float64)
    out = x.quantile(0.5, dim=-1)
    assert out.to_list() == [1.0, 3.0, 3.0]
