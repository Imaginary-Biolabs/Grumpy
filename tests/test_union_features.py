"""UnionScalarList feature tests (compact slice, stream, scalar ops, reduce, unique)."""

from __future__ import annotations

import grumpy as gr


def _union_array():
    return gr.array([1, [2, 3], 4, [5]], dtype=gr.int64)


def test_union_scalar_mul_compiled_path(tmp_path):
    x = gr.array([1.0, [2.0, 3.0], 4.0], dtype=gr.float64)
    path = str(tmp_path / "u.gr")
    gr.save(x, path, chunk_size=2)

    def mul_only(batch):
        batch = batch * 2.0
        return batch

    st = gr.stream(path, batch_size=1)
    out = list(st.apply(mul_only, cpu=1, compile=True, scheduler="rust"))
    assert [b.to_list() for b in out] == [[2.0], [[4.0, 6.0]], [8.0]]


def test_union_stream_load_slice_parity(tmp_path):
    x = _union_array()
    path = str(tmp_path / "u.gr")
    gr.save(x, path, chunk_size=2)
    full = gr.load(path)
    partial = gr._core.load_slice(path, 1, 3)
    assert partial.to_list() == full[1:3].to_list()


def test_union_stream_batches(tmp_path):
    x = _union_array()
    path = str(tmp_path / "u.gr")
    gr.save(x, path)
    batches = [b.to_list() for b in gr.stream(path, batch_size=2)]
    assert batches == [[1, [2, 3]], [4, [5]]]


def test_union_sum_all():
    x = _union_array()
    assert x.sum() == 15


def test_union_sum_dim0():
    x = _union_array()
    assert x.sum(dim=0).to_list() == [1, 5, 4, 5]


def test_union_unique():
    x = gr.array([1, [2, 1], 3, [2]], dtype=gr.int64)
    assert gr.unique(x).to_list() == [1, 2, 3]


def test_union_save_generator(tmp_path):
    path = str(tmp_path / "gen.gr")

    def batches():
        yield gr.array([1, [2]], dtype=gr.int64)
        yield gr.array([[3, 4]], dtype=gr.int64)

    gr.save(batches(), path)
    loaded = gr.load(path)
    assert loaded.to_list() == [1, [2], [3, 4]]


def test_union_shuffle_reproducible():
    a = _union_array()
    b = _union_array()
    a.shuffle(dim=0, seed=7)
    b.shuffle(dim=0, seed=7)
    assert a.to_list() == b.to_list()
    assert len(a.to_list()) == len(_union_array().to_list())


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
