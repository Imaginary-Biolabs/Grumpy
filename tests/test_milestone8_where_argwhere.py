import numpy as np

import grumpy as gr


def test_where_indices_matches_numpy():
    rng = np.random.default_rng(0)
    cond = rng.random(1000) > 0.7
    gc = gr.array(cond.tolist(), dtype=gr.bool_)
    out = np.array(gr.where(gc).to_list(), dtype=np.int64)
    ref = np.where(cond)[0]
    assert np.array_equal(out, ref)


def test_argwhere_matches_numpy_1d():
    cond = np.array([True, False, True], dtype=np.bool_)
    gc = gr.array(cond.tolist(), dtype=gr.bool_)
    out = np.array(gr.argwhere(gc).to_list(), dtype=object)
    # grumpy returns [[i],[j],...]
    ref = np.argwhere(cond)
    assert out.tolist() == ref.tolist()


def test_where_select_matches_numpy_int32():
    rng = np.random.default_rng(0)
    cond = rng.random(1000) > 0.5
    x = rng.integers(-10, 10, size=(1000,), dtype=np.int32)
    y = rng.integers(-10, 10, size=(1000,), dtype=np.int32)
    gc = gr.array(cond.tolist(), dtype=gr.bool_)
    gx = gr.array(x.tolist(), dtype=gr.int32)
    gy = gr.array(y.tolist(), dtype=gr.int32)
    out = np.array(gr.where(gc, gx, gy).to_list(), dtype=np.int32)
    ref = np.where(cond, x, y)
    assert np.array_equal(out, ref)


