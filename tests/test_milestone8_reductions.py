import math

import numpy as np

import grumpy as gr


def test_mean_dim1_example():
    x = gr.array([[1, 2, 3], [4, 5]], dtype=gr.int32)
    assert x.mean(dim=1).to_list() == [2.0, 4.5]


def test_ptp_dim0_example():
    x = gr.array([[1, 2, 3], [4, 5]], dtype=gr.int32)
    assert x.ptp(dim=0).to_list() == [3, 3, None]


def test_sum_dim0_ragged_with_none_skips_none():
    x = gr.array([[1, None, 3], [4, 5]], dtype=gr.int32)
    assert x.sum(dim=0).to_list() == [5, None, None]


def test_min_dim1_float64_nan_propagates():
    x = gr.array([[1.0, float("nan")], [2.0, 3.0]], dtype=gr.float64)
    out = x.min(dim=1).to_list()
    assert math.isnan(out[0])
    assert out[1] == 2.0


def test_rectangular_matches_numpy_reductions():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 1000, size=(64, 128), dtype=np.int32)
    x = gr.array(a.tolist(), dtype=gr.int32)

    # sum/mean (int -> int64 / float64)
    np_sum1 = a.sum(axis=1, dtype=np.int64)
    np_sum0 = a.sum(axis=0, dtype=np.int64)
    assert np.array_equal(np.array(x.sum(dim=1).to_list(), dtype=np.int64), np_sum1)
    assert np.array_equal(np.array(x.sum(dim=0).to_list(), dtype=np.int64), np_sum0)

    np_mean1 = a.mean(axis=1, dtype=np.float64)
    np_mean0 = a.mean(axis=0, dtype=np.float64)
    assert np.allclose(np.array(x.mean(dim=1).to_list(), dtype=np.float64), np_mean1)
    assert np.allclose(np.array(x.mean(dim=0).to_list(), dtype=np.float64), np_mean0)

    # min/max/ptp preserve dtype
    assert np.array_equal(np.array(x.min(dim=1).to_list(), dtype=np.int32), a.min(axis=1))
    assert np.array_equal(np.array(x.max(dim=1).to_list(), dtype=np.int32), a.max(axis=1))
    assert np.array_equal(np.array(x.ptp(dim=1).to_list(), dtype=np.int32), np.ptp(a, axis=1))


