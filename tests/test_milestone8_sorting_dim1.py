import numpy as np

import grumpy as gr


def test_sort_dim1_rectangular_matches_numpy():
    rng = np.random.default_rng(0)
    a = rng.integers(-1000, 1000, size=(64, 128), dtype=np.int32)
    x = gr.array(a.tolist(), dtype=gr.int32)
    out = np.array(x.sort(dim=1).to_list(), dtype=np.int32)
    ref = np.sort(a, axis=1)
    assert np.array_equal(out, ref)


def test_argsort_dim1_rectangular_correctness():
    rng = np.random.default_rng(0)
    a = rng.integers(-1000, 1000, size=(64, 128), dtype=np.int32)
    x = gr.array(a.tolist(), dtype=gr.int32)
    idx = np.array(x.argsort(dim=1).to_list(), dtype=np.int64)
    assert idx.shape == a.shape
    # validate each row is a permutation and produces sorted values
    for r in range(a.shape[0]):
        row_idx = idx[r]
        assert np.array_equal(np.sort(row_idx), np.arange(a.shape[1], dtype=np.int64))
        assert np.array_equal(a[r][row_idx], np.sort(a[r]))


def test_sort_dim1_ragged():
    x = gr.array([[3, 1, 2], [5], [2, 2, 1]], dtype=gr.int32)
    assert x.sort(dim=1).to_list() == [[1, 2, 3], [5], [1, 2, 2]]


def test_partition_dim1_rectangular_kth_matches_numpy_kth_element():
    rng = np.random.default_rng(0)
    a = rng.integers(-1000, 1000, size=(32, 129), dtype=np.int32)
    k = 17
    x = gr.array(a.tolist(), dtype=gr.int32)
    out = np.array(x.partition(k, dim=1).to_list(), dtype=np.int32)
    ref = np.partition(a, k, axis=1)
    assert np.array_equal(out[:, k], ref[:, k])


def test_argpartition_dim1_rectangular_kth_value_matches_numpy():
    rng = np.random.default_rng(0)
    a = rng.integers(-1000, 1000, size=(16, 257), dtype=np.int32)
    k = 33
    x = gr.array(a.tolist(), dtype=gr.int32)
    idx = np.array(x.argpartition(k, dim=1).to_list(), dtype=np.int64)
    assert idx.shape == a.shape
    for r in range(a.shape[0]):
        assert np.array_equal(np.sort(idx[r]), np.arange(a.shape[1], dtype=np.int64))
        assert a[r][idx[r]][k] == np.partition(a[r], k)[k]


def test_argmax_argmin_dim1_rectangular_matches_numpy():
    rng = np.random.default_rng(0)
    a = rng.integers(-1000, 1000, size=(64, 128), dtype=np.int32)
    x = gr.array(a.tolist(), dtype=gr.int32)
    out_max = np.array(x.argmax(dim=1).to_list(), dtype=np.int64)
    out_min = np.array(x.argmin(dim=1).to_list(), dtype=np.int64)
    assert np.array_equal(out_max, np.argmax(a, axis=1))
    assert np.array_equal(out_min, np.argmin(a, axis=1))


