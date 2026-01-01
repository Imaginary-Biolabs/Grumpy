import numpy as np

import grumpy as gr


def test_sort_argsort_int32_matches_numpy():
    rng = np.random.default_rng(0)
    a = rng.integers(-1000, 1000, size=(10000,), dtype=np.int32)
    x = gr.array(a.tolist(), dtype=gr.int32)
    assert np.array_equal(np.array(x.sort().to_list(), dtype=np.int32), np.sort(a))
    # argsort tie ordering is not guaranteed by NumPy (default quicksort is unstable),
    # so we validate the indices as a correct argsort result.
    idx = np.array(x.argsort().to_list(), dtype=np.int64)
    assert idx.shape == (a.shape[0],)
    assert np.array_equal(np.sort(idx), np.arange(a.shape[0], dtype=np.int64))
    assert np.array_equal(a[idx], np.sort(a))


def test_sort_float64_nans_last_like_numpy():
    a = np.array([np.nan, 3.0, -1.0, np.nan, 2.0], dtype=np.float64)
    x = gr.array(a.tolist(), dtype=gr.float64)
    out = np.array(x.sort().to_list(), dtype=np.float64)
    ref = np.sort(a)  # NaNs last
    assert np.allclose(out, ref, equal_nan=True)


def test_argmax_argmin_basic():
    a = np.array([1, 5, 2, 4], dtype=np.int32)
    x = gr.array(a.tolist(), dtype=gr.int32)
    assert x.argmax() == int(np.argmax(a))
    assert x.argmin() == int(np.argmin(a))


def test_nanargmax_nanargmin_match_numpy():
    a = np.array([np.nan, 1.0, 2.0], dtype=np.float64)
    x = gr.array(a.tolist(), dtype=gr.float64)
    assert x.nanargmax() == int(np.nanargmax(a))
    assert x.nanargmin() == int(np.nanargmin(a))


def test_nonzero_bool_matches_numpy():
    a = np.array([True, False, True, True], dtype=np.bool_)
    x = gr.array(a.tolist(), dtype=gr.bool_)
    out = np.array(gr.nonzero(x).to_list(), dtype=np.int64)
    ref = np.nonzero(a)[0]
    assert np.array_equal(out, ref)


def test_search_sorted_matches_numpy_left_right():
    a = np.array([1, 2, 2, 4], dtype=np.int32)
    v = np.array([0, 2, 3, 5], dtype=np.int32)
    x = gr.array(a.tolist(), dtype=gr.int32)
    gv = gr.array(v.tolist(), dtype=gr.int32)
    out_l = np.array(gr.search_sorted(x, gv, right=False).to_list(), dtype=np.int64)
    out_r = np.array(gr.search_sorted(x, gv, right=True).to_list(), dtype=np.int64)
    ref_l = np.searchsorted(a, v, side="left")
    ref_r = np.searchsorted(a, v, side="right")
    assert np.array_equal(out_l, ref_l)
    assert np.array_equal(out_r, ref_r)


def test_partition_matches_numpy_kth_element():
    rng = np.random.default_rng(0)
    a = rng.integers(-1000, 1000, size=(10001,), dtype=np.int32)
    k = 5000
    x = gr.array(a.tolist(), dtype=gr.int32)
    out = np.array(x.partition(k).to_list(), dtype=np.int32)
    ref = np.partition(a, k)
    # partition doesn't fully sort; only kth element is guaranteed and partition property.
    assert out[k] == ref[k]
    assert np.all(out[:k] <= out[k])
    assert np.all(out[k:] >= out[k])


def test_argpartition_matches_numpy_kth_value():
    rng = np.random.default_rng(0)
    a = rng.integers(-1000, 1000, size=(10001,), dtype=np.int32)
    k = 1234
    x = gr.array(a.tolist(), dtype=gr.int32)
    idx = np.array(x.argpartition(k).to_list(), dtype=np.int64)
    assert idx.shape == (a.shape[0],)
    # permutation
    assert np.array_equal(np.sort(idx), np.arange(a.shape[0], dtype=np.int64))
    # kth value matches
    ref = np.partition(a, k)[k]
    assert a[idx[k]] == ref


