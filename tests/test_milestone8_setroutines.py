import numpy as np

import grumpy as gr


def test_unique_matches_numpy_int32_flattened():
    rng = np.random.default_rng(0)
    a = rng.integers(-1000, 1000, size=(64, 128), dtype=np.int32)
    x = gr.array(a.tolist(), dtype=gr.int32)
    out = np.array(gr.unique(x).to_list(), dtype=np.int32)
    ref = np.unique(a)
    assert np.array_equal(out, ref)


def test_unique_matches_numpy_float64_with_nan():
    a = np.array([1.0, np.nan, 2.0, np.nan, -0.0, 0.0], dtype=np.float64)
    x = gr.array(a.tolist(), dtype=gr.float64)
    out = np.array(gr.unique(x).to_list(), dtype=np.float64)
    ref = np.unique(a)
    assert np.allclose(out, ref, equal_nan=True)


def test_isin_matches_numpy_rectangular_int32():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 100, size=(32, 64), dtype=np.int32)
    test = rng.integers(0, 100, size=(128,), dtype=np.int32)
    x = gr.array(a.tolist(), dtype=gr.int32)
    t = gr.array(test.tolist(), dtype=gr.int32)
    out = np.array(gr.isin(x, t).to_list(), dtype=np.bool_)
    ref = np.isin(a, test)
    assert np.array_equal(out, ref)


def test_isin_nan_is_false_like_numpy():
    x = gr.array([float("nan")], dtype=gr.float64)
    t = gr.array([float("nan")], dtype=gr.float64)
    assert gr.isin(x, t).to_list() == [False]


def test_setdiff_union_setxor_match_numpy_float_nan():
    a = np.array([1.0, np.nan], dtype=np.float64)
    b = np.array([np.nan], dtype=np.float64)
    ga = gr.array(a.tolist(), dtype=gr.float64)
    gb = gr.array(b.tolist(), dtype=gr.float64)

    out_diff = np.array(gr.setdiff(ga, gb).to_list(), dtype=np.float64)
    out_union = np.array(gr.setunion(ga, gb).to_list(), dtype=np.float64)
    out_xor = np.array(gr.setxor(ga, gb).to_list(), dtype=np.float64)

    ref_diff = np.setdiff1d(a, b)
    ref_union = np.union1d(a, b)
    ref_xor = np.setxor1d(a, b)

    assert np.allclose(out_diff, ref_diff, equal_nan=True)
    assert np.allclose(out_union, ref_union, equal_nan=True)
    assert np.allclose(out_xor, ref_xor, equal_nan=True)


