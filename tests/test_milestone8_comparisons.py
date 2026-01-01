import numpy as np

import grumpy as gr


def test_comparisons_rectangular_match_numpy_int32():
    rng = np.random.default_rng(0)
    a = rng.integers(-10, 10, size=(32, 64), dtype=np.int32)
    b = rng.integers(-10, 10, size=(32, 64), dtype=np.int32)
    x = gr.array(a.tolist(), dtype=gr.int32)
    y = gr.array(b.tolist(), dtype=gr.int32)

    assert np.array_equal(np.array(x.equal(y).to_list(), dtype=np.bool_), a == b)
    assert np.array_equal(np.array(x.not_equal(y).to_list(), dtype=np.bool_), a != b)
    assert np.array_equal(np.array(x.less(y).to_list(), dtype=np.bool_), a < b)
    assert np.array_equal(np.array(x.less_equal(y).to_list(), dtype=np.bool_), a <= b)
    assert np.array_equal(np.array(x.greater(y).to_list(), dtype=np.bool_), a > b)
    assert np.array_equal(np.array(x.greater_equal(y).to_list(), dtype=np.bool_), a >= b)


def test_predicates_float64_match_numpy():
    a = np.array([[1.0, np.nan, np.inf, -np.inf]], dtype=np.float64)
    x = gr.array(a.tolist(), dtype=gr.float64)
    assert np.array_equal(np.array(x.isnan().to_list(), dtype=np.bool_), np.isnan(a))
    assert np.array_equal(np.array(x.isinf().to_list(), dtype=np.bool_), np.isinf(a))
    assert np.array_equal(np.array(x.isfinite().to_list(), dtype=np.bool_), np.isfinite(a))


