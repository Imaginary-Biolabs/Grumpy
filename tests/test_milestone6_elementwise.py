import numpy as np

import grumpy as gr


def test_elementwise_mul_ragged():
    x = gr.array([[1, 2, 3], [4, 5]], dtype=gr.int32)
    y = gr.array([[2, 2, 2], [2, 2]], dtype=gr.int32)
    z = x * y
    assert z.to_list() == [[2, 4, 6], [8, 10]]


def test_elementwise_add_with_none_propagates_none():
    x = gr.array([[1, None, 3], [4]], dtype=gr.int32)
    y = gr.array([[10, 20, 30], [40]], dtype=gr.int32)
    z = x + y
    assert z.to_list() == [[11, None, 33], [44]]


def test_elementwise_structure_mismatch_raises():
    x = gr.array([[1, 2, 3], [4, 5]], dtype=gr.int32)
    y = gr.array([[1, 2], [3, 4, 5]], dtype=gr.int32)
    try:
        _ = x + y
        assert False, "expected structure mismatch error"
    except ValueError:
        pass


def test_elementwise_div_int_produces_float64():
    x = gr.array([[1, 2, 3], [4, 5]], dtype=gr.int32)
    y = gr.array([[2, 2, 2], [2, 2]], dtype=gr.int32)
    z = x / y
    assert z.dtype.name == "float64"
    assert z.to_list() == [[0.5, 1.0, 1.5], [2.0, 2.5]]


def test_mod_matches_numpy_for_signed_ints():
    a = gr.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], dtype=gr.int32)
    b = gr.array([3] * 11, dtype=gr.int32)
    out = (a % b).to_list()
    np_out = (np.array(a.to_list(), dtype=np.int32) % 3).tolist()
    assert out == np_out


