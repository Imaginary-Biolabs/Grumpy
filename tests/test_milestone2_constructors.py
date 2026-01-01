import numpy as np

import grumpy as gr


def test_cat_dim0_simple():
    a = gr.array([1, 2], dtype=gr.int32)
    b = gr.array([3], dtype=gr.int32)
    c = gr.cat([a, b], dim=0)
    assert c.to_list() == [1, 2, 3]


def test_cat_dim1_example_with_variable_depth_union():
    x = gr.array([[1, 2, 3], [4, 5]], dtype=gr.int32)
    y = gr.array([[1, 2, 3], [[None, 5], [6]]], dtype=gr.int32)
    z = gr.cat([x, y], dim=1)
    assert z.to_list() == [[1, 2, 3, 1, 2, 3], [4, 5, [None, 5], [6]]]


def test_cat_requires_same_dtype():
    a = gr.array([1, 2], dtype=gr.int32)
    b = gr.array([1, 2], dtype=gr.int64)
    try:
        gr.cat([a, b], dim=0)
        assert False, "expected dtype mismatch error"
    except ValueError:
        pass


def test_zeros_like_and_ones_like_preserve_structure():
    x = gr.array([[1, None, 3], [4]], dtype=gr.int64)
    z = gr.zeros_like(x)
    o = gr.ones_like(x)
    assert z.to_list() == [[0, 0, 0], [0]]
    assert o.to_list() == [[1, 1, 1], [1]]


def test_full_like_dtype_override():
    x = gr.array([[1, 2, 3], [4, 5]], dtype=gr.int32)
    y = gr.full_like(x, 2.5, dtype=gr.float32)
    assert y.dtype.name == "float32"
    assert y.to_list() == [[2.5, 2.5, 2.5], [2.5, 2.5]]


def test_ones_like_bool():
    x = gr.array([[True, False], []], dtype=gr.bool_)
    o = gr.ones_like(x)
    assert o.to_list() == [[True, True], []]


def test_to_numpy_still_works_with_cat_result():
    x = gr.array([[1, 2], [3, 4]], dtype=gr.int32)
    y = gr.cat([x, x], dim=0)
    arr = y.to_numpy()
    assert isinstance(arr, np.ndarray)
    # concatenation along dim=0 preserves rectangularity here
    assert arr.dtype == np.int32
    assert arr.shape == (4, 2)
    assert arr.tolist() == [[1, 2], [3, 4], [1, 2], [3, 4]]


