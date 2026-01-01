import numpy as np

import grumpy as gr


def test_construct_and_to_list_ragged_int():
    x = gr.array([[1, 2, 3], [4, 5]], dtype=gr.int32)
    assert x.dtype.name == "int32"
    assert x.to_list() == [[1, 2, 3], [4, 5]]


def test_dtype_inference_int_and_float_and_bool():
    a = gr.array([1, 2, 3])
    assert a.dtype.name == "int64"

    b = gr.array([1, 2.0, 3])
    assert b.dtype.name == "float64"

    c = gr.array([True, False, True])
    assert c.dtype.name == "bool"


def test_char_dtype_and_validation():
    x = gr.array(["a", "b", None], dtype=gr.char)
    assert x.to_list() == ["a", "b", None]

    try:
        gr.array(["ab"], dtype=gr.char)
        assert False, "Expected error for multi-character string"
    except ValueError:
        pass


def test_variable_depth_shape_and_nshape_example():
    y = gr.array([[1, 2, 3], [[None, 5], [6]]], dtype=gr.int64)

    assert y.shape(0) == 2
    assert y.shape(1).to_list() == [3, 2]

    s2 = y.shape(2)
    assert hasattr(s2, "to_list")
    assert s2.to_list() == [[], [2, 1]]

    ns2 = y.nshape(2)
    assert ns2.to_list() == [[], [1, 1]]


def test_astype_casting():
    x = gr.array([1, 2, None], dtype=gr.int32)
    y = x.astype(gr.float64)
    assert y.dtype.name == "float64"
    assert y.to_list() == [1.0, 2.0, None]


def test_to_numpy_rectangular_typed():
    x = gr.array([[1, 2], [3, 4]], dtype=gr.int32)
    arr = x.to_numpy()
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.int32
    assert arr.shape == (2, 2)
    assert arr.tolist() == [[1, 2], [3, 4]]


def test_to_numpy_ragged_object_fallback():
    x = gr.array([[1, 2, 3], [4, 5]], dtype=gr.int32)
    arr = x.to_numpy()
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == object
    assert arr.tolist() == [[1, 2, 3], [4, 5]]


def test_to_numpy_with_nulls_object_fallback():
    x = gr.array([1, None, 2], dtype=gr.int32)
    arr = x.to_numpy()
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == object
    assert arr.tolist() == [1, None, 2]


