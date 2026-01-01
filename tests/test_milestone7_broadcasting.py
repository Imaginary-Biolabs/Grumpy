import math

import numpy as np

import grumpy as gr


def test_broadcast_scalar_over_ragged_2d_int32():
    x = gr.array([[1, 2, 3], [4, 5]], dtype=gr.int32)
    assert (x + gr.array(1, dtype=gr.int32)).to_list() == [[2, 3, 4], [5, 6]]
    assert (gr.array(1, dtype=gr.int32) + x).to_list() == [[2, 3, 4], [5, 6]]


def test_broadcast_axis0_len1_over_2d():
    a = gr.array([[10, 20]], dtype=gr.int32)  # len==1 on axis0
    b = gr.array([[1, 2], [3, 4]], dtype=gr.int32)
    assert (a + b).to_list() == [[11, 22], [13, 24]]
    assert (b + a).to_list() == [[11, 22], [13, 24]]


def test_broadcast_per_row_len1_against_ragged_row_lengths():
    x = gr.array([[1, 2, 3], [4, 5]], dtype=gr.int32)
    y = gr.array([[10], [20]], dtype=gr.int32)
    # y row elements should broadcast to the respective row lengths
    assert (x + y).to_list() == [[11, 12, 13], [24, 25]]


def test_ragged2d_output_kernel_none_propagation():
    x = gr.array([[1, None, 3], [4, 5]], dtype=gr.int32)
    y = gr.array([[10], [20]], dtype=gr.int32)
    # Broadcast + None propagation
    assert (x + y).to_list() == [[11, None, 13], [24, 25]]


def test_broadcast_scalar_over_deep_ragged_float_with_none_nan():
    x = gr.array([[[[1.0, None], [np.nan]]], [[[]]]], dtype=gr.float64)
    y = gr.array(2.0, dtype=gr.float64)
    out = (x + y).to_list()
    assert out[0][0][0][0] == 3.0
    assert out[0][0][0][1] is None
    assert math.isnan(out[0][0][1][0])


