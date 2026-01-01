import pytest

import grumpy as gr


def test_neighbors_knn_dim0_loop_false():
    x = gr.array([[0.0, 0.0], [1.0, 0.0], [10.0, 0.0]], dtype=gr.float64)
    out = gr.neighbors(x, x, k=1, dim=0, loop=False).to_list()
    assert out == [[[0, 1]], [[1, 0]], [[2, 1]]]


def test_neighbors_radius_dim0_loop_false():
    x = gr.array([[0.0, 0.0], [1.0, 0.0], [10.0, 0.0]], dtype=gr.float64)
    out = gr.neighbors(x, x, radius=1.1, dim=0, loop=False).to_list()
    assert out == [[[0, 1]], [[1, 0]], []]


def test_neighbors_knn_dim1_grouped():
    x = gr.array([[[0.0, 0.0], [2.0, 0.0]], [[0.0, 0.0], [1.0, 0.0], [5.0, 0.0]]], dtype=gr.float64)
    out = gr.neighbors(x, x, k=1, dim=1, loop=False).to_list()
    assert out == [
        [[[0, 1]], [[1, 0]]],
        [[[0, 1]], [[1, 0]], [[2, 1]]],
    ]


def test_neighbors_requires_k_or_radius_exclusive():
    x = gr.array([[0.0, 0.0], [1.0, 0.0]], dtype=gr.float64)
    with pytest.raises(Exception):
        gr.neighbors(x, x, dim=0)
    with pytest.raises(Exception):
        gr.neighbors(x, x, k=1, radius=1.0, dim=0)


