import numpy as np

import grumpy as gr


def test_sort_dim_minus1_deep_3d_matches_python():
    a = [[[3, 1], [2]], [[5, 4, 6], []]]
    x = gr.array(a, dtype=gr.int32)
    out = x.sort(dim=-1).to_list()
    assert out == [[[1, 3], [2]], [[4, 5, 6], []]]


def test_argsort_dim_minus1_deep_3d_correctness():
    a = [[[3, 1], [2]], [[5, 4, 6], []]]
    x = gr.array(a, dtype=gr.int32)
    idx = x.argsort(dim=-1).to_list()
    # validate per-innermost-list indices produce sorted values
    for outer_i in range(len(a)):
        for inner_i in range(len(a[outer_i])):
            row = a[outer_i][inner_i]
            row_idx = idx[outer_i][inner_i]
            assert [row[j] for j in row_idx] == sorted(row)


def test_argmax_dim_minus1_deep_3d():
    a = [[[3, 1], [2]], [[5, 4, 6], []]]
    x = gr.array(a, dtype=gr.int32)
    out = x.argmax(dim=-1).to_list()
    assert out == [[0, 0], [2, None]]


def test_argmin_dim_minus1_deep_3d():
    a = [[[3, 1], [2]], [[5, 4, 6], []]]
    x = gr.array(a, dtype=gr.int32)
    out = x.argmin(dim=-1).to_list()
    assert out == [[1, 0], [1, None]]


def test_sort_dim0_on_ragged_2d_errors():
    x = gr.array([[3, 1, 2], [5], [2, 2, 1]], dtype=gr.int32)
    try:
        _ = x.sort(dim=0).to_list()
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_sort_default_dim_is_minus1_for_nested():
    x = gr.array([[3, 1, 2], [5], [2, 2, 1]], dtype=gr.int32)
    assert x.sort().to_list() == x.sort(dim=-1).to_list()


