import math

import numpy as np

import grumpy as gr


def test_deep_ragged_5d_float_none_nan_flatten_and_elementwise():
    # 5D-ish structure: outer 2, then ragged 0..2, then ragged 0..2, then ragged, then scalars.
    x = gr.array(
        [
            [
                [
                    [[1.0, None], [np.nan]],
                    [],
                ],
                [
                    [[2.0], [3.0, np.nan, None]],
                ],
            ],
            [
                [],
                [
                    [[None]],
                    [[4.0, 5.0]],
                ],
            ],
        ],
        dtype=gr.float64,
    )

    # Default flatten: should linearize all list axes and preserve None + NaN.
    flat = x.flatten().to_list()
    assert flat[0] == 1.0
    assert flat[1] is None
    assert math.isnan(flat[2])
    assert flat.count(None) == 3
    assert sum(1 for v in flat if isinstance(v, float) and math.isnan(v)) == 2

    # Elementwise should propagate None and keep NaN as a value.
    y = x + x
    y_flat = y.flatten().to_list()
    # 1.0 + 1.0
    assert y_flat[0] == 2.0
    # None + None => None
    assert y_flat[1] is None
    # NaN + NaN => NaN
    assert math.isnan(y_flat[2])


def test_deep_ragged_variable_depth_union_mixed_scalars_and_lists():
    # Variable depth: some elements are scalars, some are lists-of-lists.
    # This should create a union layout internally; operations should still be correct.
    x = gr.array(
        [
            [1, 2, [None, 3]],
            4,
            [[5], [6, None]],
        ],
        dtype=gr.int32,
    )
    assert x.to_list() == [[1, 2, [None, 3]], 4, [[5], [6, None]]]

    flat = x.flatten().to_list()
    assert flat == [1, 2, None, 3, 4, 5, 6, None]


