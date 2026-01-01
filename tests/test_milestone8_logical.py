import numpy as np

import grumpy as gr


def test_logical_ops_match_numpy_bool_rectangular():
    a = np.array([[True, False, True], [False, False, True]], dtype=np.bool_)
    b = np.array([[True, True, False], [False, True, True]], dtype=np.bool_)
    x = gr.array(a.tolist(), dtype=gr.bool_)
    y = gr.array(b.tolist(), dtype=gr.bool_)

    assert np.array_equal(np.array(x.logical_and(y).to_list(), dtype=np.bool_), np.logical_and(a, b))
    assert np.array_equal(np.array(x.logical_or(y).to_list(), dtype=np.bool_), np.logical_or(a, b))
    assert np.array_equal(np.array(x.logical_xor(y).to_list(), dtype=np.bool_), np.logical_xor(a, b))
    assert np.array_equal(np.array(x.logical_not().to_list(), dtype=np.bool_), np.logical_not(a))


