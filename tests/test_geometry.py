import numpy as np

import grumpy as gr


def test_pairwise_distances_small():
    x = gr.array([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]], dtype=gr.float64)
    out = np.array(gr.pairwise_distances(x, dim=1).to_list(), dtype=np.float64)
    expected = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, np.sqrt(2.0)], [1.0, np.sqrt(2.0), 0.0]])
    np.testing.assert_allclose(out[0], expected, rtol=0, atol=1e-12)


def test_grid_pool_counts():
    x = gr.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], dtype=gr.float64)
    out = gr.grid_pool(
        x,
        grid_size=(2, 2, 2),
        origin=(0.0, 0.0, 0.0),
        voxel_size=(1.0, 1.0, 1.0),
        dim=1,
    ).to_list()
    assert out[0][0] == 1.0  # (0,0,0)
    assert out[0][1] == 1.0  # (1,0,0)
    assert out[0][2] == 1.0  # (0,1,0)
    assert sum(out[0]) == 3.0
