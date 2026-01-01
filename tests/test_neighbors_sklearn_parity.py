import numpy as np
import pytest

import grumpy as gr


def _sklearn_knn_indices(points_f64: np.ndarray, k: int, loop: bool):
    sklearn = pytest.importorskip("sklearn")
    from sklearn.neighbors import NearestNeighbors

    assert points_f64.ndim == 2
    nn = NearestNeighbors(n_neighbors=(k if loop else k + 1), algorithm="brute", metric="euclidean")
    nn.fit(points_f64)
    idx = nn.kneighbors(points_f64, return_distance=False)
    if not loop:
        # Drop self for each row.
        idx2 = np.empty((idx.shape[0], k), dtype=idx.dtype)
        for i in range(idx.shape[0]):
            row = idx[i]
            # self can appear anywhere if there are duplicates; our test data avoids duplicates.
            row = row[row != i]
            idx2[i] = row[:k]
        idx = idx2
    return idx


def test_neighbors_knn_dim0_matches_sklearn_loop_false():
    # Use integer coordinates cast to float64 for stable exact comparisons.
    rng = np.random.default_rng(0)
    pts = rng.integers(-1000, 1000, size=(64, 3), dtype=np.int64).astype(np.float64)
    # Ensure uniqueness (avoid sklearn self-position ambiguity with duplicates)
    pts = np.unique(pts, axis=0)
    pts = pts[:64]
    k = 5

    x = gr.array(pts.tolist(), dtype=gr.float64)
    out = np.array(gr.neighbors(x, x, k=k, dim=0, loop=False).to_list(), dtype=np.int64)
    # out shape: (n, k, 2) with [src, dst]
    assert out.shape == (pts.shape[0], k, 2)
    assert np.all(out[:, :, 0] == np.arange(pts.shape[0])[:, None])
    ref = _sklearn_knn_indices(pts, k=k, loop=False)
    assert np.array_equal(out[:, :, 1], ref)


def test_neighbors_knn_dim0_matches_sklearn_loop_true_includes_self():
    rng = np.random.default_rng(1)
    pts = rng.integers(-1000, 1000, size=(40, 2), dtype=np.int64).astype(np.float64)
    pts = np.unique(pts, axis=0)[:40]
    k = 3

    x = gr.array(pts.tolist(), dtype=gr.float64)
    out = np.array(gr.neighbors(x, x, k=k, dim=0, loop=True).to_list(), dtype=np.int64)
    assert out.shape == (pts.shape[0], k, 2)
    assert np.all(out[:, :, 0] == np.arange(pts.shape[0])[:, None])
    ref = _sklearn_knn_indices(pts, k=k, loop=True)
    assert np.array_equal(out[:, :, 1], ref)
    # self should be the first neighbor (distance 0) for unique points
    assert np.array_equal(out[:, 0, 1], np.arange(pts.shape[0]))


def test_neighbors_knn_dim1_grouped_matches_sklearn_per_group_loop_false():
    rng = np.random.default_rng(2)
    groups = [
        rng.integers(-50, 50, size=(8, 2), dtype=np.int64).astype(np.float64),
        rng.integers(-50, 50, size=(5, 2), dtype=np.int64).astype(np.float64),
        rng.integers(-50, 50, size=(11, 2), dtype=np.int64).astype(np.float64),
    ]
    # make points unique per group to avoid self ambiguity
    groups = [np.unique(g, axis=0) for g in groups]
    k = 2
    x = gr.array([g.tolist() for g in groups], dtype=gr.float64)

    out = gr.neighbors(x, x, k=k, dim=1, loop=False).to_list()
    for gi, g in enumerate(groups):
        ref = _sklearn_knn_indices(g, k=k, loop=False)
        # out[gi] shape: points -> k -> [src,dst] (local indices)
        assert len(out[gi]) == g.shape[0]
        for i in range(g.shape[0]):
            assert [e[0] for e in out[gi][i]] == [i] * k
            assert [e[1] for e in out[gi][i]] == ref[i].tolist()


