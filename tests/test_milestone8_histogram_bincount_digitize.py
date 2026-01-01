import numpy as np

import grumpy as gr


def test_bincount_matches_numpy_counts():
    rng = np.random.default_rng(0)
    x = rng.integers(0, 100, size=(10_000,), dtype=np.int32)
    gx = gr.array(x.tolist(), dtype=gr.int32)
    out = np.array(gr.bincount(gx).to_list(), dtype=np.int64)
    ref = np.bincount(x)
    assert np.array_equal(out, ref)


def test_bincount_matches_numpy_with_weights():
    rng = np.random.default_rng(0)
    x = rng.integers(0, 50, size=(5000,), dtype=np.int32)
    w = rng.normal(size=(5000,)).astype(np.float64)
    gx = gr.array(x.tolist(), dtype=gr.int32)
    gw = gr.array(w.tolist(), dtype=gr.float64)
    out = np.array(gr.bincount(gx, weights=gw).to_list(), dtype=np.float64)
    ref = np.bincount(x, weights=w)
    assert np.allclose(out, ref)


def test_digitize_matches_numpy_with_nan_and_out_of_range():
    x = np.array([np.nan, -1.0, 0.1, 1.9], dtype=np.float64)
    bins = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    gx = gr.array(x.tolist(), dtype=gr.float64)
    gb = gr.array(bins.tolist(), dtype=gr.float64)
    out = np.array(gr.digitize(gx, gb, right=False).to_list(), dtype=np.int64)
    ref = np.digitize(x, bins, right=False)
    assert np.array_equal(out, ref)


def test_histogram_matches_numpy_basic():
    x = np.array([np.nan, -1.0, 0.1, 1.9], dtype=np.float64)
    gx = gr.array(x.tolist(), dtype=gr.float64)
    h, edges = gr.histogram(gx, bins=2, range=(0.0, 2.0), density=False)
    out_h = np.array(h.to_list(), dtype=np.float64)
    out_e = np.array(edges.to_list(), dtype=np.float64)
    ref_h, ref_e = np.histogram(x, bins=2, range=(0.0, 2.0), density=False)
    assert np.allclose(out_h, ref_h)
    assert np.allclose(out_e, ref_e)


def test_histogram_density_matches_numpy():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(20000,)).astype(np.float64)
    gx = gr.array(x.tolist(), dtype=gr.float64)
    h, edges = gr.histogram(gx, bins=20, range=(-3.0, 3.0), density=True)
    out_h = np.array(h.to_list(), dtype=np.float64)
    out_e = np.array(edges.to_list(), dtype=np.float64)
    ref_h, ref_e = np.histogram(x, bins=20, range=(-3.0, 3.0), density=True)
    assert np.allclose(out_e, ref_e)
    assert np.allclose(out_h, ref_h, rtol=1e-5, atol=1e-7)


