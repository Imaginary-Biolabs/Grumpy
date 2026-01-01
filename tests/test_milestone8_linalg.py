import numpy as np
import pytest

import grumpy as gr


def test_dot_inner_matches_numpy_float64():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(1000,)).astype(np.float64)
    b = rng.normal(size=(1000,)).astype(np.float64)
    ga = gr.array(a.tolist(), dtype=gr.float64)
    gb = gr.array(b.tolist(), dtype=gr.float64)
    assert np.allclose(gr.dot(ga, gb), float(np.dot(a, b)))
    assert np.allclose(gr.inner(ga, gb), float(np.inner(a, b)))


def test_dot_matches_numpy_int32_value():
    a = np.arange(1000, dtype=np.int32)
    b = (np.arange(1000, dtype=np.int32) % 7) - 3
    ga = gr.array(a.tolist(), dtype=gr.int32)
    gb = gr.array(b.tolist(), dtype=gr.int32)
    assert int(gr.dot(ga, gb)) == int(np.dot(a, b))


def test_outer_matches_numpy_float64():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(128,)).astype(np.float64)
    b = rng.normal(size=(64,)).astype(np.float64)
    ga = gr.array(a.tolist(), dtype=gr.float64)
    gb = gr.array(b.tolist(), dtype=gr.float64)
    out = np.array(gr.outer(ga, gb).to_list(), dtype=np.float64)
    ref = np.outer(a, b)
    assert np.allclose(out, ref)


def test_trace_matches_numpy_rectangular():
    rng = np.random.default_rng(0)
    a = rng.integers(-10, 10, size=(32, 17), dtype=np.int32)
    ga = gr.array(a.tolist(), dtype=gr.int32)
    assert int(gr.trace(ga)) == int(np.trace(a))


def test_norm_matches_numpy_float64():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(1024,)).astype(np.float64)
    ga = gr.array(a.tolist(), dtype=gr.float64)
    assert np.allclose(gr.norm(ga), float(np.linalg.norm(a)))


def test_cross_vec3_matches_numpy():
    a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    b = np.array([-3.0, 0.5, 7.0], dtype=np.float64)
    ga = gr.array(a.tolist(), dtype=gr.float64)
    gb = gr.array(b.tolist(), dtype=gr.float64)
    out = np.array(gr.cross(ga, gb).to_list(), dtype=np.float64)
    ref = np.cross(a, b)
    assert np.allclose(out, ref)


def test_trace_errors_on_ragged():
    x = gr.array([[1, 2, 3], [4]], dtype=gr.int32)
    with pytest.raises(Exception):
        gr.trace(x)


def test_det_inv_matches_numpy_float64():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(8, 8)).astype(np.float64)
    # make it well-conditioned-ish
    a = a + np.eye(8, dtype=np.float64) * 0.5
    ga = gr.array(a.tolist(), dtype=gr.float64)
    out_det = float(gr.det(ga))
    ref_det = float(np.linalg.det(a))
    assert np.allclose(out_det, ref_det, rtol=1e-9, atol=1e-9)

    out_inv = np.array(gr.inv(ga).to_list(), dtype=np.float64)
    ref_inv = np.linalg.inv(a)
    assert np.allclose(out_inv, ref_inv, rtol=1e-9, atol=1e-9)


def test_inv_singular_raises():
    a = np.array([[1.0, 2.0], [1.0, 2.0]], dtype=np.float64)
    ga = gr.array(a.tolist(), dtype=gr.float64)
    with pytest.raises(Exception):
        gr.inv(ga)


