import numpy as np
import pytest

import grumpy as gr


def test_einsum_dot_matches_numpy():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(2048,)).astype(np.float64)
    b = rng.normal(size=(2048,)).astype(np.float64)
    ga = gr.array(a.tolist(), dtype=gr.float64)
    gb = gr.array(b.tolist(), dtype=gr.float64)
    out = float(gr.einsum("i,i->", ga, gb))
    ref = float(np.einsum("i,i->", a, b))
    assert np.allclose(out, ref)


def test_einsum_matmul_matches_numpy():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(32, 48)).astype(np.float64)
    b = rng.normal(size=(48, 16)).astype(np.float64)
    ga = gr.array(a.tolist(), dtype=gr.float64)
    gb = gr.array(b.tolist(), dtype=gr.float64)
    out = np.array(gr.einsum("ij,jk->ik", ga, gb).to_list(), dtype=np.float64)
    ref = np.einsum("ij,jk->ik", a, b)
    assert np.allclose(out, ref)


def test_einsum_transpose_matches_numpy():
    rng = np.random.default_rng(0)
    a = rng.integers(-10, 10, size=(9, 7), dtype=np.int32)
    ga = gr.array(a.tolist(), dtype=gr.int32)
    out = np.array(gr.einsum("ij->ji", ga).to_list(), dtype=np.int32)
    ref = np.einsum("ij->ji", a)
    assert np.array_equal(out, ref)


def test_einsum_sum_rows_cols_total():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(11, 13)).astype(np.float64)
    ga = gr.array(a.tolist(), dtype=gr.float64)
    out_total = float(gr.einsum("ij->", ga))
    assert np.allclose(out_total, float(np.einsum("ij->", a)))
    out_rows = np.array(gr.einsum("ij->i", ga).to_list(), dtype=np.float64)
    out_cols = np.array(gr.einsum("ij->j", ga).to_list(), dtype=np.float64)
    assert np.allclose(out_rows, np.einsum("ij->i", a))
    assert np.allclose(out_cols, np.einsum("ij->j", a))


def test_einsum_frob_inner_matches_numpy():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(17, 19)).astype(np.float64)
    b = rng.normal(size=(17, 19)).astype(np.float64)
    ga = gr.array(a.tolist(), dtype=gr.float64)
    gb = gr.array(b.tolist(), dtype=gr.float64)
    out = float(gr.einsum("ij,ij->", ga, gb))
    ref = float(np.einsum("ij,ij->", a, b))
    assert np.allclose(out, ref)


def test_einsum_outer_matches_numpy():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(32,)).astype(np.float64)
    b = rng.normal(size=(17,)).astype(np.float64)
    ga = gr.array(a.tolist(), dtype=gr.float64)
    gb = gr.array(b.tolist(), dtype=gr.float64)
    out = np.array(gr.einsum("i,j->ij", ga, gb).to_list(), dtype=np.float64)
    ref = np.einsum("i,j->ij", a, b)
    assert np.allclose(out, ref)


def test_tensordot_axes0_outer_matches_numpy():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(64,)).astype(np.float64)
    b = rng.normal(size=(33,)).astype(np.float64)
    ga = gr.array(a.tolist(), dtype=gr.float64)
    gb = gr.array(b.tolist(), dtype=gr.float64)
    out = np.array(gr.tensordot(ga, gb, axes=0).to_list(), dtype=np.float64)
    ref = np.tensordot(a, b, axes=0)
    assert np.allclose(out, ref)


def test_tensordot_axes1_matmul_matches_numpy():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(16, 32)).astype(np.float64)
    b = rng.normal(size=(32, 9)).astype(np.float64)
    ga = gr.array(a.tolist(), dtype=gr.float64)
    gb = gr.array(b.tolist(), dtype=gr.float64)
    out = np.array(gr.tensordot(ga, gb, axes=1).to_list(), dtype=np.float64)
    ref = np.tensordot(a, b, axes=1)
    assert np.allclose(out, ref)


def test_tensordot_axes2_frob_inner_matches_numpy():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(11, 13)).astype(np.float64)
    b = rng.normal(size=(11, 13)).astype(np.float64)
    ga = gr.array(a.tolist(), dtype=gr.float64)
    gb = gr.array(b.tolist(), dtype=gr.float64)
    out = float(gr.tensordot(ga, gb, axes=2))
    ref = float(np.tensordot(a, b, axes=2))
    assert np.allclose(out, ref)


def test_einsum_errors_on_ragged():
    x = gr.array([[1, 2, 3], [4]], dtype=gr.int32)
    with pytest.raises(Exception):
        gr.einsum("ij->ji", x)


