import numpy as np

import grumpy as gr


def test_var_std_match_numpy_int32_rectangular_dim1():
    rng = np.random.default_rng(0)
    a = rng.integers(-1000, 1000, size=(32, 64), dtype=np.int32)
    x = gr.array(a.tolist(), dtype=gr.int32)
    out_var = np.array(x.var(dim=1, ddof=0).to_list(), dtype=np.float64)
    out_std = np.array(x.std(dim=1, ddof=0).to_list(), dtype=np.float64)
    ref_var = a.var(axis=1, ddof=0)
    ref_std = a.std(axis=1, ddof=0)
    assert np.allclose(out_var, ref_var)
    assert np.allclose(out_std, ref_std)


def test_var_std_match_numpy_float32_rectangular_dim1_dtype():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(16, 32)).astype(np.float32)
    x = gr.array(a.tolist(), dtype=gr.float32)
    out_var = np.array(x.var(dim=1, ddof=0).to_list(), dtype=np.float32)
    out_std = np.array(x.std(dim=1, ddof=0).to_list(), dtype=np.float32)
    ref_var = a.var(axis=1, ddof=0)
    ref_std = a.std(axis=1, ddof=0)
    assert out_var.dtype == np.float32
    assert out_std.dtype == np.float32
    assert np.allclose(out_var, ref_var, rtol=1e-5, atol=1e-6)
    assert np.allclose(out_std, ref_std, rtol=1e-5, atol=1e-6)


def test_nanvar_nanstd_ignore_nan_float64():
    a = np.array([[1.0, np.nan, 3.0], [2.0, 4.0, np.nan]], dtype=np.float64)
    x = gr.array(a.tolist(), dtype=gr.float64)
    out_var = np.array(x.nanvar(dim=1, ddof=0).to_list(), dtype=np.float64)
    out_std = np.array(x.nanstd(dim=1, ddof=0).to_list(), dtype=np.float64)
    assert np.allclose(out_var, np.nanvar(a, axis=1))
    assert np.allclose(out_std, np.nanstd(a, axis=1))


def test_quantile_percentile_match_numpy_float64_dim1():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(16, 64)).astype(np.float64)
    x = gr.array(a.tolist(), dtype=gr.float64)
    q = 0.25
    out_q = np.array(x.quantile(q, dim=1).to_list(), dtype=np.float64)
    ref_q = np.quantile(a, q, axis=1, method="linear")
    assert np.allclose(out_q, ref_q, equal_nan=True)

    p = 25.0
    out_p = np.array(x.percentile(p, dim=1).to_list(), dtype=np.float64)
    ref_p = np.percentile(a, p, axis=1, method="linear")
    assert np.allclose(out_p, ref_p, equal_nan=True)


def test_nanquantile_ignores_nan():
    a = np.array([[1.0, np.nan, 3.0]], dtype=np.float64)
    x = gr.array(a.tolist(), dtype=gr.float64)
    out = np.array(x.nanquantile(0.5, dim=1).to_list(), dtype=np.float64)
    ref = np.nanquantile(a, 0.5, axis=1, method="linear")
    assert np.allclose(out, ref, equal_nan=True)


