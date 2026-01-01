import numpy as np

import grumpy as gr


def _as_np(x):
    # Grumpy may produce None in ragged outputs; these tests only use rectangular all-valid inputs.
    return np.array(x.to_list())


def test_unary_float64_matches_numpy_rectangular():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(32, 64)).astype(np.float64)
    x = gr.array(a.tolist(), dtype=gr.float64)

    for name, fn in [
        ("sin", np.sin),
        ("cos", np.cos),
        ("tan", np.tan),
        ("exp", np.exp),
        ("log", np.log),
        ("log10", np.log10),
        ("log2", np.log2),
        ("sqrt", np.sqrt),
        ("abs", np.abs),
        ("sign", np.sign),
        ("floor", np.floor),
        ("ceil", np.ceil),
        ("round", np.round),
        ("reciprocal", np.reciprocal),
    ]:
        gx = getattr(x, name)()
        out = _as_np(gx)
        ref = fn(a)
        assert np.allclose(out, ref, equal_nan=True), name


def test_unary_int32_promotes_for_trig_log_exp_sqrt():
    a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    x = gr.array(a.tolist(), dtype=gr.int32)
    gx = x.sin()
    out = _as_np(gx)
    assert out.dtype == np.float64
    assert np.allclose(out, np.sin(a), equal_nan=True)


