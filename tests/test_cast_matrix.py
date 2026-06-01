"""Casting and promotion parity with NumPy rules (numeric dtypes)."""

from __future__ import annotations

import numpy as np
import pytest

import grumpy as gr

NUMERIC = [
    ("bool", gr.bool_, np.bool_),
    ("int8", gr.int8, np.int8),
    ("int16", gr.int16, np.int16),
    ("int32", gr.int32, np.int32),
    ("int64", gr.int64, np.int64),
    ("uint8", gr.uint8, np.uint8),
    ("uint16", gr.uint16, np.uint16),
    ("uint32", gr.uint32, np.uint32),
    ("uint64", gr.uint64, np.uint64),
    ("float16", gr.float16, np.float16),
    ("float32", gr.float32, np.float32),
    ("float64", gr.float64, np.float64),
]


@pytest.mark.parametrize("name,gr_dt,np_dt", NUMERIC)
@pytest.mark.parametrize("casting", ["safe", "same_kind", "unsafe"])
def test_can_cast_matches_numpy(name, gr_dt, np_dt, casting):
    for a_name, a_gr, a_np in NUMERIC:
        for b_name, b_gr, b_np in NUMERIC:
            expected = np.can_cast(a_np(), b_np(), casting=casting)
            assert gr.can_cast(a_gr, b_gr, casting=casting) == expected, (
                f"can_cast({a_name}->{b_name}, {casting})"
            )


@pytest.mark.parametrize("a_name,a_gr,a_np", NUMERIC)
@pytest.mark.parametrize("b_name,b_gr,b_np", NUMERIC)
def test_promote_types_matches_numpy(a_name, a_gr, a_np, b_name, b_gr, b_np):
    expected = np.promote_types(a_np(), b_np()).name
    got = gr.promote_types(a_gr, b_gr).name
    assert got == expected, f"promote_types({a_name}, {b_name})"


def _sample_values(from_name: str):
    if from_name == "bool":
        return [False, True, True, False]
    if from_name.startswith("float"):
        return [0.0, 1.5, 2.0, 3.0]
    return [0, 1, 2, 3]


@pytest.mark.parametrize("from_name,from_gr,from_np", NUMERIC)
@pytest.mark.parametrize("to_name,to_gr,to_np", NUMERIC)
def test_astype_safe_numeric_list_chain(from_name, from_gr, from_np, to_name, to_gr, to_np):
    if not np.can_cast(from_np(), to_np(), casting="safe"):
        return
    src = _sample_values(from_name)
    x = gr.array(src, dtype=from_gr)
    y = x.astype(to_gr)
    assert y.dtype.name == to_name
    ref = np.array(src, dtype=from_np()).astype(to_np()).tolist()
    if to_name.startswith("float"):
        assert y.to_list() == pytest.approx(ref)
    else:
        assert y.to_list() == ref


def test_astype_preserves_nulls_list_chain():
    x = gr.array([1, None, 3], dtype=gr.int32)
    y = x.astype(gr.float64)
    assert y.to_list() == [1.0, None, 3.0]


def test_astype_preserves_nulls_union():
    x = gr.array([1, [None, 2], 3], dtype=gr.int32)
    y = x.astype(gr.float64)
    assert y.to_list() == [1.0, [None, 2.0], 3.0]


def test_astype_mixed_int32_float64_add():
    a = gr.array([[1, 2], [3, 4]], dtype=gr.int32)
    b = gr.array([[1.0, 2.0], [3.0, 4.0]], dtype=gr.float64)
    out = (a + b).to_list()
    assert out == [[2.0, 4.0], [6.0, 8.0]]


def test_astype_safe_rejects_narrowing():
    x = gr.array([1000], dtype=gr.int32)
    with pytest.raises(Exception, match="Cannot cast|Cast overflow|casting"):
        x.astype(gr.int8)


def test_astype_same_kind_float_narrow():
    x = gr.array([1.25, 2.75], dtype=gr.float64)
    y = x.astype(gr.float32, casting="same_kind")
    assert y.dtype.name == "float32"
    ref = np.array([1.25, 2.75], dtype=np.float64).astype(np.float32).tolist()
    assert y.to_list() == pytest.approx(ref)


def test_astype_unsafe_int_narrow_wrap():
    x = gr.array([256], dtype=gr.int32)
    y = x.astype(gr.int8, casting="unsafe")
    assert y.to_list() == [0]


def test_astype_char_to_string():
    x = gr.array(["a", "b"], dtype=gr.char)
    y = x.astype(gr.string)
    assert y.dtype.name == "string"
    assert y.to_list() == ["a", "b"]


def test_promote_types_rejects_string_mix():
    with pytest.raises(Exception, match="promote|string"):
        gr.promote_types(gr.int32, gr.string)
