"""Structured error message quality tests."""

from __future__ import annotations

import re

import pytest

import grumpy as gr


def _assert_grumpy_error(exc: BaseException, code: str) -> str:
    msg = str(exc)
    assert msg.startswith(f"grumpy.{code}:"), msg
    assert "cause:" in msg, msg
    assert "fix:" in msg, msg
    return msg


def test_cast_safe_rejects_unsafe_conversion():
    a = gr.array([300], dtype=gr.int16)
    with pytest.raises(ValueError) as ei:
        a.astype(gr.int8, casting="safe")
    msg = _assert_grumpy_error(ei.value, "CastNotAllowed")
    assert "int16" in msg and "int8" in msg


def test_broadcast_union_outer_length_mismatch():
    a = gr.array([1, [2, 3], 4], dtype=gr.int64)
    b = gr.array([10, [20], 30, 40], dtype=gr.int64)
    with pytest.raises(ValueError) as ei:
        (a + b).to_list()
    _assert_grumpy_error(ei.value, "BroadcastFailed")


def test_reduce_dim_out_of_range():
    x = gr.array([[1, 2], [3, 4]], dtype=gr.int32)
    with pytest.raises(ValueError) as ei:
        x.sum(dim=5)
    msg = _assert_grumpy_error(ei.value, "ReduceDimInvalid")
    assert re.search(r"axis\s+5", msg)


def test_stream_batch_index_oob():
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "x.gr")
        gr.save(gr.array(list(range(10)), dtype=gr.int32), path)
        st = gr.stream(path, batch_size=4)
        with pytest.raises(IndexError) as ei:
            _ = st[99]
        _assert_grumpy_error(ei.value, "IndexOutOfBounds")


def test_stream_shuffle_requires_seed():
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "x.gr")
        gr.save(gr.array(list(range(4)), dtype=gr.int32), path)
        with pytest.raises(ValueError) as ei:
            gr.stream(path, batch_size=2, shuffle=True)
        _assert_grumpy_error(ei.value, "ArgumentInvalid")


def test_cat_dtype_mismatch():
    a = gr.array([1, 2], dtype=gr.int32)
    b = gr.array([3], dtype=gr.int64)
    with pytest.raises(ValueError) as ei:
        gr.cat([a, b], dim=0)
    msg = _assert_grumpy_error(ei.value, "DtypeMismatch")
    assert "int32" in msg and "int64" in msg
