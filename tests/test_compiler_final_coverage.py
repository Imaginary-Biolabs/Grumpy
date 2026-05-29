"""Cover last compiler/stream branches."""

from __future__ import annotations

import textwrap

import pytest

import grumpy as gr
from grumpy.compiler import (
    PipelineInfo,
    _try_compile,
    _try_compile_to_ops,
)
from grumpy.stream import StreamApply


def test_try_compile_to_ops_no_source(monkeypatch):
    import grumpy.compiler as cm

    monkeypatch.setattr(cm, "_get_source", lambda _fn: None)

    def f(batch):
        batch = batch + 1
        return batch

    assert _try_compile_to_ops(f) is None


def test_try_compile_parse_error(monkeypatch):
    import grumpy.compiler as cm

    monkeypatch.setattr(cm, "_get_source", lambda _fn: "def (((bad")

    def f(batch):
        return batch

    assert _try_compile_to_ops(f) is None
    assert _try_compile(f).error


def test_try_compile_empty_body():
    src = "def f(batch):\n    pass\n"
    import os
    import tempfile

    fd, path = tempfile.mkstemp(suffix=".py")
    with os.fdopen(fd, "w") as fh:
        fh.write(src)
    ns: dict = {}
    import builtins

    exec(builtins.compile(src, path, "exec"), ns, ns)
    assert _try_compile(ns["f"]).error


def test_stream_unsupported_op_rust_warns(tmp_path, monkeypatch):
    x = gr.array([1.0, 2.0], dtype=gr.float64)
    p = tmp_path / "a.gr"
    gr.save(x, str(p))
    st = gr.stream(str(p), batch_size=1)

    def t(batch):
        batch = batch * 2.0
        return batch

    sa = st.apply(t, cpu=2, compile=True, scheduler="rust")
    import grumpy.compiler as cm

    fake_info = PipelineInfo(
        run_all=lambda b: b,
        fully_compiled=True,
        fused_ops=[{"op": "not_a_real_op"}],
    )

    def fake_info_fn(_fns):
        return fake_info

    monkeypatch.setattr(cm, "compile_pipeline_info", fake_info_fn)
    with pytest.warns(UserWarning, match="not supported by Rust scheduling"):
        list(sa)


def test_stream_parallel_refill(tmp_path):
    x = gr.array([[1], [2], [3], [4]], dtype=gr.int32)
    p = tmp_path / "a.gr"
    gr.save(x, str(p))
    st = gr.stream(str(p), batch_size=1)
    out = list(st.apply(lambda b: b, cpu=2, prefetch=2))
    assert len(out) == 4
