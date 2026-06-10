"""Cover last compiler branches."""

from __future__ import annotations

import pytest

import grumpy as gr
from grumpy.compiler import (
    PipelineInfo,
    _try_compile,
    _try_compile_to_ops,
)


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


def test_compile_pipeline_info_fused_ops():
    def t(batch):
        batch = batch * 2
        return batch

    info = gr.compiler.compile_pipeline_info([t])
    assert info.fully_compiled
    assert info.fused_ops
