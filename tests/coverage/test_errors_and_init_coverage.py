"""Coverage for errors.py and remaining __init__ / compiler / _docutil branches."""

from __future__ import annotations

import types

import pytest

import grumpy as gr
from grumpy import errors
from grumpy._docutil import inject
from grumpy.compiler import _fuse_elementwise_ops, _try_compile_to_ops


def test_format_grumpy_error_minimal():
    msg = errors.format_grumpy_error("TestCode", "summary only")
    assert msg == "grumpy.TestCode: summary only"


def test_format_grumpy_error_full():
    msg = errors.format_grumpy_error(
        "IoFailed",
        "bad path",
        cause="missing file",
        fix="check path",
        path="/tmp/x.gr",
    )
    assert "grumpy.IoFailed: bad path" in msg
    assert "cause: missing file" in msg
    assert "path: /tmp/x.gr" in msg
    assert "fix: check path" in msg


def test_raise_grumpy_error_default_exc():
    with pytest.raises(ValueError, match="grumpy.ArgumentInvalid"):
        errors.raise_grumpy_error("ArgumentInvalid", "bad value", fix="fix it")


def test_raise_grumpy_error_custom_exc():
    with pytest.raises(IndexError, match="grumpy.IndexOutOfBounds"):
        errors.raise_grumpy_error(
            "IndexOutOfBounds",
            "out of range",
            exc=IndexError,
        )


def test_arg_invalid_and_arg_one_of():
    with pytest.raises(ValueError, match="invalid argument 'x'"):
        errors.arg_invalid("x", "negative", fix="use a non-negative value")
    with pytest.raises(ValueError, match="expected one of"):
        errors.arg_one_of("mode", "bogus", ("a", "b"))


def test_index_out_of_range():
    with pytest.raises(IndexError, match="index 3 is out of range"):
        errors.index_out_of_range(3, 3)


def test_neighbors_invalid_gpu():
    x = gr.array([[0.0, 0.0], [1.0, 1.0]], dtype=gr.float64)
    with pytest.raises(ValueError, match="invalid argument 'gpu'"):
        gr.neighbors(x, x, k=1, gpu="maybe")


def test_binary_out_parameters():
    a = gr.array([[1, 2], [3, 4]], dtype=gr.int32)
    b = gr.array([[2, 3], [4, 5]], dtype=gr.int32)
    out = gr.zeros_like(a)
    assert gr.multiply(a, b, out).to_list() == [[2, 6], [12, 20]]
    out = gr.zeros_like(a)
    assert gr.add(a, b, out).to_list() == [[3, 5], [7, 9]]
    out = gr.zeros_like(a)
    assert gr.subtract(a, b, out).to_list() == [[-1, -1], [-1, -1]]


def test_save_generator_empty_and_batches(tmp_path):
    p = str(tmp_path / "gen.gr")
    with pytest.raises(ValueError, match="iterator produced no batches"):

        def empty():
            return
            yield  # pragma: no cover

        gr.save(empty(), p)

    df = gr.dataframe({"x": [1, 2]})

    def batches():
        yield df
        yield gr.dataframe({"x": [3]})

    gr.save(batches(), p, chunk_size=1)
    loaded = gr.load(p)
    assert loaded.to_dict() == gr.dataframe({"x": [1, 2, 3]}).to_dict()


def test_gpu_backend_callable():
    # Smoke-test the public wrapper; backend may be None without GPU.
    _ = gr.gpu_backend()


def test_try_compile_wrong_batch_arg_name():
    def f(x):
        x = x * 2
        return x

    assert _try_compile_to_ops(f) is None


def test_fuse_mul_scalar_sum_all():
    ops = [
        {"op": "mul_scalar", "value": 2, "is_int": True},
        {"op": "reduce", "reduce": "sum"},
    ]
    fused = _fuse_elementwise_ops(ops)
    assert len(fused) == 1
    assert fused[0]["op"] == "mul_scalar_sum_all"
    assert fused[0]["value"] == 2


def test_docutil_inject_missing_and_descriptor_paths():
    class Box:
        value = 1

    inject(Box, "missing", "doc text")
    inject(Box, "__doc__", "class doc")

    class WithProp:
        @property
        def item(self):
            return 0

    inject(WithProp, "item", "item doc")
    assert WithProp.item.__doc__ == "item doc"

    class WithGetSet:
        pass

    WithGetSet.tag = types.GetSetDescriptorType  # type: ignore[misc, assignment]

    class Host:
        __slots__ = ()

    desc = type.__dict__["__module__"]
    setattr(Host, "slot_attr", desc)
    inject(Host, "slot_attr", "slot doc")
    assert getattr(Host, "slot_attr").__doc__ == "slot doc"
