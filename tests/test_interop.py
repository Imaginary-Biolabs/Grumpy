import numpy as np
import pytest

import grumpy as gr


def test_to_numpy_1d_and_2d():
    x = gr.array([1, 2, 3], dtype=gr.int32)
    arr = x.to_numpy()
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.int32
    assert arr.shape == (3,)
    assert arr.tolist() == [1, 2, 3]

    y = gr.array([[1, 2], [3, 4]], dtype=gr.int32)
    arr2 = y.to_numpy()
    assert arr2.shape == (2, 2)
    assert arr2.tolist() == [[1, 2], [3, 4]]


def test_to_numpy_3d():
    data = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    x = gr.array(data, dtype=gr.int32)
    arr = x.to_numpy()
    assert arr.shape == (2, 2, 2)
    assert arr.tolist() == data


def test_to_numpy_rejects_ragged():
    x = gr.array([[1, 2, 3], [4, 5]], dtype=gr.int32)
    with pytest.raises(ValueError, match="ragged|rectangular|ShapeMismatch"):
        x.to_numpy()


def test_to_numpy_rejects_nulls():
    x = gr.array([1, None, 2], dtype=gr.int32)
    with pytest.raises(ValueError, match="null|rectangular|ShapeMismatch"):
        x.to_numpy()


def test_from_numpy_roundtrip():
    np_arr = np.arange(12, dtype=np.int32).reshape(3, 4)
    x = gr.from_numpy(np_arr)
    assert x.dtype.name == "int32"
    assert x.to_numpy().tolist() == np_arr.tolist()


def test_from_numpy_requires_c_contiguous():
    np_arr = np.arange(12, dtype=np.int32).reshape(3, 4)
    fortran = np.asfortranarray(np_arr)
    with pytest.raises(ValueError, match="C-contiguous"):
        gr.from_numpy(fortran)


def test_is_rectangular():
    rect = gr.array([[1, 2], [3, 4]], dtype=gr.int32)
    ragged = gr.array([[1, 2, 3], [4, 5]], dtype=gr.int32)
    assert gr.is_rectangular(rect)
    assert not gr.is_rectangular(ragged)


def test_open_batch_slice_to_numpy():
    # OffsetView batches from gr.open should convert without gather when contiguous.
    import tempfile
    import os

    df = gr.dataframe({"x": gr.from_numpy(np.arange(20, dtype=np.int32).reshape(10, 2))})
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ds")
        gr.save(df, path)
        with gr.open(path) as session:
            col = session["x"]
            batch = col[2:6]
            arr = batch.to_numpy()
            assert arr.shape == (4, 2)
            assert arr.dtype == np.int32


@pytest.mark.parametrize("framework", ["torch", "tensorflow"])
def test_to_framework_and_back(framework):
    pytest.importorskip(framework)
    x = gr.array([[1.0, 2.0], [3.0, 4.0]], dtype=gr.float32)
    if framework == "torch":
        import torch

        t = x.to_torch()
        assert isinstance(t, torch.Tensor)
        assert t.shape == (2, 2)
        y = gr.from_torch(t)
    else:
        import tensorflow as tf

        t = x.to_tensorflow()
        assert isinstance(t, tf.Tensor)
        assert tuple(t.shape.as_list()) == (2, 2)
        y = gr.from_tensorflow(t)
    assert y.to_numpy().tolist() == [[1.0, 2.0], [3.0, 4.0]]
