import numpy as np
import pytest

import grumpy as gr


@pytest.mark.skipif(not gr.gpu_available(), reason="no GPU backend")
def test_gpu_knn_dim0_matches_cpu():
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(128, 3)).astype(np.float64)
    x = gr.array(pts.tolist(), dtype=gr.float64)
    cpu = gr.neighbors(x, x, k=8, dim=0, loop=False, gpu=False).to_list()
    gpu = gr.neighbors(x, x, k=8, dim=0, loop=False, gpu="force").to_list()
    assert cpu == gpu


@pytest.mark.skipif(not gr.gpu_available(), reason="no GPU backend")
def test_gpu_knn_dim1_auto_skips_stream_sized_batch():
    """Auto should stay on CPU for typical stream batch sizes (32 proteins)."""
    rng = np.random.default_rng(1)
    pts = rng.normal(size=(32, 128, 3)).astype(np.float64)
    x = gr.array(pts.tolist(), dtype=gr.float64)
    cpu = gr.neighbors(x, x, k=16, dim=1, loop=False, gpu=False).to_list()
    auto = gr.neighbors(x, x, k=16, dim=1, loop=False, gpu="auto").to_list()
    assert cpu == auto


@pytest.mark.skipif(not gr.gpu_available(), reason="no GPU backend")
def test_gpu_knn_dim1_protein_like_matches_cpu():
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(32, 128, 3)).astype(np.float64)
    x = gr.array(pts.tolist(), dtype=gr.float64)
    cpu = gr.neighbors(x, x, k=16, dim=1, loop=False, gpu=False).to_list()
    gpu = gr.neighbors(x, x, k=16, dim=1, loop=False, gpu="force").to_list()
    assert cpu == gpu
