"""Tests for session-scoped chunk LRU on ``gr.open``."""

from __future__ import annotations

import pytest

import grumpy as gr


def _range_df(n: int):
    return gr.dataframe({"value": list(range(n))}, schema=["molecule"])


def test_chunk_cache_hit_same_chunk(tmp_path):
    path = str(tmp_path / "df.gr")
    gr.save(_range_df(32), path, chunk_size=8)

    with gr.open(path, cache="chunks", chunk_budget_mb=16) as h:
        gr._core.reset_io_bytes_read()
        _ = h[0:4].to_dict()
        first = gr._core.io_bytes_read()

        gr._core.reset_io_bytes_read()
        _ = h[4:8].to_dict()
        same_chunk = gr._core.io_bytes_read()

        gr._core.reset_io_bytes_read()
        _ = h[0:4].to_dict()
        repeat = gr._core.io_bytes_read()

        bytes_used, n_chunks = gr._core.io_cache_stats(path)
        assert n_chunks > 0
        assert same_chunk == 0
        assert repeat == 0
        assert first > 0


def test_chunk_lru_eviction_respects_budget(tmp_path):
    path = str(tmp_path / "df.gr")
    gr.save(_range_df(128), path, chunk_size=4)
    budget_mb = 1
    budget_bytes = budget_mb * 1024 * 1024

    with gr.open(path, cache="chunks", chunk_budget_mb=budget_mb) as h:
        for start in range(0, 128, 4):
            _ = h[start : start + 4].to_dict()
        bytes_used, n_chunks = gr._core.io_cache_stats(path)
        assert bytes_used <= budget_bytes
        assert n_chunks >= 1


def test_close_clears_chunk_cache(tmp_path):
    path = str(tmp_path / "df.gr")
    gr.save(_range_df(16), path, chunk_size=4)

    h = gr.open(path, cache="chunks", chunk_budget_mb=4)
    _ = h[0:4].to_dict()
    assert gr._core.io_cache_stats(path)[1] > 0
    h.close()

    with pytest.raises(ValueError, match="no active open cache"):
        gr._core.io_cache_stats(path)


def test_cache_none_rereads_each_time(tmp_path):
    path = str(tmp_path / "df.gr")
    gr.save(_range_df(16), path, chunk_size=4)

    with gr.open(path, cache="none") as h:
        gr._core.reset_io_bytes_read()
        _ = h[0:4].to_dict()
        first = gr._core.io_bytes_read()
        gr._core.reset_io_bytes_read()
        _ = h[0:4].to_dict()
        second = gr._core.io_bytes_read()
        assert first > 0
        assert second == first
