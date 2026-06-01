"""Partial I/O for union-root layouts (no full scalar/list pool reads)."""

from __future__ import annotations

import grumpy as gr


def _large_union():
    big = list(range(500))
    return gr.array([1, big, 2, big, 3, big, 4, big], dtype=gr.int64)


def test_union_partial_io_single_list_row(tmp_path):
    x = _large_union()
    path = str(tmp_path / "union.gr")
    gr.save(x, path, chunk_size=128)

    gr._core.reset_io_bytes_read()
    one = gr._core.load_slice(path, 1, 2)
    partial_bytes = gr._core.io_bytes_read()

    gr._core.reset_io_bytes_read()
    full = gr._core.load_slice(path, 0, 8)
    full_bytes = gr._core.io_bytes_read()

    assert one.to_list() == full[1:2].to_list()
    assert partial_bytes > 0
    assert partial_bytes < full_bytes


def test_union_partial_io_scalar_row(tmp_path):
    x = _large_union()
    path = str(tmp_path / "union.gr")
    gr.save(x, path, chunk_size=128)

    gr._core.reset_io_bytes_read()
    scalar = gr._core.load_slice(path, 0, 1)
    partial_bytes = gr._core.io_bytes_read()

    gr._core.reset_io_bytes_read()
    gr._core.load_slice(path, 1, 2)
    list_bytes = gr._core.io_bytes_read()

    assert scalar.to_list() == [1]
    assert partial_bytes > 0
    assert partial_bytes < list_bytes


def test_union_partial_io_dataframe_column(tmp_path):
    big = list(range(400))
    df = gr.dataframe(
        {
            "molecule_id": [1, big, 2, big],
            "molecule_val": [10.0, [float(i) for i in big], 20.0, [float(i) for i in big]],
        },
        schema=["molecule"],
    )
    path = str(tmp_path / "df.gr")
    gr.save(df, path, chunk_size=64)

    gr._core.reset_io_bytes_read()
    batch = gr._core.load_slice(path, 1, 2)
    partial_bytes = gr._core.io_bytes_read()

    gr._core.reset_io_bytes_read()
    full = gr._core.load_slice(path, 0, 4)
    full_bytes = gr._core.io_bytes_read()

    assert batch.to_dict()["molecule_id"] == full[1:2].to_dict()["molecule_id"]
    assert partial_bytes > 0
    assert partial_bytes < full_bytes
