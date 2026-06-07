"""Tests for gr.open lazy handles, canon metadata, and dataframe.shape."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import grumpy as gr


def _protein_df():
    return gr.dataframe(
        {
            "scene_id": ["S0", "S1"],
            "molecule_id": [["M0", "M1"], ["M2"]],
            "residue_name": [
                [["A", "B"], ["C"]],
                [["D", "E"]],
            ],
            "atom_number": [
                [[[1, 2], [3]], [[4, 5, 6]]],
                [[[7, 8], [9]]],
            ],
        },
        schema=["scene", "molecule", "residue", "atom"],
    )


def test_canon_persisted_in_metadata(tmp_path):
    df = _protein_df()
    path = str(tmp_path / "df.gr")
    gr.save(df, path)
    meta = json.loads(Path(path).joinpath("grumpy.json").read_text())
    assert "canon" in meta["root"]
    assert meta["root"]["canon"]["nrows"] == 2
    assert len(meta["root"]["canon"]["offsets"]) == 4


def test_dataframe_shape_from_canon(tmp_path):
    df = _protein_df()
    path = str(tmp_path / "df.gr")
    gr.save(df, path)
    loaded = gr.load(path)
    assert loaded.shape(dim=0) == 2
    assert loaded.shape(dim="scene") == 2
    mol_shape = loaded.shape(dim=1).to_list()
    assert mol_shape == [2, 1]
    assert loaded.shape(dim="molecule").to_list() == mol_shape
    sub = loaded.scene[0]
    assert sub.shape(dim=0) == 2
    assert sub.shape(dim="molecule") == 2
    assert sub.shape(dim=1).to_list() == [2, 1]


def test_open_repr_and_len(tmp_path):
    df = _protein_df()
    path = str(tmp_path / "df.gr")
    gr.save(df, path)
    h = gr.open(path)
    assert "OpenDataFrame" in repr(h)
    assert len(h) == 2


def test_open_shape_without_column_io(tmp_path):
    df = _protein_df()
    path = str(tmp_path / "df.gr")
    gr.save(df, path)
    h = gr.open(path)
    gr._core.reset_io_bytes_read()
    assert h.shape(dim=0) == 2
    assert h.shape(dim="molecule").to_list() == [2, 1]
    assert gr._core.io_bytes_read() == 0


def test_open_row_index_materializes(tmp_path):
    df = _protein_df()
    path = str(tmp_path / "df.gr")
    gr.save(df, path, chunk_size=1)
    h = gr.open(path)
    sub = h[[0, 1]]
    assert type(sub).__name__ == "GrumpyDataFrame"
    assert sub.to_dict() == df[[0, 1]].to_dict()


def test_open_accessor_index_materializes(tmp_path):
    df = _protein_df()
    path = str(tmp_path / "df.gr")
    gr.save(df, path, chunk_size=1)
    h = gr.open(path)
    sub = h.scene[0].molecule[1]
    assert type(sub).__name__ == "GrumpyDataFrame"
    assert sub.to_dict() == df.scene[0].molecule[1].to_dict()


def test_open_column_proxy(tmp_path):
    df = _protein_df()
    path = str(tmp_path / "df.gr")
    gr.save(df, path, chunk_size=1)
    h = gr.open(path)
    col = h.atom_number
    assert type(col).__name__ == "OpenColumn"
    assert col[0].to_list() == df.atom_number[0].to_list()


def test_open_column_bracket_select(tmp_path):
    df = _protein_df()
    path = str(tmp_path / "df.gr")
    gr.save(df, path)
    h = gr.open(path)
    col = h["residue_name"]
    assert type(col).__name__ == "OpenColumn"
    assert col[[0]].to_list() == df["residue_name"][[0]].to_dict()["residue_name"]


def test_open_load_full(tmp_path):
    df = _protein_df()
    path = str(tmp_path / "df.gr")
    gr.save(df, path)
    h = gr.open(path)
    full = h.load()
    assert full.to_dict() == df.to_dict()


def test_open_close_and_context_manager(tmp_path):
    df = _protein_df()
    path = str(tmp_path / "df.gr")
    gr.save(df, path)

    h = gr.open(path)
    assert not h.closed
    h.close()
    assert h.closed
    assert "closed" in repr(h)
    with pytest.raises(ValueError, match="IoFailed|is closed"):
        h.load()

    with gr.open(path) as handle:
        assert not handle.closed
        sub = handle.scene[0]
        assert sub.to_dict() == df.scene[0].to_dict()
    assert handle.closed


def test_open_column_fails_after_close(tmp_path):
    df = _protein_df()
    path = str(tmp_path / "df.gr")
    gr.save(df, path)
    h = gr.open(path)
    col = h.atom_number
    h.close()
    with pytest.raises(ValueError, match="IoFailed|is closed"):
        col[0]


def test_open_partial_io(tmp_path):
    df = _protein_df()
    path = str(tmp_path / "df.gr")
    gr.save(df, path, chunk_size=1)
    gr._core.reset_io_bytes_read()
    gr._core.clear_path_caches()
    h = gr.open(path)
    _ = h.scene[0].to_dict()
    partial = gr._core.io_bytes_read()
    gr._core.reset_io_bytes_read()
    gr._core.clear_path_caches()
    gr.load(path)
    full = gr._core.io_bytes_read()
    assert 0 < partial < full
