"""Tests for schema-level single-level indexing on dataframes."""

from __future__ import annotations

import pytest

import grumpy as gr


def _protein_df():
    """Two scenes, variable molecules per scene."""
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


def test_scalar_drill_down_chain():
    df = _protein_df()
    sub = df.scene[0].molecule[1]
    assert len(sub) == 1
    d = sub.to_dict()
    assert d["molecule_id"] == ["M1"]
    assert d["residue_name"] == ["C"]
    assert d["atom_number"] == [[4, 5, 6]]


def test_fancy_subset_single_level():
    df = _protein_df()
    sub = df.scene[[0, 1]]
    assert len(sub) == 2
    d = sub.to_dict()
    assert d["scene_id"] == ["S0", "S1"]
    assert d["molecule_id"] == [["M0", "M1"], ["M2"]]


def test_fancy_subset_root_equivalent():
    df = _protein_df()
    a = df[[0, 1]].to_dict()
    b = df.scene[[0, 1]].to_dict()
    assert a == b


def test_drill_down_scalar_equivalent():
    df = _protein_df()
    a = df.scene[0].molecule[1].to_dict()
    b = df.scene[0].molecule[1].to_dict()
    assert a == b


def test_multi_tuple_rejected():
    df = _protein_df()
    with pytest.raises(Exception, match="grumpy\\."):
        _ = df[0, 1]


def test_nested_list_batch_rejected():
    df = _protein_df()
    with pytest.raises(Exception, match="grumpy\\."):
        _ = df[[[0, 1], [0, 0]], [[1, 0], [1, 0]]]


def test_colon_and_ellipsis_rejected():
    df = _protein_df()
    with pytest.raises(Exception, match="grumpy\\."):
        _ = df[0, :, 0]
    with pytest.raises(Exception, match="grumpy\\."):
        _ = df[1, ..., 0]


def test_axis0_slice_and_bool_mask_unchanged():
    df = _protein_df()
    assert df[:1].to_dict()["scene_id"] == ["S0"]
    assert df[[True, False]].to_dict()["scene_id"] == ["S0"]


def test_schema_indexing_requires_schema_for_multi_tuple():
    df = gr.dataframe({"a": [1, 2, 3]})
    with pytest.raises(Exception, match="grumpy\\."):
        _ = df[0, 1]


def test_molecule_level_schema_without_scene():
    df = gr.dataframe(
        {
            "molecule_id": ["one", "two"],
            "residue_name": [["A", "B", "C"], ["D", "E"]],
            "atom_number": [[[1, 2], [3, 4, 5], [6]], [[7, 8], [9]]],
        },
        schema=["molecule", ("residue", "group"), "atom"],
    )
    sub = df.molecule[0].residue[1]
    assert len(sub) == 1
    assert sub.to_dict()["residue_name"] == ["B"]
    assert sub.to_dict()["atom_number"] == [[3, 4, 5]]


def test_skip_level_on_root_rejected():
    df = _protein_df()
    with pytest.raises(Exception, match="grumpy\\."):
        _ = df.molecule[0]


def test_column_select_unchanged():
    df = _protein_df()
    sub = df["scene_id", "molecule_id"]
    assert list(sub.to_dict().keys()) == ["scene_id", "molecule_id"]
