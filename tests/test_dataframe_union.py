"""DataFrame schema validation and dot-notation with UnionScalarList columns."""

from __future__ import annotations

import pytest

import grumpy as gr


def test_dataframe_union_schema_construction():
    df = gr.dataframe(
        {
            "molecule_id": [1, [2, 3], 4],
            "molecule_name": ["a", "b", "c"],
        },
        schema=["molecule"],
    )
    assert df.to_dict()["molecule_id"] == [1, [2, 3], 4]
    assert df.to_dict()["molecule_name"] == ["a", "b", "c"]


def test_dataframe_union_schema_rejects_bad_length():
    df = gr.dataframe(
        {"molecule_id": [1, [2, 3], 4]},
        schema=["molecule"],
    )
    with pytest.raises(ValueError, match="length"):
        df["molecule_weight"] = [0.5, 0.6]


def test_dataframe_union_schema_rejects_bad_prefix():
    df = gr.dataframe(
        {"molecule_id": [1, [2, 3], 4]},
        schema=["molecule"],
    )
    with pytest.raises(ValueError, match="does not start with any valid schema prefix"):
        df["foo_col"] = [1, 2, 3]


def test_dataframe_union_dot_notation_flatten():
    df = gr.dataframe(
        {
            "molecule_id": [1, [2, 3], 4],
            "molecule_val": [10, [20, 30], 40],
        },
        schema=["molecule"],
    )
    assert df.molecule_id.to_list() == [1, 2, 3, 4]
    assert df.molecule_val.to_list() == [10, 20, 30, 40]


def test_dataframe_union_dot_notation_level_and_assignment():
    df = gr.dataframe(
        {
            "molecule_id": [1, [2, 3], 4],
            "residue_name": [["A", "B"], ["C", "D", "E"], ["F"]],
        },
        schema=["molecule", "residue"],
    )
    # molecule level: union preserved
    assert df.molecule.molecule_id.to_list() == [1, [2, 3], 4]
    # residue level: peel molecule axis from list-chain column
    assert df.residue.residue_name.to_list() == ["A", "B", "C", "D", "E", "F"]

    df2 = df[[True, False, True]]
    df2["molecule_weight"] = [0.5, 0.9]
    assert df2["molecule_weight"].to_dict()["molecule_weight"] == [0.5, 0.9]
