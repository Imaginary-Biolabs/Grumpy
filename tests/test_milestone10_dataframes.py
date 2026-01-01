import pytest

import grumpy as gr


def test_dataframe_basic_construction_and_to_dict():
    df = gr.dataframe({"a": [1, 2, 3], "b": [4, [5, 6]]})
    d = df.to_dict()
    assert d["a"] == [1, 2, 3]
    assert d["b"] == [4, [5, 6]]


def test_dataframe_row_slice_and_bool_mask():
    df = gr.dataframe({"a": [1, 2, 3], "b": [4, [5, 6], 7]})
    d2 = df[:2].to_dict()
    assert d2["a"] == [1, 2]
    assert d2["b"] == [4, [5, 6]]

    d3 = df[[True, False, True]].to_dict()
    assert d3["a"] == [1, 3]
    assert d3["b"] == [4, 7]


def test_dataframe_column_subset_by_string_and_tuple():
    df = gr.dataframe({"a": [1, 2, 3], "b": [4, 5, 6]})
    assert df["a"].to_dict() == {"a": [1, 2, 3]}
    assert df["b", "a"].to_dict() == {"b": [4, 5, 6], "a": [1, 2, 3]}


def test_dataframe_max_applies_to_all_columns_numeric_only():
    df = gr.dataframe({"a": [1, 2, 3], "b": [4, [5, 6]]})
    out = df.max()
    assert out["a"] == 3
    assert out["b"] == 6


def test_dataframe_dot_notation_get_and_set():
    df = gr.dataframe(
        {
            "molecule_id": ["one", "two"],
            "residue_name": [["A", "B", "C"], ["D", "E"]],
            "atom_number": [[[1, 2], [3, 4, 5], [6]], [[7, 8], [9]]],
        },
        schema=["molecule", ("residue", "group"), "atom"],
    )
    # df.atom_number fully flattens
    assert df.atom_number.to_list() == [1, 2, 3, 4, 5, 6, 7, 8, 9]
    # df.residue.atom_number flattens molecules, keeps residue->atom
    assert df.residue.atom_number.to_list() == [[1, 2], [3, 4, 5], [6], [7, 8], [9]]

    # residue-level assignment: provide flat-by-residue vector, stored as molecule->residue
    df.residue.residue_weight = [0.5, 0.7, 0.8, 0.9, 1.0]
    assert df["residue_weight"].to_dict()["residue_weight"] == [[0.5, 0.7, 0.8], [0.9, 1.0]]


def test_dataframe_schema_prefix_and_shape_constraints_on_setitem():
    df = gr.dataframe(
        {
            "molecule_id": ["one", "two"],
            "residue_name": [["A", "B", "C"], ["D", "E"]],
            "atom_number": [[[1, 2], [3, 4, 5], [6]], [[7, 8], [9]]],
        },
        schema=["molecule", ("residue", "group"), "atom"],
    )
    # invalid prefix
    with pytest.raises(Exception):
        df["foo_weight"] = [0.5, 0.6]
    # invalid length
    with pytest.raises(Exception):
        df["molecule_weight"] = [0.5]
    # ok after filtering to one molecule
    df2 = df[[True, False]]
    df2["molecule_weight"] = [0.5]
    assert df2["molecule_weight"].to_dict()["molecule_weight"] == [0.5]


