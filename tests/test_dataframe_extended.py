import pytest
import grumpy as gr


def make_df():
    return gr.dataframe(
        {
            "molecule_id": ["one", "two", "three"],
            "residue_name": [["A", "B"], ["C"], ["D", "E", "F"]],
            "atom_number": [[[1], [2, 3]], [[4, 5]], [[6], [7], [8, 9]]],
        },
        schema=["molecule", "residue", "atom"],
    )


def test_dataframe_row_slice_and_bool_filter():
    df = make_df()
    assert df[:2].to_dict()["molecule_id"] == ["one", "two"]
    df2 = df[[True, False, True]]
    assert df2.to_dict()["molecule_id"] == ["one", "three"]


def test_dataframe_column_selection_and_order():
    df = make_df()
    sub = df["residue_name", "molecule_id"]
    d = sub.to_dict()
    assert list(d.keys()) == ["residue_name", "molecule_id"]
    assert d["molecule_id"] == ["one", "two", "three"]


def test_dataframe_dot_notation_chaining_and_assignment():
    df = make_df()
    # flatten all axes by default
    assert df.atom_number.to_list() == [1, 2, 3, 4, 5, 6, 7, 8, 9]
    # keep residue->atom
    assert df.residue.atom_number.to_list() == [[1], [2, 3], [4, 5], [6], [7], [8, 9]]

    # Assign residue-level data as a flat-by-residue vector, re-nested per-molecule.
    df.residue.residue_weight = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    assert df["residue_weight"].to_dict()["residue_weight"] == [[0.1, 0.2], [0.3], [0.4, 0.5, 0.6]]


def test_dataframe_schema_rejects_bad_prefix():
    df = make_df()
    with pytest.raises(ValueError, match="does not start with any valid schema prefix"):
        df["badprefix_col"] = [1, 2, 3]


